"""Runner: the single main loop shared by backtest/shadow/paper/live.

Per-bar pipeline (docs/architecture.md section 9):

    bar -> broker.on_bar (sim fills resting orders; live: no-op)
        -> portfolio.on_bar (TTL / end-of-day forced exits)
        -> risk.halted gate
        -> strategy.on_bar (pure, intents only)
        -> risk.size (intent -> order or None)
        -> position state machine allows?
        -> store.journal_order FIRST, then broker.submit

Fill callback chain (sync in sim, async in live):

    fill -> portfolio.apply_fill -> store.journal_fill
         -> risk.on_fill (circuit-breaker review)
         -> engine: submit TP/SL brackets on entry; close the trade record
            and cancel sibling exits on exit.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from trading.broker.base import Broker
from trading.broker.null import NullBroker
from trading.broker.sim import FillModel, SimBroker
from trading.config.schema import Config
from trading.core.clock import Clock, SimClock, WallClock
from trading.core.types import Bar, Fill, Order, OrderSide, OrderType, Position
from trading.data.bar_builder import aggregate
from trading.data.feed import DataFeed
from trading.data.historical import HistoricalFeed
from trading.persistence.store import Store
from trading.portfolio.portfolio import Portfolio
from trading.risk.manager import Halt, RiskLimits, RiskManager
from trading.strategy.base import SignalIntent, Strategy
from trading.strategy.v_reversal import VReversalStrategy


class _Context:
    """Read-only strategy view backed by the runner's bar buffer."""

    def __init__(self, runner: "Runner") -> None:
        self._r = runner

    def history(self, symbol: str, bar_seconds: int, n: int) -> list[Bar]:
        base = [b for b in self._r._bars if b.symbol == symbol]
        if bar_seconds == self._r.cfg.bar_seconds:
            return base[-n:]
        return aggregate(base, bar_seconds)[-n:]

    def position(self, symbol: str) -> Position:
        return self._r.portfolio.position(symbol)

    def now(self) -> datetime:
        return self._r.clock.now()


class Runner:
    def __init__(self, feed: DataFeed, strategy: Strategy, risk: RiskManager,
                 portfolio: Portfolio, broker: Broker, store: Store,
                 clock: Clock, cfg: Config) -> None:
        self.feed, self.strategy, self.risk = feed, strategy, risk
        self.portfolio, self.broker, self.store = portfolio, broker, store
        self.clock, self.cfg = clock, cfg
        self.ctx = _Context(self)
        self._bars: deque[Bar] = deque(maxlen=max(512, strategy.warmup_bars() * 4))
        self._orders: dict[str, Order] = {}
        self._pending_entries: dict[str, tuple[SignalIntent, Optional[datetime]]] = {}
        self._brackets: dict[str, list[str]] = {}   # symbol -> resting exit ids
        self._open_trade: Optional[dict] = None
        self.trades: list[dict] = []
        broker.set_callbacks(self._on_fill, self._on_order_update, self._on_error)

    # -- main loop ---------------------------------------------------------
    def run(self) -> pd.DataFrame:
        self.feed.subscribe(self.cfg.symbol, self.cfg.bar_seconds)
        for bar in self.feed.bars():
            if not bar.complete:
                continue
            if isinstance(self.clock, SimClock):
                self.clock.advance(bar.start + timedelta(seconds=bar.duration_s))
            self._bars.append(bar)
            if self.cfg.mode != "backtest":
                # forward-run journal: bar + feed-health snapshot at receipt
                # (shadow_report replays these for hypothetical fills)
                h = self.feed.health()
                self.store.journal_bar(bar, h.gap_count, h.lag_ms)

            # 0. simulated venue processes this bar (fills resting orders)
            self.broker.on_bar(bar)

            # 1. TTL / end-of-day forced exits
            for exit_order in self.portfolio.on_bar(bar):
                self._cancel_brackets(exit_order.symbol)
                self._submit(exit_order)

            acct = self.portfolio.snapshot()
            self.store.snapshot_equity(self.clock.now(), acct.cash, acct.equity)

            # 2. circuit breaker gate
            if self.risk.halted:
                continue

            # 3. pure strategy -> intents; 4. risk sizing; 5. state machine
            for intent in self.strategy.on_bar(bar, self.ctx):
                order = self.risk.size(intent, self.portfolio.snapshot(), bar.close)
                if order is None:
                    continue
                pos = self.portfolio.position(order.symbol)
                if not self.portfolio.state(order.symbol).allows(order, pos.qty):
                    continue
                deadline = None
                if intent.ttl_bars:
                    deadline = bar.start + timedelta(
                        seconds=bar.duration_s * (1 + intent.ttl_bars))
                    # = signal-bar end + ttl window
                self._pending_entries[order.id] = (intent, deadline)
                self._submit(order)
        return self._finalize()

    def _submit(self, order: Order) -> None:
        order.created_at = self.clock.now()
        self._orders[order.id] = order
        self.store.journal_order(order)          # journal-first, then wire
        self.portfolio.on_order_submitted(order)
        self.broker.submit(order)

    # -- broker callbacks --------------------------------------------------
    def _on_fill(self, fill: Fill) -> None:
        order = self._orders[fill.order_id]
        opening = self.portfolio.position(order.symbol).qty == 0
        eq_before = self.portfolio.snapshot().equity if opening else None

        self.portfolio.apply_fill(fill)
        self.store.journal_fill(fill)
        for action in self.risk.on_fill(fill, self.portfolio.snapshot()):
            if isinstance(action, Halt):
                print(f"[risk] HALT: {action.reason}")

        if opening:
            self._after_entry(order, fill, eq_before)
        else:
            self._after_exit(order, fill)

    def _after_entry(self, order: Order, fill: Fill, eq_before: float) -> None:
        intent, deadline = self._pending_entries.pop(order.id, (None, None))
        self.portfolio.plan_exit(order.symbol, deadline)
        d = 1 if fill.qty > 0 else -1
        self._open_trade = {
            "entry_time": fill.ts, "direction": d, "entry": fill.price,
            "eq_before": eq_before,
        }
        if intent is None:
            return
        qty = abs(fill.qty)
        exit_side = OrderSide.SELL if d > 0 else OrderSide.BUY
        entry_notional = qty * fill.price   # fee reference for BOTH exit legs
        legs = []
        if intent.tp_pct:
            legs.append(Order(id=Order.new_id(), symbol=order.symbol, side=exit_side,
                              qty=qty, type=OrderType.LIMIT,
                              limit_price=fill.price * (1 + intent.tp_pct * d),
                              notional=entry_notional, tag="tp"))
        if intent.sl_pct:
            legs.append(Order(id=Order.new_id(), symbol=order.symbol, side=exit_side,
                              qty=qty, type=OrderType.STOP,
                              stop_price=fill.price * (1 - intent.sl_pct * d),
                              notional=entry_notional, tag="sl"))
        self._brackets[order.symbol] = [o.id for o in legs]
        for leg in legs:
            self._submit(leg)

    def _after_exit(self, order: Order, fill: Fill) -> None:
        self._cancel_brackets(order.symbol, keep=order.id)
        eq_after = self.portfolio.snapshot().equity
        t = self._open_trade or {}
        hit = {"tp": "tp", "sl": "sl", "ttl_close": "close", "eod": "eod"}.get(order.tag, order.tag)
        eq_before = t.get("eq_before", eq_after)
        self.trades.append({
            "entry_time": t.get("entry_time"), "exit_time": fill.ts,
            "direction": t.get("direction"), "hit": hit,
            "entry": t.get("entry"), "exit": fill.price,
            "ret": eq_after / eq_before - 1 if eq_before else 0.0,
            "eq_before": eq_before, "eq_after": eq_after,
        })
        self._open_trade = None

    def _cancel_brackets(self, symbol: str, keep: Optional[str] = None) -> None:
        for oid in self._brackets.pop(symbol, []):
            if oid != keep:
                self.broker.cancel(oid)

    def _on_order_update(self, order: Order) -> None:
        self.store.journal_order(order)

    def _on_error(self, exc: Exception) -> None:
        print(f"[broker] error: {exc!r}")

    # -- outputs -----------------------------------------------------------
    def _finalize(self) -> pd.DataFrame:
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        trades = pd.DataFrame(self.trades)
        trades.to_csv(out / "trades.csv", index=False,
                      date_format="%Y-%m-%dT%H:%M:%S%z")
        equity = self.portfolio.snapshot().equity
        if len(trades):
            summary = (
                f"Trades: {len(trades)}, total return: {equity / self.cfg.capital - 1:.2%}, "
                f"win rate: {(trades['ret'] > 0).mean():.2%}, "
                f"tp/sl/close: {(trades['hit'] == 'tp').sum()}/"
                f"{(trades['hit'] == 'sl').sum()}/{(trades['hit'] == 'close').sum()}"
            )
        else:
            summary = "No trades generated."
        if self.risk.halted:
            summary += " | HALTED"
        summary += f" | feed gaps: {self.feed.health().gap_count}"
        (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
        print(summary)
        return trades


def build(mode: str, cfg: Config) -> Runner:
    """Wire a Runner for one of the four modes. Only the feed/broker/clock
    differ; strategy, risk, portfolio and persistence are identical."""
    cfg.mode = mode
    store = Store(cfg.resolved_db_path())
    if mode == "backtest":
        clock: Clock = SimClock()
        feed: DataFeed = HistoricalFeed(cfg.data_path, cfg.symbol, cfg.bar_seconds)
        broker: Broker = SimBroker(FillModel(cfg.fill.fee_bp, cfg.fill.slippage_bp))
    elif mode == "shadow":
        # NullBroker journals the signal stream for shadow-vs-backtest
        # diffing. Feed: CSV replay by default (backward compatible);
        # --live switches to real-time Alpaca REST polling (iex).
        clock = WallClock()
        if cfg.live:
            from trading.data.live_alpaca import AlpacaPollingFeed
            feed = AlpacaPollingFeed(
                cfg.symbol, cfg.bar_seconds,
                once_session=cfg.once_session, max_bars=cfg.max_bars,
            )
        else:
            feed = HistoricalFeed(cfg.data_path, cfg.symbol, cfg.bar_seconds)
        broker = NullBroker()
    elif mode in ("paper", "live"):
        from trading.broker.alpaca import AlpacaBroker
        clock = WallClock()
        broker = AlpacaBroker(cfg.alpaca_key_id, cfg.alpaca_secret_key,
                              paper=(mode == "paper"))  # raises: shell only
        raise NotImplementedError("live feed adapter not built in this phase")
    else:
        raise ValueError(f"unknown mode: {mode}")

    strategy = VReversalStrategy(
        symbol=cfg.symbol,
        trigger=cfg.strategy.trigger, tp=cfg.strategy.tp, sl=cfg.strategy.sl,
        allowed_hours=cfg.strategy.allowed_hours,
        signal_seconds=cfg.strategy.signal_seconds,
    )
    limits = RiskLimits(
        max_position_pct=cfg.risk.max_position_pct,
        per_trade_risk_pct=cfg.risk.per_trade_risk_pct,
        daily_loss_stop_pct=cfg.risk.daily_loss_stop_pct,
        max_drawdown_stop_pct=cfg.risk.max_drawdown_stop_pct,
        max_open_orders=cfg.risk.max_open_orders,
        worst_case_loss_pct=cfg.risk.worst_case_loss_pct,
        sizing=cfg.risk.sizing,
    )
    risk = RiskManager(limits, clock, store, initial_equity=cfg.capital,
                       resume_halt=(mode != "backtest"))
    portfolio = Portfolio(cfg.symbol, cfg.capital, eod_flat_et=cfg.eod_flat_et)
    runner = Runner(feed, strategy, risk, portfolio, broker, store, clock, cfg)
    store.start_run(Order.new_id(), clock.now(), mode, dataclasses.asdict(cfg))
    return runner
