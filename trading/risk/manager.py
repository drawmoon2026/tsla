"""RiskManager: sizing, daily loss stop, drawdown circuit breaker.

All sizing is in DOLLARS/SHARES (never unit-less multipliers). Two modes:

- ``risk_budget`` (default): shares = equity * per_trade_risk_pct
  / (sl_pct * price)  — inverse-volatility style: the loss if the SL is hit
  equals the risk budget. Capped by max_position_pct.
- ``full_equity``: dollar-notional order for the whole equity (capped by
  max_position_pct). This reproduces run_sim's 100%-of-equity compounding
  and is what the consistency check against live_trading/run_sim.py uses.

Circuit breakers are FRONT-loaded (checked in ``size`` before any order is
built) and the Halt state is persisted via the Store — a restart stays
halted until a human clears it. Check order mirrors run_sim exactly:
daily state reset -> drawdown halt -> daily block -> sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Union
from zoneinfo import ZoneInfo

from trading.core.clock import Clock
from trading.core.types import AccountState, Fill, Order, OrderSide, OrderType
from trading.strategy.base import SignalIntent

_ET = ZoneInfo("America/New_York")


@dataclass
class RiskLimits:
    max_position_pct: float = 0.30      # single-symbol notional <= equity * pct
    per_trade_risk_pct: float = 0.005   # loss at SL <= equity * pct
    daily_loss_stop_pct: float = 0.02   # stop opening new trades for the day
    max_drawdown_stop_pct: float = 0.10 # global halt (persisted)
    max_open_orders: int = 4
    worst_case_loss_pct: float = 0.05   # grid/martingale full-ladder loss cap
    sizing: str = "risk_budget"         # "risk_budget" | "full_equity"


@dataclass(frozen=True)
class Halt:
    reason: str


@dataclass(frozen=True)
class FlattenAll:
    reason: str


RiskAction = Union[Halt, FlattenAll]


class RiskManager:
    def __init__(
        self,
        limits: RiskLimits,
        clock: Clock,
        store=None,                    # persistence.store.Store | None
        initial_equity: float = 0.0,
        resume_halt: bool = False,     # True outside backtest: honour persisted halt
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._store = store
        self._peak = initial_equity
        self._day: Optional[date] = None
        self._day_start_equity = initial_equity
        self._day_blocked = False
        self._halted_reason: Optional[str] = None
        if resume_halt and store is not None:
            persisted = store.get_halt()
            if persisted:
                self._halted_reason = persisted

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    def _halt(self, reason: str) -> None:
        self._halted_reason = reason
        if self._store is not None:
            self._store.set_halt(self._clock.now(), reason)

    def size(self, intent: SignalIntent, account: AccountState,
             last_price: float) -> Optional[Order]:
        """Intent -> quantified Order, or None if any limit blocks it."""
        if intent.kind != "entry" or intent.direction == 0:
            return None
        equity = account.equity

        # daily state reset (ET calendar day of virtual/wall now)
        today = self._clock.now().astimezone(_ET).date()
        if today != self._day:
            self._day = today
            self._day_start_equity = equity
            self._day_blocked = False

        if self.halted:
            return None
        if self._peak > 0 and (equity - self._peak) / self._peak <= -self.limits.max_drawdown_stop_pct:
            self._halt(f"max drawdown stop: equity={equity:.2f} peak={self._peak:.2f}")
            return None
        if self._day_blocked:
            return None
        if last_price <= 0 or equity <= 0:
            return None

        cap_notional = equity * min(1.0, self.limits.max_position_pct)
        if self.limits.sizing == "full_equity":
            notional = cap_notional
            qty = notional / last_price     # indicative; broker resolves at fill
        else:
            if not intent.sl_pct or intent.sl_pct <= 0:
                return None
            qty = equity * self.limits.per_trade_risk_pct / (intent.sl_pct * last_price)
            qty = min(qty, cap_notional / last_price)
            notional = None
        if qty <= 0:
            return None

        return Order(
            id=Order.new_id(),
            symbol=intent.symbol,
            side=OrderSide.BUY if intent.direction > 0 else OrderSide.SELL,
            qty=qty,
            type=OrderType.MARKET,
            notional=notional,
            tag=intent.tag or "entry",
        )

    def on_fill(self, fill: Fill, account: AccountState) -> list[RiskAction]:
        """Post-fill review. Only realized (flat) equity trips the breakers,
        matching run_sim which evaluates per completed trade."""
        pos = account.positions.get(next(iter(account.positions), ""), None)
        if pos is not None and pos.qty != 0:
            return []
        equity = account.equity
        actions: list[RiskAction] = []
        if self._day_start_equity > 0 and \
                (equity - self._day_start_equity) / self._day_start_equity <= -self.limits.daily_loss_stop_pct:
            self._day_blocked = True
        self._peak = max(self._peak, equity)
        if self._peak > 0 and (equity - self._peak) / self._peak <= -self.limits.max_drawdown_stop_pct:
            self._halt(f"max drawdown stop after fill {fill.order_id}")
            actions.append(Halt(self._halted_reason or "drawdown"))
        return actions
