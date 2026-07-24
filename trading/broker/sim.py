"""SimBroker: the backtest fill model — all pessimistic assumptions live here.

Semantics are DELEGATED to src.common.execution (the audited single source
of truth used by every simulator in this repo):

- MARKET (tif="day"): fills at the NEXT bar's open with adverse slippage
  (``entry_fill``) — 1 bar of latency, never the signal bar's close.
- MARKET (tif="cls"): fills at the CURRENT bar's close, no slippage
  (TTL / end-of-day flatten; run_sim's "close" exit).
- STOP / LIMIT exit pairs are settled per bar through ``settle_bracket`` on
  a one-bar window: gap-through stops fill at the open (min/max of open and
  stop) with adverse slippage, take-profit limits need a strict crossing and
  never fill better than the open, and when one bar touches both levels the
  STOP wins (worst case for the trader).
- Fees: per side; charged on ``order.notional`` when set (the engine sets
  exit-order notional to the ENTRY notional, reproducing CostModel's
  ``ret = ... - 2 * fee`` on entry notional), else on fill notional.

Fill timestamps are the START of the bar that produced the fill, matching
the repo-wide bar-START stamp convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from src.common.execution import CostModel, entry_fill, settle_bracket
from trading.core.types import (
    AccountState, Bar, Fill, Order, OrderSide, OrderStatus, OrderType, Position,
)


@dataclass(frozen=True)
class FillModel:
    fee_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def cost(self) -> CostModel:
        return CostModel(fee_bp=self.fee_bps, slippage_bp=self.slippage_bps)


def _noop(*_args) -> None:
    pass


class SimBroker:
    def __init__(self, fill_model: FillModel) -> None:
        self._cost = fill_model.cost
        self._pending_market: list[Order] = []      # fill at next bar open
        self._resting: dict[str, Order] = {}        # stop/limit exits
        self._current: Optional[Bar] = None
        self._positions: dict[str, Position] = {}
        self._on_fill: Callable[[Fill], None] = _noop
        self._on_order_update: Callable[[Order], None] = _noop
        self._on_error: Callable[[Exception], None] = _noop

    def set_callbacks(self, on_fill, on_order_update, on_error) -> None:
        self._on_fill = on_fill
        self._on_order_update = on_order_update
        self._on_error = on_error

    # -- order entry -------------------------------------------------------
    def submit(self, order: Order) -> None:
        order.status = OrderStatus.SUBMITTED
        order.broker_id = order.id  # sim: local id == broker id
        if order.type is OrderType.MARKET and order.tif == "cls":
            if self._current is None:
                raise RuntimeError("MOC order before any bar")
            self._fill(order, self._current.close, self._current)
        elif order.type is OrderType.MARKET:
            self._pending_market.append(order)
        else:
            self._resting[order.id] = order

    def cancel(self, order_id: str) -> None:
        order = self._resting.pop(order_id, None)
        if order is None:
            order = next((o for o in self._pending_market if o.id == order_id), None)
            if order is not None:
                self._pending_market.remove(order)
        if order is not None:
            order.status = OrderStatus.CANCELLED
            self._on_order_update(order)

    # -- market data drive -------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        """Phase 1: fill queued market orders at this bar's open. Fill
        callbacks may submit bracket exits, which phase 2 then evaluates
        against this same bar (mirrors settle_bracket walking the window
        from the entry bar onwards)."""
        self._current = bar
        queued, self._pending_market = self._pending_market, []
        for order in queued:
            direction = 1 if order.side is OrderSide.BUY else -1
            self._fill(order, entry_fill(bar.open, direction, self._cost), bar)
        self._settle_brackets(bar)

    def _settle_brackets(self, bar: Bar) -> None:
        by_symbol: dict[str, list[Order]] = {}
        for o in self._resting.values():
            by_symbol.setdefault(o.symbol, []).append(o)
        for symbol, orders in by_symbol.items():
            if bar.symbol != symbol:
                continue
            sl = next((o for o in orders if o.type is OrderType.STOP), None)
            tp = next((o for o in orders if o.type is OrderType.LIMIT), None)
            ref = sl or tp
            if ref is None:
                continue
            # exit SELL closes a long (+1); exit BUY closes a short (-1)
            direction = 1 if ref.side is OrderSide.SELL else -1
            sl_px = sl.stop_price if sl else -math.inf * direction
            tp_px = tp.limit_price if tp else math.inf * direction
            window = pd.DataFrame(
                {"Open": [bar.open], "High": [bar.high],
                 "Low": [bar.low], "Close": [bar.close]},
                index=[bar.start],
            )
            res = settle_bracket(window, direction, entry_px=bar.open,
                                 tp_px=tp_px, sl_px=sl_px, cost=self._cost)
            winner = sl if res.hit == "sl" else tp if res.hit == "tp" else None
            if winner is None:
                continue
            loser = tp if winner is sl else sl
            self._resting.pop(winner.id, None)
            if loser is not None:
                self._resting.pop(loser.id, None)
                loser.status = OrderStatus.CANCELLED
                self._on_order_update(loser)
            self._fill(winner, res.exit_px, bar)

    # -- internals ---------------------------------------------------------
    def _fill(self, order: Order, price: float, bar: Bar) -> None:
        qty = (order.notional / price) if order.notional and order.type is OrderType.MARKET \
            and order.tif == "day" else order.qty
        signed = qty if order.side is OrderSide.BUY else -qty
        fee_ref = order.notional if order.notional else abs(qty) * price
        fee = self._cost.fee * fee_ref
        order.status = OrderStatus.FILLED
        pos = self._positions.setdefault(order.symbol, Position(symbol=order.symbol))
        if pos.qty == 0:
            pos.avg_price = price
        pos.qty += signed
        fill = Fill(order_id=order.id, ts=bar.start, qty=signed, price=price, fee=fee)
        self._on_order_update(order)
        self._on_fill(fill)

    # -- account views -----------------------------------------------------
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def account(self) -> AccountState:
        # portfolio is the source of truth in backtest; sim keeps positions only
        return AccountState(cash=0.0, equity=0.0, positions=self.positions())
