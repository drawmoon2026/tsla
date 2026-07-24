"""Position state machine + account bookkeeping (architecture section 6).

State machine per symbol:

    FLAT --entry submitted--> PENDING_ENTRY --fill--> HOLDING
    PENDING_ENTRY --cancel/reject--> FLAT
    HOLDING --MOC exit submitted--> PENDING_EXIT --fill--> FLAT
    HOLDING --TP/SL bracket fill--> FLAT
    (bracket orders resting is part of HOLDING, not PENDING_EXIT)

``apply_fill`` is the ONLY entry point that mutates the position/cash — the
whole account state is reconstructable from the Fill journal. ``on_bar``
emits forced-exit orders (TTL deadline, end-of-day 15:55 ET flatten) as
market-on-close orders (tif="cls": fill at the current bar's close, matching
run_sim's TTL exit at the last 5m close of the execution window).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from trading.core.types import (
    AccountState, Bar, Fill, Order, OrderSide, OrderType, Position,
)

_ET = ZoneInfo("America/New_York")


class PositionState(Enum):
    FLAT = "flat"
    PENDING_ENTRY = "pending_entry"
    HOLDING = "holding"
    PENDING_EXIT = "pending_exit"

    def allows(self, order: Order, pos_qty: float = 0.0) -> bool:
        """May `order` be submitted in this state? Entries only from FLAT;
        position-reducing orders only while HOLDING."""
        if self is PositionState.FLAT:
            return True
        if self is PositionState.HOLDING:
            order_sign = 1.0 if order.side is OrderSide.BUY else -1.0
            return pos_qty * order_sign < 0  # reduces the position
        return False


class Portfolio:
    """Single-symbol account (extension point: dict of per-symbol machines)."""

    def __init__(self, symbol: str, cash: float,
                 eod_flat_et: tuple[int, int] = (15, 55),
                 entry_timeout_bars: int = 3) -> None:
        self.symbol = symbol
        self.cash = cash
        self._pos = Position(symbol=symbol)
        self._state = PositionState.FLAT
        self._last_price: Optional[float] = None
        self._ttl_deadline: Optional[datetime] = None
        self._eod_flat_et = eod_flat_et
        # PENDING_ENTRY --timeout--> FLAT: an entry unfilled after this many
        # bars is abandoned (NullBroker in shadow mode never fills; SimBroker
        # market orders fill on the next bar, so backtest is unaffected).
        # Live: the runner must also cancel the stale order at the broker.
        self._entry_timeout_bars = entry_timeout_bars
        self._entry_age = 0

    # -- read side ---------------------------------------------------------
    def state(self, symbol: str) -> PositionState:
        return self._state

    def position(self, symbol: str) -> Position:
        return self._pos

    def snapshot(self) -> AccountState:
        mark = self._last_price if self._last_price is not None else self._pos.avg_price
        equity = self.cash + self._pos.qty * mark
        return AccountState(cash=self.cash, equity=equity,
                            positions={self.symbol: self._pos})

    # -- write side --------------------------------------------------------
    def plan_exit(self, symbol: str, ttl_deadline: Optional[datetime]) -> None:
        """Attach the forced-exit deadline for the open trade (set by the
        engine at entry; metadata only, not a position mutation)."""
        self._ttl_deadline = ttl_deadline

    def on_order_submitted(self, order: Order) -> None:
        """State transitions on submit. Brackets (TP/SL) keep HOLDING."""
        if self._state is PositionState.FLAT:
            self._state = PositionState.PENDING_ENTRY
            self._entry_age = 0
        elif self._state is PositionState.HOLDING and order.tif == "cls":
            self._state = PositionState.PENDING_EXIT

    def apply_fill(self, fill: Fill) -> None:
        """The single position/cash mutation entry point."""
        was_flat = self._pos.qty == 0
        self.cash -= fill.qty * fill.price
        self.cash -= fill.fee
        if was_flat:
            self._pos.avg_price = fill.price
        else:
            closed = min(abs(fill.qty), abs(self._pos.qty))
            direction = 1.0 if self._pos.qty > 0 else -1.0
            self._pos.realized_pnl += closed * direction * (fill.price - self._pos.avg_price)
        self._pos.qty += fill.qty
        if self._pos.qty == 0:
            self._state = PositionState.FLAT
            self._ttl_deadline = None
        else:
            self._state = PositionState.HOLDING

    def on_bar(self, bar: Bar) -> list[Order]:
        """TTL / end-of-day forced-exit check. Returns market-on-close exit
        orders; the engine journals and submits them."""
        self._last_price = bar.close
        if self._state is PositionState.PENDING_ENTRY:
            self._entry_age += 1
            if self._entry_age >= self._entry_timeout_bars:
                self._state = PositionState.FLAT   # abandon stale entry
        if self._state is not PositionState.HOLDING or self._pos.qty == 0:
            return []
        bar_end = bar.start + timedelta(seconds=bar.duration_s)
        et_end = bar_end.astimezone(_ET)
        eod = et_end.replace(hour=self._eod_flat_et[0], minute=self._eod_flat_et[1],
                             second=0, microsecond=0)
        tag = None
        if self._ttl_deadline is not None and bar_end >= self._ttl_deadline:
            tag = "ttl_close"
        elif et_end >= eod:
            tag = "eod"
        if tag is None:
            return []
        return [Order(
            id=Order.new_id(),
            symbol=self.symbol,
            side=OrderSide.SELL if self._pos.qty > 0 else OrderSide.BUY,
            qty=abs(self._pos.qty),
            type=OrderType.MARKET,
            tif="cls",
            notional=abs(self._pos.qty) * self._pos.avg_price,  # fee ref = entry notional
            tag=tag,
        )]
