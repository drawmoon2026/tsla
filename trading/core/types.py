"""Core data types shared by every layer (docs/architecture.md section 2).

Timestamp conventions (fixed at the type level, do not re-interpret):
- All datetimes are tz-aware UTC.
- ``Bar.start`` is the bar START time (yfinance convention); the bar covers
  ``[start, start + duration_s)``.
- ``Fill.ts`` is the START time of the bar in which the fill occurred
  (simulation) or the broker-reported fill time (live).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Bar:
    symbol: str
    start: datetime        # tz-aware UTC; bar covers [start, start + duration_s)
    duration_s: int        # 300 = 5m, 3600 = 1h
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_ticks: int = 0       # sub-bars/ticks aggregated into this bar
    complete: bool = True  # strategies must only consume complete bars


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order intent quantified in shares (or dollars via ``notional``).

    Extensions over the architecture sketch (both map to real Alpaca order
    features, and SimBroker needs them to reproduce the pessimistic model in
    src/common/execution.py):

    - ``notional``: dollar-sized order (Alpaca "notional" orders). When set,
      SimBroker resolves ``qty = notional / fill_price`` at fill time and
      charges fees on ``notional`` (for exit orders the engine sets it to the
      ENTRY notional so both sides pay fees on entry notional, exactly like
      ``CostModel``'s ``ret = ... - 2 * fee``).
    - ``tif``: "day" (default) or "cls" (market-on-close). A MARKET+"cls"
      order fills at the CURRENT bar close with no slippage (live: submitted
      just before bar close); a MARKET+"day" order fills at the NEXT bar open
      with adverse slippage (1-bar latency, pessimistic).
    """

    id: str
    symbol: str
    side: OrderSide
    qty: float             # shares, always positive; sign lives in `side`
    type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    broker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    tag: str = ""          # attribution: signal/bracket-leg that produced it
    notional: Optional[float] = None
    tif: str = "day"       # "day" | "cls"

    @staticmethod
    def new_id() -> str:
        """Local idempotent id (also sent as broker client_order_id)."""
        return uuid.uuid4().hex


@dataclass(frozen=True)
class Fill:
    order_id: str
    ts: datetime
    qty: float             # signed: buy > 0, sell < 0
    price: float           # actual fill price (slippage/gaps live here)
    fee: float


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class AccountState:
    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
