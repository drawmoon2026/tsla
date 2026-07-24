"""Strategy protocol (docs/architecture.md section 4).

``on_bar`` is purely functional: it reads the world through the read-only
StrategyContext and returns intents. It never places orders, never mutates
state, never knows a broker exists — so backtest and live run byte-identical
strategy logic. ``ctx.history()`` can only look BACK; look-ahead bias is
impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from trading.core.types import Bar, Position


@dataclass(frozen=True)
class SignalIntent:
    """Strategy output: an intent, not an order. Sizing is the risk layer's
    job, so there is deliberately no quantity here."""

    symbol: str
    direction: int              # +1 long / -1 short / 0 = flatten intent
    kind: str                   # "entry" | "exit" | "scale_in"
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    ttl_bars: Optional[int] = None   # forced exit after this many base bars
    confidence: float = 1.0     # mount point for a future GBDT filter
    tag: str = ""


class StrategyContext(Protocol):
    """Everything a strategy is allowed to see — read only."""

    def history(self, symbol: str, bar_seconds: int, n: int) -> list[Bar]: ...
    def position(self, symbol: str) -> Position: ...
    def now(self) -> datetime: ...


class Strategy(Protocol):
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> list[SignalIntent]: ...
    def warmup_bars(self) -> int: ...
