"""DataFeed protocol + FeedHealth (docs/architecture.md section 3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Protocol

from trading.core.types import Bar


@dataclass
class FeedHealth:
    last_bar_at: Optional[datetime] = None  # UTC start of last bar yielded
    gap_count: int = 0                      # missing intraday bars detected
    lag_ms: float = 0.0                     # live: bar close -> receipt delay


class DataFeed(Protocol):
    def subscribe(self, symbol: str, bar_seconds: int) -> None: ...

    def bars(self) -> Iterator[Bar]:
        """Blocking iterator over COMPLETE bars, in time order.

        - HistoricalFeed: replays stored bars; the virtual clock follows.
        - LiveFeed: yields each bar once it has closed.
        Iterator exhaustion = data source drained (backtest) or stop signal
        (live). Gaps are never silently skipped: they increment
        ``health().gap_count`` so risk/strategy layers can react.
        """
        ...

    def health(self) -> FeedHealth: ...
