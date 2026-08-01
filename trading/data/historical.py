"""HistoricalFeed: CSV replay through src.common.data_io.load_bars.

Single place where price CSVs enter the trading system. ``load_bars``
enforces the conventions (UTC tz-aware index, bar-START stamps, sanity check
that bars fall in ET regular hours), so they are not re-implemented here.
"""

from __future__ import annotations

from typing import Iterator, Optional
from zoneinfo import ZoneInfo

from src.common.data_io import infer_bar_minutes, load_bars
from trading.core.types import Bar
from trading.data.feed import FeedHealth

_ET = ZoneInfo("America/New_York")


class HistoricalFeed:
    """Replays a 5m (or other intraday) OHLCV CSV as complete Bars.

    Gap accounting: within the same ET trading day, a jump of more than one
    bar duration between consecutive rows counts the missing bars into
    ``health().gap_count``. Overnight jumps are not gaps.
    """

    def __init__(self, path: str, symbol: str, bar_seconds: Optional[int] = None) -> None:
        self._df = load_bars(path)
        inferred = infer_bar_minutes(self._df) * 60
        if bar_seconds is not None and bar_seconds != inferred:
            raise ValueError(f"{path}: inferred {inferred}s bars, config says {bar_seconds}s")
        self._bar_seconds = inferred
        self._symbol = symbol
        self._health = FeedHealth()

    def subscribe(self, symbol: str, bar_seconds: int) -> None:
        if symbol != self._symbol:
            raise ValueError(f"feed loaded for {self._symbol}, subscribe asked {symbol}")
        if bar_seconds != self._bar_seconds:
            raise ValueError(f"feed has {self._bar_seconds}s bars, subscribe asked {bar_seconds}s")

    def bars(self) -> Iterator[Bar]:
        prev_ts = None
        for ts, row in self._df.iterrows():
            if prev_ts is not None:
                delta = (ts - prev_ts).total_seconds()
                same_day = ts.astimezone(_ET).date() == prev_ts.astimezone(_ET).date()
                if same_day and delta > self._bar_seconds:
                    self._health.gap_count += int(delta // self._bar_seconds) - 1
            prev_ts = ts
            self._health.last_bar_at = ts
            yield Bar(
                symbol=self._symbol,
                start=ts.to_pydatetime(),
                duration_s=self._bar_seconds,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
                n_ticks=1,
                complete=True,
            )

    def health(self) -> FeedHealth:
        return self._health
