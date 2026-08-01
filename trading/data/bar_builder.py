"""Bar aggregation with explicit timestamp semantics.

Mirrors src.common.data_io.resample_bars exactly (same bucketing, same
completeness rule) but operates on ``Bar`` objects so the live path and the
backtest path share one implementation:

- Input ``Bar.start`` is the bar START (guaranteed by the type).
- Buckets are anchored to each ET trading day's FIRST bar (9:30 on full
  sessions), labelled by bucket START, closed="left".
- Buckets never span two trading days (overnight gaps can not leak into an
  "1H return").
- A bucket holding fewer than ``min_frac`` of the expected sub-bars is
  emitted with ``complete=False``; strategies must ignore incomplete buckets
  (resample_bars drops them; filtering ``complete`` gives the same set).
"""

from __future__ import annotations

from datetime import timedelta
from itertools import groupby
from zoneinfo import ZoneInfo

from trading.core.types import Bar

_ET = ZoneInfo("America/New_York")


def aggregate(bars: list[Bar], target_seconds: int, min_frac: float = 0.8) -> list[Bar]:
    """Aggregate intraday bars (e.g. 5m) into ``target_seconds`` buckets."""
    if not bars:
        return []
    sub_s = bars[0].duration_s
    if target_seconds < sub_s or target_seconds % sub_s:
        raise ValueError(f"cannot aggregate {sub_s}s bars into {target_seconds}s buckets")
    expected = target_seconds / sub_s

    out: list[Bar] = []
    for _, day_iter in groupby(bars, key=lambda b: b.start.astimezone(_ET).date()):
        day = list(day_iter)
        anchor = day[0].start
        buckets: dict[int, list[Bar]] = {}
        for b in day:
            idx = int((b.start - anchor).total_seconds() // target_seconds)
            buckets.setdefault(idx, []).append(b)
        for idx in sorted(buckets):
            mem = buckets[idx]
            out.append(Bar(
                symbol=mem[0].symbol,
                start=anchor + timedelta(seconds=idx * target_seconds),
                duration_s=target_seconds,
                open=mem[0].open,
                high=max(b.high for b in mem),
                low=min(b.low for b in mem),
                close=mem[-1].close,
                volume=sum(b.volume for b in mem),
                n_ticks=len(mem),
                complete=len(mem) >= min_frac * expected,
            ))
    return out
