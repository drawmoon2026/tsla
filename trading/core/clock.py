"""Clock abstraction: virtual time in backtest, wall time otherwise.

Convention: ``now()`` returns tz-aware UTC. In backtest the Runner advances
the SimClock to each bar's END time (start + duration) before any component
asks for the time, so "now" is always the moment the latest bar completed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SimClock:
    """Data-driven virtual clock; the Runner calls ``advance`` per bar."""

    def __init__(self) -> None:
        self._now: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def advance(self, ts: datetime) -> None:
        if ts < self._now:
            raise ValueError(f"SimClock cannot go backwards: {ts} < {self._now}")
        self._now = ts

    def now(self) -> datetime:
        return self._now


class WallClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)
