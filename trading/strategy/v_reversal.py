"""1H breakout-follow strategy, migrated from live_trading/run_sim.py.

Signal: the intraday close-to-close return of the just-completed 1H bucket
(overnight gap excluded — the day's first bucket never signals). If
``|ret| >= trigger``, follow the move: direction = sign(ret). The trade is
expressed as an intent with TP/SL percentages and a TTL of one signal bucket
(exit at the close of the last base bar inside the next 1H window).

Replication notes vs run_sim.py (bit-for-bit trade parity):
- run_sim only trades when the EXECUTION bucket exists in the resampled 1H
  series (complete, adjacent to the signal bucket). Live we cannot see the
  future, so the causal equivalent is enforced instead: a full 1H execution
  window must fit inside the regular session (skips the partial 15:30–16:00
  remainder that resample_bars drops).
- The signal fires on the base bar whose END lands on a 1H bucket boundary
  (anchored to the day's first bar, i.e. 9:30). The entry MARKET order then
  fills at the NEXT bar's open — exactly run_sim's "enter at the first 5m
  open of the next 1H bucket".
"""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

from trading.core.types import Bar
from trading.data.bar_builder import aggregate
from trading.strategy.base import SignalIntent, StrategyContext

_ET = ZoneInfo("America/New_York")


class VReversalStrategy:
    def __init__(
        self,
        symbol: str,
        trigger: float = 0.015,
        tp: float = 0.03,
        sl: float = 0.01,
        allowed_hours: Sequence[int] = (9, 10, 11, 12, 15),
        signal_seconds: int = 3600,
        session_close_et: tuple[int, int] = (16, 0),
    ) -> None:
        if trigger < 0:
            raise ValueError("trigger must be >= 0 (negative means 'always trade')")
        self.symbol = symbol
        self.trigger = trigger
        self.tp = tp
        self.sl = sl
        self.allowed_hours = tuple(allowed_hours)
        self.signal_seconds = signal_seconds
        self.session_close_et = session_close_et

    def warmup_bars(self) -> int:
        # one full trading day of base bars is plenty: the signal only needs
        # the current ET day's history (overnight returns are excluded anyway)
        return 100

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> list[SignalIntent]:
        if bar.symbol != self.symbol or not bar.complete:
            return []
        t_end = bar.start + timedelta(seconds=bar.duration_s)

        hist = ctx.history(self.symbol, bar.duration_s, self.warmup_bars())
        day_date = bar.start.astimezone(_ET).date()
        day_bars = [b for b in hist if b.start.astimezone(_ET).date() == day_date]
        if not day_bars:
            return []

        # signal moment: this bar END lands on a bucket boundary (day-anchored)
        anchor = day_bars[0].start
        elapsed = (t_end - anchor).total_seconds()
        if elapsed <= 0 or elapsed % self.signal_seconds:
            return []

        # execution window [t_end, t_end + 1H) must fit inside the session —
        # causal stand-in for "execution bucket survives resample_bars"
        et_end = t_end.astimezone(_ET)
        session_close = et_end.replace(
            hour=self.session_close_et[0], minute=self.session_close_et[1],
            second=0, microsecond=0,
        )
        if et_end + timedelta(seconds=self.signal_seconds) > session_close:
            return []
        if et_end.hour not in self.allowed_hours:
            return []
        if ctx.position(self.symbol).qty != 0:
            return []

        # just-completed signal bucket + same-day predecessor (overnight-free)
        buckets = [b for b in aggregate(day_bars, self.signal_seconds) if b.complete]
        sig_start = t_end - timedelta(seconds=self.signal_seconds)
        idx = next((i for i, b in enumerate(buckets) if b.start == sig_start), None)
        if idx is None or idx == 0:
            return []
        ret = buckets[idx].close / buckets[idx - 1].close - 1
        if abs(ret) < self.trigger:
            return []
        direction = 1 if ret > 0 else -1

        return [SignalIntent(
            symbol=self.symbol,
            direction=direction,
            kind="entry",
            tp_pct=self.tp,
            sl_pct=self.sl,
            ttl_bars=self.signal_seconds // bar.duration_s,
            tag=f"vrev:{et_end:%Y-%m-%dT%H:%M}:ret={ret:+.4f}",
        )]
