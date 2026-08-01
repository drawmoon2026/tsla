"""AlpacaPollingFeed: real-time bars via Alpaca Market Data REST polling.

No websocket: for 5-minute bars a poll ~15s after each bucket closes is
plenty, needs only ``requests``, and survives laptop sleep / flaky networks
trivially (every poll is stateless; missed buckets are refetched by start
time). Credentials are reused from ``src.alpaca_data.load_keys()``
(.env / env vars, either ALPACA_* or APCA_* naming).

Data-feed caveat (IEX vs SIP — affects the 8-week forward validation):
    Free accounts get REAL-TIME data only from the ``iex`` feed (IEX
    exchange only, ~2% of consolidated volume). The ``sip`` feed
    (consolidated tape) that produced the backtest CSVs via
    src/alpaca_data.py is delayed 15 minutes for free accounts, which is
    useless for live 5m polling — so this feed uses **iex**, including for
    the startup backfill (one consistent feed within a run). Consequences:
    - OHLC can differ slightly from SIP (worst at open/close and in thin
      minutes); Volume is much smaller and NOT comparable to backtest.
    - Thinly traded 5m buckets can be entirely absent on IEX; such holes
      are reported honestly via ``health().gap_count``.
    Forward-run signals therefore live in a slightly different data regime
    than the SIP backtest — treat small trigger-boundary discrepancies in
    the shadow-vs-backtest diff as expected feed skew, not necessarily bugs.

Behavior:
- Startup backfill: the last ``backfill_days`` (default 2) distinct ET
  trading days present in the data, i.e. today's bars so far (strategy
  warmup) plus the full previous session (the overnight-excluding
  intraday-return logic needs a complete prior day boundary). Backfilled
  bars are yielded through the normal iterator, so the engine's history
  buffer fills exactly like in a backtest replay.
- Live: ~``poll_delay_s`` (default 15s) after each 5m bucket ends, GET
  /v2/stocks/{symbol}/bars from the expected bar start. HTTP/network errors
  retry with exponential backoff (capped). A bar that still has not
  appeared one full bucket after it was due is counted as a gap and
  skipped (never silently).
- After the session close (16:00 ET; 13:00 ET on NYSE half days — per-day
  close via intel.market_calendar, wired 2026-08-02 for ops-review P1-8):
  with ``once_session=True`` the iterator returns; otherwise it prints the
  next session open (next TRADING day 9:30 ET — weekends and NYSE holidays
  skipped, so holidays no longer produce all-day gap noise) and sleeps
  until then.
- Only complete regular-session bars (bar end <= now, 9:30 <= ET <
  session close) are yielded, with ``complete=True``, in start-time order.

``max_bars`` stops the iterator after N yielded bars (smoke tests /
bounded runs); ``stop()`` requests a graceful exit from another thread or
signal handler.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

import requests

from intel.market_calendar import (  # 统一 NYSE 交易日历（P1-8 接线 2026-08-02）
    close_time_et, is_trading_day, next_trading_day,
)
from src.alpaca_data import DATA_URL, load_keys
from trading.core.types import Bar
from trading.data.feed import FeedHealth

_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc

_TIMEFRAMES = {60: "1Min", 300: "5Min", 900: "15Min", 3600: "1Hour"}


def _now() -> datetime:
    return datetime.now(_UTC)


def _fmt_et(ts: datetime) -> str:
    return ts.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S ET")


class AlpacaPollingFeed:
    def __init__(
        self,
        symbol: str,
        bar_seconds: int = 300,
        *,
        feed: str = "iex",
        backfill_days: int = 2,
        backfill_lookback_days: int = 10,
        poll_delay_s: float = 15.0,
        once_session: bool = False,
        max_bars: Optional[int] = None,
        session_open_et: tuple[int, int] = (9, 30),
        session_close_et: tuple[int, int] = (16, 0),
        retry_poll_s: float = 10.0,
        max_backoff_s: float = 60.0,
    ) -> None:
        if bar_seconds not in _TIMEFRAMES:
            raise ValueError(f"unsupported bar_seconds={bar_seconds} (choose {sorted(_TIMEFRAMES)})")
        self._key, self._secret = load_keys()
        self._symbol = symbol.upper()
        self._bar_s = bar_seconds
        self._bar_td = timedelta(seconds=bar_seconds)
        self._feed = feed
        self._backfill_days = backfill_days
        self._lookback_days = backfill_lookback_days
        self._poll_delay = poll_delay_s
        self._once_session = once_session
        self._max_bars = max_bars
        self._open_et = session_open_et
        self._close_et = session_close_et
        self._retry_poll = retry_poll_s
        self._max_backoff = max_backoff_s

        self._health = FeedHealth()
        self._stop = False
        self._yielded = 0
        self._last_start: Optional[datetime] = None  # start of last yielded bar

    # -- DataFeed protocol -------------------------------------------------
    def subscribe(self, symbol: str, bar_seconds: int) -> None:
        if symbol.upper() != self._symbol:
            raise ValueError(f"feed built for {self._symbol}, subscribe asked {symbol}")
        if bar_seconds != self._bar_s:
            raise ValueError(f"feed has {self._bar_s}s bars, subscribe asked {bar_seconds}s")

    def health(self) -> FeedHealth:
        return self._health

    def stop(self) -> None:
        """Request a graceful exit; bars() returns within ~1s."""
        self._stop = True

    def bars(self) -> Iterator[Bar]:
        yield from self._backfill()
        if self._done():
            return
        yield from self._live_loop()

    # -- backfill ----------------------------------------------------------
    def _backfill(self) -> Iterator[Bar]:
        since = _now() - timedelta(days=self._lookback_days)
        fetched = self._fetch(since, max_tries=8)
        days = sorted({b.start.astimezone(_ET).date() for b in fetched})
        keep = set(days[-self._backfill_days:])
        bars = [b for b in fetched if b.start.astimezone(_ET).date() in keep]
        print(f"[live_alpaca] backfilled {len(bars)} bars "
              f"({', '.join(str(d) for d in sorted(keep)) or 'no data'}) via feed={self._feed}")
        for b in bars:
            emitted = self._emit(b, live=False)
            if emitted is not None:
                yield emitted
            if self._done():
                return

    # -- live polling ------------------------------------------------------
    def _live_loop(self) -> Iterator[Bar]:
        session_end: Optional[datetime] = None
        if self._once_session:
            d = _now().astimezone(_ET).date()
            _, close = self._open_close(d)
            # 原：d.weekday() < 5（假日误当交易日）；现走统一日历
            if is_trading_day(d) and _now() < close:
                session_end = close
            else:
                print("[live_alpaca] --once-session: market closed, stopping after backfill")
                return

        announced: Optional[datetime] = None
        while not self._stop and not self._done():
            expected = self._next_expected(self._last_start)
            if session_end is not None and expected + self._bar_td > session_end:
                print(f"[live_alpaca] session complete "
                      f"({_fmt_et(session_end)}) — stopping (--once-session)")
                return
            due = expected + self._bar_td + timedelta(seconds=self._poll_delay)
            now = _now()
            if now < due:
                if announced != expected:
                    if due - now > timedelta(minutes=2 * self._bar_s / 60):
                        print(f"[live_alpaca] market closed — next session bar "
                              f"{_fmt_et(expected)}, first poll {_fmt_et(due)}")
                    else:
                        print(f"[live_alpaca] waiting for bar {_fmt_et(expected)} "
                              f"(poll at {_fmt_et(due)})")
                    announced = expected
                self._sleep((due - now).total_seconds())
                continue

            got = self._fetch(expected, limit=200)
            new = [b for b in got
                   if self._last_start is None or b.start > self._last_start]
            new = [b for b in new if b.start >= expected]
            if new:
                for b in new:
                    emitted = self._emit(b, live=True)
                    if emitted is not None:
                        yield emitted
                    if self._done():
                        return
            else:
                if _now() >= due + self._bar_td:
                    # a whole extra bucket has passed: this bar isn't coming
                    self._health.gap_count += 1
                    print(f"[live_alpaca] bar {_fmt_et(expected)} never arrived "
                          f"(iex hole / halt / holiday) — counted as gap, skipping")
                    self._last_start = expected
                else:
                    self._sleep(self._retry_poll)

        if self._done():
            print(f"[live_alpaca] max_bars={self._max_bars} reached — stopping")

    # -- HTTP --------------------------------------------------------------
    def _fetch(self, start: datetime, limit: int = 1000,
               max_tries: Optional[int] = None) -> list[Bar]:
        """GET bars from `start` to now; complete RTH bars only, ascending.

        Retries HTTP/network errors with exponential backoff (2s..capped);
        raises after `max_tries` failures when set (backfill), else retries
        until stop() (live).
        """
        headers = {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}
        params: dict = {
            "timeframe": _TIMEFRAMES[self._bar_s],
            "start": start.astimezone(_UTC).isoformat(),
            "limit": min(limit, 10_000),
            "adjustment": "split",   # matches src/alpaca_data.py backtest CSVs
            "feed": self._feed,
            "sort": "asc",
        }
        backoff, tries = 2.0, 0
        rows: list[dict] = []
        while not self._stop:
            try:
                r = requests.get(DATA_URL.format(symbol=self._symbol),
                                 headers=headers, params=params, timeout=30)
                if r.status_code == 429:
                    self._sleep(3.0)
                    continue
                r.raise_for_status()
                payload = r.json()
                rows.extend(payload.get("bars") or [])
                token = payload.get("next_page_token")
                if token:
                    params["page_token"] = token
                    continue
                return self._to_bars(rows)
            except requests.RequestException as exc:
                tries += 1
                if max_tries is not None and tries >= max_tries:
                    raise RuntimeError(
                        f"Alpaca fetch failed {tries}x, giving up: {exc!r}") from exc
                print(f"[live_alpaca] fetch error ({exc!r}) — retrying in {backoff:.0f}s")
                self._sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
        return []

    def _to_bars(self, rows: list[dict]) -> list[Bar]:
        now = _now()
        out: list[Bar] = []
        for row in rows:
            ts = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_UTC)
            if ts + self._bar_td > now:            # in-progress bucket: not complete
                continue
            et = ts.astimezone(_ET)
            # RTH 过滤：收盘时刻按当日日历（半日市 13:00，P1-8）。原口径：
            # et.hour < self._close_et[0]（固定 16:00，半日市会放进午后场外 bar）
            close_t = min(dtime(*self._close_et), close_time_et(et.date()))
            in_rth = (dtime(et.hour, et.minute) >= dtime(*self._open_et)
                      and dtime(et.hour, et.minute) < close_t)
            if not in_rth:
                continue
            out.append(Bar(
                symbol=self._symbol,
                start=ts.astimezone(_UTC),
                duration_s=self._bar_s,
                open=float(row["o"]), high=float(row["h"]),
                low=float(row["l"]), close=float(row["c"]),
                volume=float(row.get("v", 0.0)),
                n_ticks=int(row.get("n", 0)),
                complete=True,
            ))
        out.sort(key=lambda b: b.start)
        return out

    # -- bookkeeping -------------------------------------------------------
    def _emit(self, bar: Bar, live: bool) -> Optional[Bar]:
        """Account gaps/health for `bar`; return it if it should be yielded."""
        if self._last_start is not None and bar.start <= self._last_start:
            return None                            # duplicate / already skipped
        if self._last_start is not None:
            self._health.gap_count += self._gaps_between(self._last_start, bar.start)
        self._last_start = bar.start
        self._health.last_bar_at = bar.start
        if live:
            self._health.lag_ms = (_now() - (bar.start + self._bar_td)).total_seconds() * 1000
        self._yielded += 1
        return bar

    def _gaps_between(self, prev_start: datetime, actual_start: datetime) -> int:
        """Missing session buckets strictly between two bar starts, counted
        on the ACTUAL bar's ET day only (overnight rolls are not gaps —
        same rule as HistoricalFeed)."""
        actual_day = actual_start.astimezone(_ET).date()
        exp, n, guard = self._next_expected(prev_start), 0, 0
        while exp < actual_start and guard < 10_000:
            if exp.astimezone(_ET).date() == actual_day:
                n += 1
            exp = self._next_expected(exp)
            guard += 1
        return n

    def _done(self) -> bool:
        return self._stop or (self._max_bars is not None and self._yielded >= self._max_bars)

    def _sleep(self, seconds: float) -> None:
        import time as _time
        end = _time.monotonic() + seconds
        while not self._stop:
            remaining = end - _time.monotonic()
            if remaining <= 0:
                return
            _time.sleep(min(1.0, remaining))

    # -- session calendar (unified NYSE calendar: holidays + half days) ----
    def _open_close(self, d: date) -> tuple[datetime, datetime]:
        o = datetime.combine(d, dtime(*self._open_et), tzinfo=_ET)
        # 半日市 13:00 收盘（P1-8）：按日历取当日收盘；原为固定 self._close_et。
        # 取 min 以保留调用方显式传入的更早收盘（测试/特殊会话）。
        c = datetime.combine(d, min(dtime(*self._close_et), close_time_et(d)),
                             tzinfo=_ET)
        return o.astimezone(_UTC), c.astimezone(_UTC)

    @staticmethod
    def _next_weekday(d: date) -> date:
        # 原：仅跳周末（假日整日按 gap 计噪音）；现走统一日历跳周末+假日
        return next_trading_day(d)

    def _next_expected(self, after: Optional[datetime]) -> datetime:
        """Next expected session bar START strictly after bar-start `after`
        (or the bucket relevant right now when `after` is None)."""
        if after is None:
            now = _now()
            d = now.astimezone(_ET).date()
            open_, close = self._open_close(d)
            # 原：d.weekday() >= 5；现假日同样直接滚到下一交易日
            if not is_trading_day(d) or now >= close:
                return self._open_close(self._next_weekday(d))[0]
            if now < open_:
                return open_
            k = int((now - open_).total_seconds() // self._bar_s)
            return open_ + k * self._bar_td
        cand = after + self._bar_td
        open_, close = self._open_close(after.astimezone(_ET).date())
        if cand + self._bar_td <= close:
            return cand
        return self._open_close(self._next_weekday(after.astimezone(_ET).date()))[0]
