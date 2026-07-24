"""E8-A + S2: pooled-GBDT gated 5m V-rebound long with bear-market disable.

Frozen candidate (docs/strategy-lab.md E11 freeze declaration 2026-07-24):
    signal   5m V-event trough confirm (src/v_stats.py grid union, lb=lf=3)
    gate     leave-TSLA-out LightGBM prob >= 0.43250235336744486 (top ~10%)
    geometry long only, tp +0.5% / sl -2.0% / timeout 48 x 5m bars, day-flat
    S2       no entries while yesterday's daily close is >20% below the max
             of the last 252 daily closes (rolling(252) min_periods=252,
             shift(1)) — research/e11_bear_switch.switch_states, verbatim
Parameters are loaded from models/e8a/meta.json (research/e8a_freeze_model.py)
and are NOT tunable here — the freeze declaration forbids it.

Real-time equivalence notes (vs the research pipeline):
- V detection re-runs the v_stats grid (36 combos, deduped) on a rolling
  bar buffer every time the newest bar confirms a local trough (the bar
  ``lookforward`` bars after it closes). One deliberate deviation: the
  research detector SKIPS a trigger whose rebound window extends past the
  end of the data ("unresolved"); live, "the data ends now" is every bar,
  and the outcome is irrelevant to acting on the trigger — such pending
  triggers are emitted. Buffer truncation can also drop the odd event whose
  scan path crosses the buffer edge; differences are expected to be small
  and are quantified by the replay smoke check.
- Features replicate research/ml_common.build_dataset at the confirm-bar
  close (info <= confirm close only), vol-normalized as in e8_pooled_gbdt;
  rv20_rel uses the frozen TSLA train-period median rv20 from meta.json.
- Events whose features would be NaN in research (short vol history, no
  prev-day return, rv20=0) are skipped, matching the research NaN-row drop.
- Daily context for S2 / prev_day_ret comes from local daily-close history
  (bundled CSVs + a persistent cache updated on day rollover) plus an
  optional Alpaca 1Day backfill at startup for live runs, so the switch
  never silently runs on stale data without logging it.

The strategy emits at most ONE intent per confirm bar (multiple grid events
sharing a confirm bar share the entry bar in research too; the intent fires
if ANY of them passes the gate, which reproduces the research entry set).
"""

from __future__ import annotations

import json
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading.core.types import Bar
from trading.strategy.base import SignalIntent, StrategyContext

_ET = ZoneInfo("America/New_York")
_ROOT = Path(__file__).resolve().parents[2]

# v_stats detection grid — frozen with the candidate (meta.json v_detection)
_GRID_X = (0.005, 0.01, 0.02)
_GRID_Y = (0.003, 0.005, 0.01)
_GRID_T = (12, 24, 48, 78)
_LB = _LF = 3
_MAX_DROP_BARS = 78


def _detect_grid_confirm_now(closes: np.ndarray, et_dates: np.ndarray
                             ) -> list[tuple[int, int]]:
    """Run the v_stats grid on the buffer; return deduped (peak_i, trough_i)
    pairs whose trough CONFIRMS on the last bar (trough_i + lf == n-1).

    Logic is a line-for-line port of src.v_stats.detect_v_events with one
    patch: a trigger at the buffer end whose rebound window is incomplete is
    EMITTED (pending) instead of skipped as unresolved — see module docstring.
    """
    from src.v_stats import find_local_extrema

    n = len(closes)
    peaks_mask, troughs_mask = find_local_extrema(pd.Series(closes), _LB, _LF)
    target = n - 1 - _LF
    found: set[tuple[int, int]] = set()

    for x in _GRID_X:
        for y in _GRID_Y:
            for tbars in _GRID_T:
                i = _LB
                while i < n - _LF:
                    if not peaks_mask[i]:
                        i += 1
                        continue
                    orig_i = i
                    peak_i, peak_p = i, closes[i]
                    trough_i: Optional[int] = None
                    j = i + 1
                    while j < n - _LF:
                        if et_dates[j] != et_dates[peak_i] or (j - peak_i) > _MAX_DROP_BARS:
                            break
                        if closes[j] > peak_p:
                            peak_i, peak_p = j, closes[j]
                        elif troughs_mask[j]:
                            if (peak_p - closes[j]) / peak_p >= x:
                                trough_i = j
                                break
                        j += 1
                    if trough_i is None:
                        i = orig_i + 1
                        continue
                    trough_p = closes[trough_i]
                    rebound_i: Optional[int] = None
                    k = trough_i + 1
                    limit = min(n - 1, trough_i + tbars)
                    while k <= limit:
                        if (closes[k] - trough_p) / trough_p >= y:
                            rebound_i = k
                            break
                        k += 1
                    if trough_i == target:
                        found.add((peak_i, trough_i))
                    if rebound_i is not None:
                        i = rebound_i + 1
                    else:
                        i = orig_i + 1
    return sorted(found)


class E8AVReversalStrategy:
    def __init__(
        self,
        symbol: str = "TSLA",
        model_dir: str | Path = _ROOT / "models" / "e8a",
        daily_history_csvs: Sequence[str | Path] = (
            _ROOT / "data" / "TSLA_5m_3y.csv",
            _ROOT / "data" / "TSLA_1h_alpaca.csv",
        ),
        daily_cache_path: str | Path = _ROOT / "data" / "TSLA_daily_e8a_cache.csv",
        live_backfill: bool = False,
        buffer_bars: int = 800,
    ) -> None:
        import joblib

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.model = joblib.load(model_dir / "model.joblib")
        self.symbol = symbol
        self.features: list[str] = self.meta["features"]
        self.gate: float = float(self.meta["gate"]["threshold"])
        g = self.meta["geometry"]
        self.tp, self.sl = float(g["tp_pct"]), float(g["sl_pct"])
        self.timeout_bars = int(g["timeout_bars"])
        s2 = self.meta["s2_switch"]
        self.s2_lookback = int(s2["lookback_days"])
        self.s2_dd = float(s2["dd_threshold"])
        self.rv20_med_train = float(self.meta["tsla_rv20_med_train"])
        self.symbol_categories: list[str] = self.meta["symbol_categories"]

        self._bars: deque[Bar] = deque(maxlen=buffer_bars)
        self._cur_day: Optional[date] = None
        self._last_bar: Optional[Bar] = None
        self._signaled: set[datetime] = set()   # confirm-bar starts already fired
        self._s2_off = False
        self._cache_path = Path(daily_cache_path)
        self._daily: dict[date, float] = self._load_daily(daily_history_csvs)
        if live_backfill:
            self._backfill_daily()
        # visible to the runner's shadow-status writer
        self.status_info: dict = {"strategy": "e8a", "s2": None, "signals": 0}

    # -- daily-close context (S2 + prev_day_ret) ---------------------------
    def _load_daily(self, csvs: Sequence[str | Path]) -> dict[date, float]:
        from src.common.data_io import ET, load_bars

        daily: dict[date, float] = {}
        for p in csvs:
            p = Path(p)
            if not p.exists():
                print(f"[e8a] daily history {p} missing — skipped")
                continue
            bars = load_bars(str(p))
            dc = bars["Close"].groupby(
                np.asarray(bars.index.tz_convert(ET).date)).last()
            for d, c in dc.items():          # earlier files take precedence
                daily.setdefault(d, float(c))
        if self._cache_path.exists():
            cache = pd.read_csv(self._cache_path)
            for _, r in cache.iterrows():
                daily.setdefault(
                    datetime.strptime(str(r["date"]), "%Y-%m-%d").date(),
                    float(r["close"]))
        if daily:
            print(f"[e8a] daily closes loaded: {len(daily)} days "
                  f"({min(daily)} .. {max(daily)})")
        return daily

    def _append_cache(self, d: date, close: float) -> None:
        try:
            new = not self._cache_path.exists()
            with open(self._cache_path, "a", encoding="utf-8") as f:
                if new:
                    f.write("date,close\n")
                f.write(f"{d.isoformat()},{close}\n")
        except OSError as exc:
            print(f"[e8a] daily cache write failed: {exc!r}")

    def _backfill_daily(self) -> None:
        """Fetch missing recent daily closes from Alpaca (live runs only)."""
        try:
            import requests

            from src.alpaca_data import DATA_URL, load_keys

            last = max(self._daily) if self._daily else date(2018, 1, 1)
            today_et = datetime.now(_ET).date()
            if last >= today_et - timedelta(days=1):
                return
            key, secret = load_keys()
            r = requests.get(
                DATA_URL.format(symbol=self.symbol),
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                params={"timeframe": "1Day",
                        "start": (last + timedelta(days=1)).isoformat(),
                        "adjustment": "split", "feed": "iex", "sort": "asc",
                        "limit": 1000},
                timeout=30)
            r.raise_for_status()
            got = 0
            for row in r.json().get("bars") or []:
                d = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")) \
                    .astimezone(_ET).date()
                if d >= today_et:            # today's daily bar is in-progress
                    continue
                if d not in self._daily:
                    self._daily[d] = float(row["c"])
                    self._append_cache(d, float(row["c"]))
                    got += 1
            print(f"[e8a] daily backfill: +{got} days (through {max(self._daily)})")
        except Exception as exc:  # noqa: BLE001 — backfill is best-effort
            print(f"[e8a] daily backfill failed ({exc!r}) — "
                  "S2 will use the newest cached close")

    def _s2_state_for(self, today: date) -> bool:
        """True = disabled. e11 switch_states verbatim: yesterday's close vs
        the max of the 252 daily closes ending at yesterday; inactive until
        252 days of history exist (min_periods)."""
        prior = sorted(d for d in self._daily if d < today)
        info = {"off": False, "dd_pct": None, "asof": None,
                "threshold": self.s2_dd, "n_days": len(prior)}
        if prior:
            yday = prior[-1]
            window = prior[-self.s2_lookback:]
            hi = max(self._daily[d] for d in window)
            dd = self._daily[yday] / hi - 1.0
            info.update(asof=yday.isoformat(), dd_pct=round(dd, 6))
            if len(window) >= self.s2_lookback and dd < self.s2_dd:
                info["off"] = True
            stale = (today - yday).days
            if stale > 4:
                info["stale_days"] = stale
                print(f"[e8a] WARNING: newest daily close is {yday} "
                      f"({stale} calendar days old) — S2 may lag")
        self.status_info["s2"] = info
        return bool(info["off"])

    # -- Strategy protocol -------------------------------------------------
    def warmup_bars(self) -> int:
        return 100

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> list[SignalIntent]:
        if bar.symbol != self.symbol or not bar.complete:
            return []
        day = bar.start.astimezone(_ET).date()
        if self._cur_day is None or day != self._cur_day:
            if self._cur_day is not None and self._last_bar is not None:
                prev_close = float(self._last_bar.close)
                if self._cur_day not in self._daily:
                    self._daily[self._cur_day] = prev_close
                    self._append_cache(self._cur_day, prev_close)
            self._cur_day = day
            self._s2_off = self._s2_state_for(day)
            if self._s2_off:
                print(f"[e8a] {day}: S2 OFF-switch active "
                      f"(dd {self.status_info['s2']['dd_pct']:+.2%} < "
                      f"{self.s2_dd:+.0%}) — no entries today")
        self._bars.append(bar)
        self._last_bar = bar

        if self._s2_off:
            return []
        if ctx.position(self.symbol).qty != 0:
            return []
        n = len(self._bars)
        if n < _LB + _LF + 2:
            return []

        # cheap pre-check: is the bar `lf` bars back a fresh local trough?
        closes = np.fromiter((b.close for b in self._bars), dtype=float, count=n)
        t = n - 1 - _LF
        if t < _LB or closes[t] > closes[t - _LB: t + _LF + 1].min():
            return []
        if bar.start in self._signaled:
            return []

        et_dates = np.asarray([b.start.astimezone(_ET).date() for b in self._bars])
        events = _detect_grid_confirm_now(closes, et_dates)
        if not events:
            return []

        feats_common = self._features_at_confirm(closes, et_dates, cpos=n - 1)
        if feats_common is None:
            return []
        best: Optional[tuple[float, dict]] = None
        for peak_i, trough_i in events:
            row = self._event_features(peak_i, trough_i, feats_common)
            if row is None:
                continue
            prob = self._predict(row)
            if best is None or prob > best[0]:
                best = (prob, row)
        if best is None:
            return []
        prob, _ = best
        if prob < self.gate:
            return []

        self._signaled.add(bar.start)
        self.status_info["signals"] += 1
        et_end = (bar.start + timedelta(seconds=bar.duration_s)).astimezone(_ET)
        return [SignalIntent(
            symbol=self.symbol,
            direction=1,
            kind="entry",
            tp_pct=self.tp,
            sl_pct=self.sl,
            ttl_bars=self.timeout_bars,
            confidence=float(prob),
            tag=f"e8a:{et_end:%Y-%m-%dT%H:%M}:p={prob:.4f}",
        )]

    # -- feature construction (ml_common.build_dataset live equivalent) ----
    def _features_at_confirm(self, closes: np.ndarray, et_dates: np.ndarray,
                             cpos: int) -> Optional[dict]:
        from src.common.data_io import intraday_returns

        bars = self._bars
        idx = pd.DatetimeIndex([b.start for b in bars])
        # rv20: std of overnight-gap-free 5m returns, 20 bars ending at cpos
        iret = intraday_returns(pd.Series(closes, index=idx))
        rv20 = float(iret.rolling(20, min_periods=10).std().iloc[cpos])
        if not np.isfinite(rv20) or rv20 == 0.0:
            return None
        today = et_dates[cpos]
        day_first = int(np.argmax(et_dates == today))
        if et_dates[day_first] != today:
            return None
        prior = sorted(d for d in self._daily if d < today)
        if len(prior) < 2:
            return None
        prev_day_ret = self._daily[prior[-1]] / self._daily[prior[-2]] - 1.0
        confirm_et = bars[cpos].start.astimezone(_ET)
        return {
            "cpos": cpos,
            "rv20": rv20,
            "day_first": day_first,
            "et_hour": confirm_et.hour + confirm_et.minute / 60.0,
            "open_to_now_ret": float(closes[cpos] / bars[day_first].open - 1.0),
            "prev_day_ret": float(prev_day_ret),
            "bars_since_open": cpos - day_first,
        }

    def _event_features(self, peak_i: int, trough_i: int,
                        common: dict) -> Optional[dict]:
        bars = self._bars
        vols = np.fromiter((b.volume for b in bars), dtype=float, count=len(bars))
        lo = max(0, trough_i - 20)
        prior_vols = vols[lo:trough_i]
        if len(prior_vols) < 10:
            return None                       # research: NaN -> row dropped
        vol_ma20 = float(prior_vols.mean())
        if not np.isfinite(vol_ma20) or vol_ma20 <= 0:
            return None
        drop_pct = (bars[peak_i].close - bars[trough_i].close) / bars[peak_i].close
        rv = common["rv20"]
        return {
            "drop_pct_n": drop_pct / rv,
            "drop_speed_n": (drop_pct / max(trough_i - peak_i, 1)) / rv,
            "vol_ratio_trough": bars[trough_i].volume / vol_ma20,
            "et_hour": common["et_hour"],
            "open_to_now_ret_n": common["open_to_now_ret"] / rv,
            "prev_day_ret_n": common["prev_day_ret"] / rv,
            "rv20_rel": rv / self.rv20_med_train,
            "bars_since_open": common["bars_since_open"],
        }

    def _predict(self, row: dict) -> float:
        X = pd.DataFrame([{f: row[f] for f in self.features if f != "symbol"}])
        X["symbol"] = pd.Categorical([self.symbol],
                                     categories=self.symbol_categories)
        return float(self.model.predict_proba(X[self.features])[0, 1])
