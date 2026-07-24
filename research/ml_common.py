#!/usr/bin/env python
"""Shared feature/label construction for the GBDT V-rebound filter.

Single source of truth extracted from research/ml_filter.py (E8 refactor) so
the single-symbol pipeline (ml_filter.py) and the pooled cross-symbol
pipeline (e8_pooled_gbdt.py) use EXACTLY the same event loading, feature
engineering, triple-barrier labelling and model hyper-parameters.

Anti-lookahead contract (unchanged from ml_filter.py)
-----------------------------------------------------
- Bars are stamped by bar-START time (UTC tz-aware).
- `trough_confirm_t` is the start time of the bar `lf` bars after the trough
  (src/v_stats.py). The trough is identifiable only once that bar CLOSES:
    * actionable entry = Open of the FIRST bar AFTER the confirm bar;
    * features may use anything up to and including the confirm bar's close;
    * anything after the confirm bar's close feeds the LABEL only.
- `outcome` (rebounded/timeout) is resolved AFTER entry -> future info.
  It must never leak into the model input; labels are always re-computed by
  the triple-barrier logic below, never taken from `outcome`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, intraday_returns  # noqa: E402

# ------------------------------------------------------------------ protocol
LABEL_TP = 0.005      # +y: take-profit touch (via High) -> label 1
LABEL_SL = 0.010      # -x: stop touch (via Low) -> label 0
LABEL_HORIZON = 24    # bars (24 x 5m = 2h); timeout -> label 0
SEED = 42

FEATURES = [
    "drop_pct",          # peak->trough drawdown (known at confirm)
    "drop_speed",        # drop_pct / bars peak->trough
    "vol_ratio_trough",  # trough-bar volume / mean volume of prior 20 bars
    "et_hour",           # ET hour of confirm time (fractional)
    "open_to_now_ret",   # day open -> confirm-bar close return
    "prev_day_ret",      # previous trading day close-to-close return
    "rv20",              # std of intraday 5m returns, 20 bars ending at confirm
    "bars_since_open",   # confirm bar's position within the trading day
]

# `outcome` (rebounded/timeout) is resolved AFTER entry -> future info.
FORBIDDEN_FEATURES = {"outcome", "rebound_t", "rebound_p", "rebound_pct", "bars_to_rebound"}
assert not (set(FEATURES) & FORBIDDEN_FEATURES), "future-information column in FEATURES"


def load_events(events_csv: Path) -> pd.DataFrame:
    ev = pd.read_csv(events_csv)
    for c in ("peak_t", "trough_t", "trough_confirm_t", "rebound_t"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    if "outcome" not in ev.columns:
        # legacy table (pre-survivorship-fix): every recorded event rebounded
        ev["outcome"] = "rebounded"
    return ev.sort_values("trough_confirm_t").reset_index(drop=True)


def build_dataset(ev: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return one row per usable event: features + label + bookkeeping."""
    idx = bars.index
    n = len(bars)
    opens = bars["Open"].to_numpy()
    highs = bars["High"].to_numpy()
    lows = bars["Low"].to_numpy()
    closes = bars["Close"].to_numpy()

    et_index = idx.tz_convert(ET)
    et_dates = np.asarray(et_index.date)
    # position of each day's first bar, per bar
    day_change = np.r_[True, et_dates[1:] != et_dates[:-1]]
    day_first_pos = np.maximum.accumulate(np.where(day_change, np.arange(n), -1))

    # per-day close series for prev-day return (close-to-close)
    daily_close = bars["Close"].groupby(et_dates).last()
    daily_ret = daily_close.pct_change()  # ret of day d vs d-1, known at d's close
    days_sorted = list(daily_close.index)
    day_pos = {d: i for i, d in enumerate(days_sorted)}

    # volume ratio denominator: mean volume of the 20 bars BEFORE each bar
    vol_ma20_prior = (
        bars["Volume"].shift(1).rolling(20, min_periods=10).mean().to_numpy()
    )
    # realized vol: std of overnight-gap-free 5m returns, 20 bars ending at pos
    iret = intraday_returns(bars["Close"])
    rv20 = iret.rolling(20, min_periods=10).std().to_numpy()

    confirm_pos = idx.get_indexer(ev["trough_confirm_t"])
    trough_pos = idx.get_indexer(ev["trough_t"])
    if (confirm_pos < 0).any() or (trough_pos < 0).any():
        bad = int((confirm_pos < 0).sum() + (trough_pos < 0).sum())
        raise RuntimeError(f"{bad} event timestamps not found in the bar index")

    rows: list[dict] = []
    drops = {"entry_beyond_data": 0, "entry_next_day": 0, "label_unresolved": 0}
    n_ambiguous = 0

    for i, e in ev.iterrows():
        cpos = int(confirm_pos[i])
        tpos = int(trough_pos[i])
        epos = cpos + 1
        if epos >= n:
            drops["entry_beyond_data"] += 1
            continue
        if et_dates[epos] != et_dates[cpos]:
            # confirm bar was the day's last bar; "next bar open" would be an
            # overnight-gap entry on the following day -> different trade, skip
            drops["entry_next_day"] += 1
            continue

        # ---------------- label (uses ONLY post-confirm information) --------
        entry = opens[epos]
        tp_px = entry * (1.0 + LABEL_TP)
        sl_px = entry * (1.0 - LABEL_SL)
        label: int | None = None
        last = min(epos + LABEL_HORIZON - 1, n - 1)
        for k in range(epos, last + 1):
            if et_dates[k] != et_dates[epos]:
                label = 0  # day ended first -> forced flat -> timeout
                break
            hit_tp = highs[k] >= tp_px
            hit_sl = lows[k] <= sl_px
            if hit_tp and hit_sl:
                # both barriers inside one 5m bar: order unknowable ->
                # conservative label 0 (assume stop first)
                n_ambiguous += 1
                label = 0
                break
            if hit_sl:
                label = 0
                break
            if hit_tp:
                label = 1
                break
        if label is None:
            if last == n - 1 and et_dates[last] == et_dates[epos]:
                # data (not the day) ran out mid-horizon: outcome unknown
                drops["label_unresolved"] += 1
                continue
            label = 0  # full horizon elapsed without a touch -> timeout

        # ---------------- features (info <= confirm-bar close only) ---------
        d = day_pos[et_dates[cpos]]
        prev_day_ret = float(daily_ret.iloc[d - 1]) if d >= 1 else np.nan

        rows.append(
            {
                "peak_t": e["peak_t"],
                "trough_t": e["trough_t"],
                "trough_confirm_t": e["trough_confirm_t"],
                "entry_t": idx[epos],
                "entry_price": float(entry),
                "et_day": et_dates[cpos],
                "outcome": e["outcome"],  # bookkeeping ONLY — never a feature
                "label": int(label),
                "drop_pct": float(e["drop_pct"]),
                "drop_speed": float(e["drop_pct"]) / max(int(e["bars_to_trough"]), 1),
                "vol_ratio_trough": float(
                    bars["Volume"].iloc[tpos] / vol_ma20_prior[tpos]
                )
                if np.isfinite(vol_ma20_prior[tpos]) and vol_ma20_prior[tpos] > 0
                else np.nan,
                "et_hour": et_index[cpos].hour + et_index[cpos].minute / 60.0,
                "open_to_now_ret": float(
                    closes[cpos] / opens[day_first_pos[cpos]] - 1.0
                ),
                "prev_day_ret": prev_day_ret,
                "rv20": float(rv20[cpos]),
                "bars_since_open": int(cpos - day_first_pos[cpos]),
            }
        )

    info = {"drops": drops, "n_ambiguous_bars": n_ambiguous}
    return pd.DataFrame(rows), info


def make_lgbm(seed: int = SEED):
    """The exact LightGBM configuration validated in E3/ml_filter."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        verbosity=-1,
    )
