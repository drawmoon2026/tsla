"""Single source of truth for loading price CSVs and resampling intraday bars.

Conventions enforced here (do NOT re-implement elsewhere):
- Index is tz-aware UTC. `pd.to_datetime(utc=True)` handles mixed offsets
  (ET cache spanning a DST switch), single-offset, and already-UTC inputs.
- Intraday timestamps are bar-START times (yfinance convention).
- Resampling never merges bars across trading days, uses closed="left",
  and drops partial buckets (e.g. the 30-minute remainder before 16:00).
"""

from __future__ import annotations

import pandas as pd

ET = "America/New_York"
OHLCV_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def load_bars(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV into a UTC tz-aware, sorted DataFrame.

    Raises if the timestamp column can't be made tz-aware or if the data
    doesn't look like US regular-session bars (guards against a CSV in naive
    local time being silently mis-labelled as UTC).
    """
    df = pd.read_csv(path)
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[ts_col], utc=True)
    df = df.drop(columns=[ts_col])
    df.index = idx
    df.index.name = "Datetime"
    df = df.sort_index()

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in cols:
        raise ValueError(f"{path}: no Close column found (columns={list(df.columns)})")
    df = df[cols].astype(float)

    et_hours = df.index.tz_convert(ET).hour
    rth_frac = ((et_hours >= 9) & (et_hours < 16)).mean()
    if len(df) and rth_frac < 0.5:
        raise ValueError(
            f"{path}: only {rth_frac:.0%} of bars fall in ET regular hours — "
            "timestamps are probably naive local time mis-read as UTC"
        )
    return df


def infer_bar_minutes(df: pd.DataFrame) -> int:
    """Median spacing of consecutive bars, in minutes."""
    if len(df) < 2:
        raise ValueError("need at least 2 bars to infer bar size")
    return int(pd.Series(df.index).diff().median().total_seconds() // 60)


def resample_bars(df: pd.DataFrame, minutes: int, min_frac: float = 0.8) -> pd.DataFrame:
    """Aggregate bar-start-stamped intraday bars into `minutes` buckets.

    - Buckets are anchored to each ET trading day's first bar time (9:30),
      labelled by bucket START, and never span two trading days.
    - Buckets holding fewer than `min_frac` of the expected sub-bars are
      dropped (kills the 25-minute remainder bar and single-bar buckets that
      previously produced fake "1H" signals out of overnight gaps).
    - Output column `n_sub` = sub-bars aggregated, kept for diagnostics.
    """
    if df.empty:
        return df.copy()
    sub_min = infer_bar_minutes(df)
    expected = minutes / sub_min
    if expected < 1:
        raise ValueError(f"cannot resample {sub_min}m bars into {minutes}m buckets")

    et = df.index.tz_convert(ET)
    parts = []
    for _, day in df.groupby(et.date, sort=True):
        day_et = day.index.tz_convert(ET)
        anchor = day_et[0]
        offset_min = anchor.hour * 60 + anchor.minute
        agg = day.resample(
            f"{minutes}min", offset=f"{offset_min}min", closed="left", label="left"
        ).agg(OHLCV_AGG)
        agg["n_sub"] = day["Close"].resample(
            f"{minutes}min", offset=f"{offset_min}min", closed="left", label="left"
        ).count()
        parts.append(agg)

    out = pd.concat(parts).dropna(subset=["Close"])
    out = out[out["n_sub"] >= min_frac * expected]
    return out


def intraday_returns(close: pd.Series) -> pd.Series:
    """close-to-close returns with the overnight gap removed: the first bar of
    each ET trading day gets NaN instead of (yesterday close -> today close)."""
    ret = close.pct_change()
    et_date = pd.Series(close.index.tz_convert(ET).date, index=close.index)
    first_of_day = et_date != et_date.shift(1)
    ret[first_of_day] = float("nan")
    return ret
