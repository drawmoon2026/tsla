from __future__ import annotations

import pandas as pd


def resample_1h(df: pd.DataFrame, offset_minutes: int = 30) -> pd.DataFrame:
    """
    Resample lower-frequency bars (e.g., 1m/5m) to 1H, aligned to 9:30 ET using offset.
    Expects columns: Open, High, Low, Close, Volume.
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample("1h", offset=f"{offset_minutes}min", label="right", closed="right").agg(agg)
    out["Close_ret"] = out["Close"].pct_change()
    return out.dropna()
