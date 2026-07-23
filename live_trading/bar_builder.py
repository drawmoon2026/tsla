from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import intraday_returns, resample_bars


def resample_1h(df: pd.DataFrame, offset_minutes: int = 30) -> pd.DataFrame:
    """Thin wrapper over src.common.data_io.resample_bars(df, 60).

    Labels are bucket-START times, buckets are anchored to each ET trading
    day's first bar (9:30), never span two days, and partial buckets (e.g. the
    trailing 30-minute remainder) are dropped.

    `offset_minutes` is IGNORED: bucket anchoring is now derived per trading
    day by resample_bars, so a manual offset is neither needed nor honoured.
    The parameter is kept only for signature compatibility.

    `Close_ret` is the intraday close-to-close return with the overnight gap
    removed (first bucket of each day is NaN).
    """
    out = resample_bars(df, 60)
    out["Close_ret"] = intraday_returns(out["Close"])
    return out
