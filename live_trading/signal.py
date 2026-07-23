from __future__ import annotations

import pandas as pd


def compute_trigger(prev_bar: pd.Series, trigger: float) -> int:
    """
    Return direction for next bar based on previous 1H bar close-to-close return.
    +1 for long, -1 for short, 0 for no-trade.
    """
    ret = prev_bar["Close_ret"]
    if pd.isna(ret) or abs(ret) < trigger:
        return 0
    return 1 if ret > 0 else -1
