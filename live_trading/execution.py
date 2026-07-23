from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class FillResult:
    hit: str  # tp/sl/close
    entry: float
    exit: float
    ret: float


def simulate_bar(fill_dir: int, bar: pd.DataFrame, tp: float, sl: float, slip_bp: float = 0.0) -> FillResult:
    """
    Simulate execution within a single bar (aggregated lower-frequency bar) using its High/Low/Close.
    fill_dir: +1 long, -1 short. Entry at bar Open with slippage.
    """
    open_px = bar["Open"] * (1 + slip_bp / 10_000 * fill_dir)
    tp_px = open_px * (1 + tp * fill_dir)
    sl_px = open_px * (1 - sl * fill_dir)
    high, low, close = bar["High"], bar["Low"], bar["Close"]

    hit = "close"
    exit_px = close

    if fill_dir == 1:
        tp_hit = high >= tp_px
        sl_hit = low <= sl_px
    else:
        tp_hit = low <= tp_px
        sl_hit = high >= sl_px

    if sl_hit and tp_hit:
        hit = "sl"
        exit_px = sl_px
    elif sl_hit:
        hit = "sl"
        exit_px = sl_px
    elif tp_hit:
        hit = "tp"
        exit_px = tp_px

    ret = fill_dir * (exit_px - open_px) / open_px
    return FillResult(hit=hit, entry=open_px, exit=exit_px, ret=ret)
