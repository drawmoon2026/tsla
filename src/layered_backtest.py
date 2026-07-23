"""
Layered mean-reversion V-capture backtest on ZigZag swing events.

Strategy:
- Capital 10,000 split into 5 slices (2,000 each).
- Entry: slice1 at swing peak (anchor); slices2-3 at -5% from anchor; slices4-5 at -10% from anchor.
- Exit: each filled slice takes profit at +2% from its own entry; otherwise marked-to-market at rebound swing high.
- No stop; max holding until rebound swing high (from ZigZag sequence).

Assumptions:
- Intra-leg path is approximated as linear between swing points to time the -5%/-10% fills.
- Rebound leg monotonic to rebound swing; max price equals rebound price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

CAPITAL = 10_000
SLICES = 5
SLICE_CAP = CAPITAL / SLICES
TP = 0.02  # 2%
BAR_MIN = 5


@dataclass
class SliceResult:
    event_peak: pd.Timestamp
    slice_id: int
    filled: bool
    p_entry: float
    p_rebound: float
    hit_tp: bool
    pnl: float
    hold_bars: int


@dataclass
class EventResult:
    peak_t: pd.Timestamp
    drop_pct: float
    rebound_pct: float
    slices_filled: int
    invested: float
    pnl: float
    bars_to_rebound: int
    max_hold_bars: int


def load_price(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df["Close"]


def load_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["peak_t", "trough_t", "rebound_t"])


def simulate(price: pd.Series, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slice_rows: List[SliceResult] = []
    event_rows: List[EventResult] = []

    for _, r in events.iterrows():
        peak_t, trough_t, rebound_t = r["peak_t"], r["trough_t"], r["rebound_t"]
        if peak_t.tzinfo is None:
            peak_t = peak_t.tz_localize("UTC")
            trough_t = trough_t.tz_localize("UTC")
            rebound_t = rebound_t.tz_localize("UTC")

        try:
            p_peak = price.loc[peak_t]
            p_trough = price.loc[trough_t]
            p_reb = price.loc[rebound_t]
        except KeyError:
            continue

        drop = (p_peak - p_trough) / p_peak
        fills = []
        fills.append((1, peak_t, p_peak))
        if drop >= 0.05:
            t_fill = peak_t + (trough_t - peak_t) * (0.05 / drop)
            p_fill = p_peak * 0.95
            fills.extend([(2, t_fill, p_fill), (3, t_fill, p_fill)])
        if drop >= 0.10:
            t_fill = peak_t + (trough_t - peak_t) * (0.10 / drop)
            p_fill = p_peak * 0.90
            fills.extend([(4, t_fill, p_fill), (5, t_fill, p_fill)])

        slice_pnl = 0.0
        invested = 0.0
        max_hold = 0
        for sid, t_entry, p_entry in fills:
            invested += SLICE_CAP
            target = p_entry * (1 + TP)
            hit_tp = p_reb >= target
            pnl = SLICE_CAP * TP if hit_tp else SLICE_CAP * (p_reb / p_entry - 1)
            hold_bars = int(math.ceil((rebound_t - t_entry).total_seconds() / 60 / BAR_MIN))
            max_hold = max(max_hold, hold_bars)
            slice_pnl += pnl
            slice_rows.append(
                SliceResult(
                    event_peak=peak_t,
                    slice_id=sid,
                    filled=True,
                    p_entry=p_entry,
                    p_rebound=p_reb,
                    hit_tp=hit_tp,
                    pnl=pnl,
                    hold_bars=hold_bars,
                )
            )

        event_rows.append(
            EventResult(
                peak_t=peak_t,
                drop_pct=drop,
                rebound_pct=r["rebound_pct"],
                slices_filled=len(fills),
                invested=invested,
                pnl=slice_pnl,
                bars_to_rebound=int(
                    round((rebound_t - peak_t).total_seconds() / 60 / BAR_MIN)
                ),
                max_hold_bars=max_hold,
            )
        )

    return pd.DataFrame(slice_rows), pd.DataFrame(event_rows)


def metrics(events: pd.DataFrame, slices: pd.DataFrame) -> dict:
    # equity curve for max drawdown
    eq = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    for pnl in events.sort_values("peak_t")["pnl"]:
        eq += pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    util_avg = events["invested"].mean() / CAPITAL
    util_max = events["invested"].max() / CAPITAL
    turnover = events["invested"].sum() / CAPITAL

    return {
        "expected_return": events["pnl"].sum() / CAPITAL,
        "max_drawdown": max_dd,
        "util_avg": util_avg,
        "util_max": util_max,
        "turnover": turnover,
        "slice_tp_rate": slices["hit_tp"].mean(),
        "hold_mean_bars": slices["hold_bars"].mean(),
        "hold_median_bars": slices["hold_bars"].median(),
        "fill_distribution": events["slices_filled"].value_counts(),
    }


def main():
    price = load_price("data/TSLA_5m_60d.csv")
    events = load_events("zigzag_events.csv")
    slices, ev = simulate(price, events)
    m = metrics(ev, slices)

    print("Expected return: {:.2%}".format(m["expected_return"]))
    print("Max drawdown: {:.2%}".format(m["max_drawdown"]))
    print("Avg capital utilization: {:.1%}, max: {:.1%}".format(m["util_avg"], m["util_max"]))
    print("Turnover: {:.2f}x".format(m["turnover"]))
    print("Slice TP hit rate: {:.1%}".format(m["slice_tp_rate"]))
    print("Holding bars mean {:.1f}, median {:.1f}".format(m["hold_mean_bars"], m["hold_median_bars"]))
    print("Fill distribution (slices):")
    print(m["fill_distribution"])


if __name__ == "__main__":
    main()
