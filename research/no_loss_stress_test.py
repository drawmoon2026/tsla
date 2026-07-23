"""Stress test of the "never sell at a loss" accumulation strategy.

User's proposal: buy tranches on dips, sell each tranche ONLY when it is in
profit (+tp). Closed-trade win rate is 100% by construction — this script
measures where the risk actually lives instead:

- mark-to-market equity drawdown (unrealized losses are still losses),
- how long tranches stay underwater (capital lock-up),
- cash exhaustion (how early the ladder runs out during a real crash),
- final equity vs simply holding the stock.

Test window includes TSLA's real 2021-11 -> 2023-01 crash (-75% peak to
trough, ~3 years back to the high) — the scenario "big companies always
come back" must survive.

Usage:
    .venv/bin/python research/no_loss_stress_test.py
    .venv/bin/python research/no_loss_stress_test.py --sizing martingale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import load_bars

FEE = 0.0001  # 1bp per side


def run(bars: pd.DataFrame, capital: float, step: float, tp: float, sizing: str,
        mults: tuple = (1, 2, 4, 8)) -> dict:
    cash = capital
    tranches: list[dict] = []          # open positions: {cost, shares, t_open}
    closed: list[dict] = []
    high_mark = float(bars["Close"].iloc[0])   # running high of the current episode
    next_trigger = high_mark * (1 - step)      # next ladder level to buy
    exhausted_bars = 0
    eq_curve = []

    n_levels = 10 if sizing == "equal" else len(mults)
    base = capital / n_levels if sizing == "equal" else capital / sum(mults)

    for ts, row in bars.iterrows():
        px_low, px_high, px_close = float(row["Low"]), float(row["High"]), float(row["Close"])

        # sells: any tranche whose +tp target is touched this bar
        still_open = []
        for tr in tranches:
            target = tr["cost"] * (1 + tp)
            if px_high >= target:
                proceeds = tr["shares"] * target * (1 - FEE)
                cash += proceeds
                closed.append({
                    "t_open": tr["t_open"], "t_close": ts,
                    "days_held": (ts - tr["t_open"]).days,
                    "ret": target / tr["cost"] - 1 - 2 * FEE,
                })
            else:
                still_open.append(tr)
        if tranches and not still_open:
            # ladder fully closed: start a new episode from here
            high_mark = px_close
            next_trigger = high_mark * (1 - step)
        tranches = still_open

        if not tranches and px_close > high_mark:
            # flat and making new highs: the dip reference follows the high
            high_mark = px_close
            next_trigger = high_mark * (1 - step)

        # buys: cross as many ladder levels as this bar's low reaches
        while px_low <= next_trigger:
            lvl = len(tranches)
            stake = base if sizing == "equal" else (base * mults[lvl] if lvl < len(mults) else 0.0)
            if stake > 0 and cash >= stake:
                fill = next_trigger
                shares = stake * (1 - FEE) / fill
                tranches.append({"cost": fill, "shares": shares, "t_open": ts})
                cash -= stake
            else:
                exhausted_bars += 1
            next_trigger *= (1 - step)

        mtm = cash + sum(tr["shares"] * px_close for tr in tranches)
        eq_curve.append((ts, mtm))

    eq = pd.Series(dict(eq_curve))
    peak = eq.cummax()
    dd = (eq - peak) / peak
    # longest continuous stretch below the running peak, in days
    below = dd < 0
    grp = (below != below.shift()).cumsum()
    uw_days = 0
    for _, seg in eq.groupby(grp):
        if below.loc[seg.index[0]]:
            uw_days = max(uw_days, (seg.index[-1] - seg.index[0]).days)

    closed_df = pd.DataFrame(closed)
    open_underwater_days = [
        (bars.index[-1] - tr["t_open"]).days for tr in tranches
    ]
    last_px = float(bars["Close"].iloc[-1])
    bh_return = last_px / float(bars["Close"].iloc[0]) - 1

    return {
        "closed_trades": len(closed_df),
        "closed_win_rate": float((closed_df["ret"] > 0).mean()) if len(closed_df) else np.nan,
        "max_days_held_closed": int(closed_df["days_held"].max()) if len(closed_df) else 0,
        "still_open_tranches": len(tranches),
        "open_underwater_days_max": max(open_underwater_days) if open_underwater_days else 0,
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / capital - 1),
        "buy_and_hold_return": bh_return,
        "mtm_max_drawdown": float(dd.min()),
        "longest_underwater_days": uw_days,
        "cash_exhausted_bars": exhausted_bars,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test 'only sell in profit' accumulation.")
    ap.add_argument("--bars_csv", default="data/TSLA_1h_alpaca.csv")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--step", type=float, default=0.05, help="Buy another tranche every X drop (default 5%%).")
    ap.add_argument("--tp", type=float, default=0.02, help="Per-tranche take profit (default 2%%).")
    ap.add_argument("--sizing", choices=["equal", "martingale"], default="equal")
    args = ap.parse_args()

    bars = load_bars(args.bars_csv)
    print(f"Bars: {len(bars)}  range {bars.index.min().date()} .. {bars.index.max().date()}")
    print(f"Strategy: buy every -{args.step:.0%} step, sell tranche at +{args.tp:.0%}, "
          f"sizing={args.sizing}, NO stop loss, capital ${args.capital:,.0f}")
    stats = run(bars, args.capital, args.step, args.tp, args.sizing)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:,.4f}" if abs(v) < 10 else f"  {k:28s} {v:,.0f}")
        else:
            print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
