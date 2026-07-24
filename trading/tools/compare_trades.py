"""Trade-stream consistency diff: new backtest Runner vs legacy run_sim.

    python -m trading.tools.compare_trades \
        outputs/trading_backtest/trades.csv outputs/_arch_check/sim_trades.csv

Checks (regression contract from docs/architecture.md section 11):
- same trade count; entry_time and direction identical row-by-row;
- per-trade |ret diff| < tol (default 1e-9); exit_time / hit / prices
  reported when they differ.
Exit code 0 = consistent, 1 = divergent.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def compare(path_a: str, path_b: str, tol: float = 1e-9) -> int:
    a = pd.read_csv(path_a, parse_dates=["entry_time", "exit_time"])
    b = pd.read_csv(path_b, parse_dates=["entry_time", "exit_time"])
    print(f"A: {path_a} ({len(a)} trades)\nB: {path_b} ({len(b)} trades)")

    only_a = set(a["entry_time"]) - set(b["entry_time"])
    only_b = set(b["entry_time"]) - set(a["entry_time"])
    fail = False
    if only_a or only_b:
        fail = True
        for ts in sorted(only_a):
            print(f"  entry only in A: {ts}")
        for ts in sorted(only_b):
            print(f"  entry only in B: {ts}")

    m = a.merge(b, on="entry_time", suffixes=("_a", "_b"))
    if len(m):
        dir_bad = m[m["direction_a"] != m["direction_b"]]
        if len(dir_bad):
            fail = True
            print(f"  direction mismatches: {len(dir_bad)}\n{dir_bad[['entry_time']]}")
        ret_diff = (m["ret_a"] - m["ret_b"]).abs()
        print(f"  matched entries: {len(m)}  max |ret diff|: {ret_diff.max():.3e}")
        bad = m[ret_diff >= tol]
        if len(bad):
            fail = True
            print(f"  ret diffs >= {tol}:")
            print(bad[["entry_time", "direction_a", "hit_a", "hit_b",
                       "ret_a", "ret_b"]].to_string(index=False))
        for col in ("exit_time", "hit"):
            mm = m[m[f"{col}_a"] != m[f"{col}_b"]]
            if len(mm):
                fail = True
                print(f"  {col} mismatches: {len(mm)}")
                print(mm[["entry_time", f"{col}_a", f"{col}_b"]].to_string(index=False))
    print("RESULT:", "DIVERGENT" if fail else "CONSISTENT")
    return 1 if fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff two trade CSVs (entry/direction/ret parity).")
    ap.add_argument("path_a", help="trades.csv from trading.run backtest")
    ap.add_argument("path_b", help="sim_trades.csv from live_trading/run_sim.py")
    ap.add_argument("--tol", type=float, default=1e-9)
    sys.exit(compare(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
