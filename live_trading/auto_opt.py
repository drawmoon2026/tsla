"""
Automated parameter search for the 1H breakout-follow strategy.
Stops when target_return is reached or max_iters exhausted.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_trading.config import load_config, Config
from live_trading.run_sim import run_sim


def sample_params() -> Tuple[float, float, float]:
    trigger = random.uniform(0.01, 0.02)      # 1%–2%
    tp = random.uniform(0.015, 0.035)         # 1.5%–3.5%
    sl = random.uniform(0.005, 0.015)         # 0.5%–1.5%
    return round(trigger, 4), round(tp, 4), round(sl, 4)


def auto_opt(feed: str, outdir: Path, target: float, max_iters: int, seed: int | None) -> dict:
    random.seed(seed)
    best = {"ret": -1e9, "trigger": None, "tp": None, "sl": None, "trades": None}
    rows = []

    for i in range(1, max_iters + 1):
        trig, tp, sl = sample_params()
        cfg: Config = load_config()
        cfg.trigger, cfg.tp, cfg.sl = trig, tp, sl
        trades_df, total_return = run_sim(cfg, feed, outdir / f"run_{i}")
        rows.append({"iter": i, "trigger": trig, "tp": tp, "sl": sl, "ret": total_return})

        if total_return > best["ret"]:
            best.update({"ret": total_return, "trigger": trig, "tp": tp, "sl": sl, "trades": len(trades_df)})
            trades_df.to_csv(outdir / "best_trades.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
            (outdir / "best_summary.txt").write_text(
                f"iter={i}, ret={total_return:.2%}, trigger={trig}, tp={tp}, sl={sl}, trades={len(trades_df)}",
                encoding="utf-8",
            )

        if total_return >= target:
            break

    pd.DataFrame(rows).to_csv(outdir / "search_log.csv", index=False)
    return best


def parse_args():
    ap = argparse.ArgumentParser(description="Auto parameter search until target return or max_iters reached.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--output_dir", default="outputs/auto_opt")
    ap.add_argument("--target_return", type=float, default=0.10, help="Target total return (e.g., 0.10 = 10%)")
    ap.add_argument("--max_iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best = auto_opt(args.feed, outdir, args.target_return, args.max_iters, args.seed)
    print("Best:", best)


if __name__ == "__main__":
    main()
