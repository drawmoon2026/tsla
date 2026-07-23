"""Cost-inclusive backtest of the GBDT rebound filter, layered by OOF probability.

Answers ONE question: does AUC 0.602 survive contact with the pessimistic
execution model? Design choices that keep it honest:

- Only out-of-fold probabilities are used (each event scored by a model that
  never saw it); events from the first training block (no OOF score) are
  excluded from ALL layers including the baseline, so layers are comparable.
- The trade mirrors the label exactly: enter at the first 5m bar open after
  trough_confirm_t, TP +0.5% / SL -1% / timeout 24 bars, never held overnight.
- Settlement via src.common.execution.settle_bracket (gap-through stops,
  strict-cross TP, SL priority, 1bp fee + 2bp slippage per side).
- One position at a time: overlapping signals are skipped, identically across
  layers (a higher threshold can therefore *unlock* later signals — that is
  part of what a filter does in production).

Usage:
    .venv/bin/python research/prob_layered_backtest.py
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

from src.common.data_io import ET, load_bars
from src.common.execution import CostModel, entry_fill, settle_bracket

TP = 0.005
SL = 0.010
TIMEOUT_BARS = 24


def run_layer(
    signals: pd.DataFrame, bars: pd.DataFrame, cost: CostModel, capital: float
) -> dict:
    et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
    eq = capital
    peak = capital
    max_dd = 0.0
    rets = []
    busy_until = None
    bars_in_market = 0

    for _, s in signals.iterrows():
        entry_t = s["entry_t"]
        if busy_until is not None and entry_t <= busy_until:
            continue
        loc = bars.index.get_indexer([entry_t])
        if loc[0] == -1:
            continue
        i = loc[0]
        day = et_dates.iloc[i]
        window = bars.iloc[i : i + TIMEOUT_BARS]
        window = window[et_dates.loc[window.index] == day]  # never hold overnight
        if window.empty:
            continue

        entry_px = entry_fill(float(window.iloc[0]["Open"]), 1, cost)
        res = settle_bracket(window, 1, entry_px, entry_px * (1 + TP), entry_px * (1 - SL), cost)

        rets.append(res.ret)
        eq *= 1 + res.ret
        peak = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak)
        busy_until = res.exit_time
        bars_in_market += window.index.get_loc(res.exit_time) + 1

    rets = np.array(rets)
    n = len(rets)
    return {
        "trades": n,
        "total_return": eq / capital - 1,
        "avg_ret_bp": rets.mean() * 10_000 if n else np.nan,
        "median_ret_bp": np.median(rets) * 10_000 if n else np.nan,
        "win_rate": (rets > 0).mean() if n else np.nan,
        "max_dd": max_dd,
        "t_stat": rets.mean() / rets.std(ddof=1) * np.sqrt(n) if n > 1 and rets.std(ddof=1) > 0 else np.nan,
        "bars_in_market": bars_in_market,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probability-layered, cost-inclusive backtest of the GBDT filter.")
    ap.add_argument("--scores_csv", default="outputs/ml_filter_2y/per_event_scores.csv")
    ap.add_argument("--bars_csv", default="data/TSLA_5m_alpaca.csv")
    ap.add_argument("--out_dir", default="outputs/prob_layered_backtest")
    ap.add_argument("--fee_bp", type=float, default=1.0)
    ap.add_argument("--slippage_bp", type=float, default=2.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args.bars_csv)
    cost = CostModel(fee_bp=args.fee_bp, slippage_bp=args.slippage_bp)

    scores = pd.read_csv(args.scores_csv, parse_dates=["entry_t"])
    scores = scores[scores["prob"].notna()].sort_values("entry_t")
    print(f"OOF-scored signals: {len(scores)} "
          f"({scores['entry_t'].min().date()} .. {scores['entry_t'].max().date()})")

    layers = [("baseline (all)", None), ("prob>=0.5", 0.5), ("prob>=0.6", 0.6), ("prob>=0.7", 0.7)]
    rows = []
    for name, thr in layers:
        sel = scores if thr is None else scores[scores["prob"] >= thr]
        stats = run_layer(sel, bars, cost, args.capital)
        stats["layer"] = name
        stats["signals"] = len(sel)
        rows.append(stats)

    df = pd.DataFrame(rows)[
        ["layer", "signals", "trades", "total_return", "avg_ret_bp", "median_ret_bp",
         "win_rate", "max_dd", "t_stat", "bars_in_market"]
    ]
    df.to_csv(outdir / "layers.csv", index=False)

    lines = [
        "GBDT 概率分层成本后回测（OOF 概率，三重障碍 +0.5%/-1%/24bar，不过夜，"
        f"单边 fee {args.fee_bp}bp + slip {args.slippage_bp}bp）",
        df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "判读: 若 avg_ret_bp 不随阈值单调上升、或最高层 t_stat < 2，则 AUC 优势未能覆盖交易成本。",
        "样本期 2024-2026 单一 regime，本结果不构成实盘依据。",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
