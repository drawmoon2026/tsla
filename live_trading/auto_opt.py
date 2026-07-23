"""
Automated parameter search for the 1H breakout-follow strategy, with an
honest evaluation protocol:

- The feed is split by ET trading day: first `train_frac` of days for search,
  the rest as a held-out validation set the optimizer never scores against.
- Random search runs on the TRAIN split only; candidates need >= min_trades.
- The top-K train candidates are then evaluated once on VALIDATION, and the
  winner is picked by validation score = return / max(1%, |max drawdown|).
- The winner's neighborhood (params +/-10%) is re-evaluated on validation:
  if the score collapses under small perturbations, the "optimum" is noise.
- There is deliberately NO "stop when target return is reached" — that was
  publication bias on random draws.

Reported numbers you may act on: the VALIDATION block of best_summary.txt.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_trading.config import Config, load_config
from live_trading.run_sim import run_sim
from src.common.data_io import ET, load_bars


def sample_params() -> tuple[float, float, float]:
    trigger = random.uniform(0.01, 0.02)      # 1%-2%
    tp = random.uniform(0.015, 0.035)         # 1.5%-3.5%
    sl = random.uniform(0.005, 0.015)         # 0.5%-1.5%
    return round(trigger, 4), round(tp, 4), round(sl, 4)


def split_feed(feed: str, outdir: Path, train_frac: float) -> tuple[Path, Path, dict]:
    df = load_bars(feed)
    days = sorted(pd.unique(df.index.tz_convert(ET).date))
    n_train = int(len(days) * train_frac)
    if n_train < 5 or len(days) - n_train < 3:
        raise ValueError(f"not enough trading days to split: {len(days)}")
    train_days = set(days[:n_train])
    et_date = pd.Series(df.index.tz_convert(ET).date, index=df.index)

    train_path = outdir / "train.csv"
    valid_path = outdir / "valid.csv"
    df[et_date.isin(train_days).values].to_csv(
        train_path, index_label="Datetime", date_format="%Y-%m-%dT%H:%M:%S%z")
    df[~et_date.isin(train_days).values].to_csv(
        valid_path, index_label="Datetime", date_format="%Y-%m-%dT%H:%M:%S%z")
    meta = {
        "train_days": n_train,
        "valid_days": len(days) - n_train,
        "train_range": f"{days[0]} .. {days[n_train - 1]}",
        "valid_range": f"{days[n_train]} .. {days[-1]}",
    }
    return train_path, valid_path, meta


def evaluate(trigger: float, tp: float, sl: float, feed: Path, scratch: Path) -> dict:
    cfg: Config = load_config()
    cfg.trigger, cfg.tp, cfg.sl = trigger, tp, sl
    cfg.validate()
    trades_df, total_return = run_sim(cfg, str(feed), scratch)
    if trades_df.empty:
        max_dd = 0.0
    else:
        eq = (1 + trades_df["ret"]).cumprod()
        max_dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    score = total_return / max(0.01, abs(max_dd))
    return {
        "trigger": trigger, "tp": tp, "sl": sl,
        "ret": total_return, "max_dd": max_dd, "trades": len(trades_df),
        "score": score,
    }


def auto_opt(
    feed: str, outdir: Path, max_iters: int, seed: int | None,
    train_frac: float, top_k: int, min_trades: int,
) -> dict | None:
    random.seed(seed)
    train_path, valid_path, meta = split_feed(feed, outdir, train_frac)
    scratch = outdir / "_eval"
    print(f"Split: train {meta['train_range']} ({meta['train_days']}d), "
          f"valid {meta['valid_range']} ({meta['valid_days']}d)")

    # 1) random search on TRAIN only
    seen: set[tuple[float, float, float]] = set()
    train_rows = []
    for i in range(1, max_iters + 1):
        params = sample_params()
        if params in seen:
            continue
        seen.add(params)
        row = evaluate(*params, train_path, scratch)
        row["iter"] = i
        train_rows.append(row)
    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(outdir / "search_log.csv", index=False)

    gated = train_df[train_df["trades"] >= min_trades]
    if gated.empty:
        print(f"No candidate reached {min_trades} train trades — nothing selectable.")
        return None
    top = gated.sort_values("score", ascending=False).head(top_k)

    # 2) score the top-K once on VALIDATION
    valid_rows = []
    for _, c in top.iterrows():
        row = evaluate(c["trigger"], c["tp"], c["sl"], valid_path, scratch)
        row["train_ret"] = c["ret"]
        row["train_trades"] = c["trades"]
        valid_rows.append(row)
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv(outdir / "valid_eval.csv", index=False)
    best = valid_df.sort_values("score", ascending=False).iloc[0]

    # 3) neighborhood perturbation on VALIDATION (+/-10% per param)
    perturb_rows = []
    for name in ("trigger", "tp", "sl"):
        for f in (0.9, 1.1):
            p = {"trigger": best["trigger"], "tp": best["tp"], "sl": best["sl"]}
            p[name] = round(p[name] * f, 4)
            r = evaluate(p["trigger"], p["tp"], p["sl"], valid_path, scratch)
            r["perturbed"] = f"{name}x{f}"
            perturb_rows.append(r)
    perturb_df = pd.DataFrame(perturb_rows)
    perturb_df.to_csv(outdir / "perturbation.csv", index=False)
    fragile = perturb_df["ret"].min() < best["ret"] - 0.5 * abs(best["ret"])

    summary = "\n".join([
        f"数据切分: train {meta['train_range']} ({meta['train_days']}d) | "
        f"valid {meta['valid_range']} ({meta['valid_days']}d)",
        f"最优参数 (按验证集 score 选出): trigger={best['trigger']}, tp={best['tp']}, sl={best['sl']}",
        f"训练集: ret={best['train_ret']:.2%}, trades={int(best['train_trades'])}",
        f"验证集: ret={best['ret']:.2%}, max_dd={best['max_dd']:.2%}, trades={int(best['trades'])}, score={best['score']:.2f}",
        f"邻域扰动(±10%, 验证集): ret 范围 [{perturb_df['ret'].min():.2%}, {perturb_df['ret'].max():.2%}], "
        f"均值 {perturb_df['ret'].mean():.2%}",
        ("警告: 参数邻域收益塌方 — 该最优点大概率是噪声，不要使用。" if fragile
         else "邻域稳定性: 通过（±10% 扰动未导致收益塌方）。"),
        "注意: 只有验证集数字具备参考意义；训练集数字含选择偏差。",
    ])
    (outdir / "best_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    shutil.rmtree(scratch, ignore_errors=True)
    return best.to_dict()


def parse_args():
    ap = argparse.ArgumentParser(description="Random search with held-out validation and perturbation test.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--output_dir", default="outputs/auto_opt")
    ap.add_argument("--max_iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.7, help="Fraction of trading days used for search.")
    ap.add_argument("--top_k", type=int, default=10, help="Train candidates promoted to validation.")
    ap.add_argument("--min_trades", type=int, default=5, help="Min train trades for a candidate to qualify.")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    auto_opt(args.feed, outdir, args.max_iters, args.seed, args.train_frac, args.top_k, args.min_trades)


if __name__ == "__main__":
    main()
