"""
Executable 1H breakout-follow strategy (no lookahead on direction):
- Load/resample via src.common.data_io: 5m -> 1H buckets anchored to each ET
  trading day's 9:30 open, labelled by bucket START, partial buckets dropped.
- Signal: PREVIOUS completed 1H bar's intraday close-to-close return magnitude
  >= trigger -> enter at the NEXT bar's open (signal confirmed strictly before
  entry; overnight gaps never generate signals, and signals from the previous
  day's last bar are not carried across the overnight gap).
- Exit: TP/SL settled through src.common.execution.settle_bracket on the entry
  bar itself (1H-bar-granularity backtest: the holding window is that one bar).
  Pessimistic fills: gap-through stops at open, strict-cross limits, SL wins on
  same-bar double touch, per-side fees + adverse slippage.
- Grid search over trigger / TP / SL; report best equity (in-sample selection).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ensure project root on path when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, intraday_returns, load_bars, resample_bars
from src.common.execution import CostModel, entry_fill, settle_bracket


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry: float
    exit: float
    ret: float
    hit: str  # tp / sl / close
    eq_before: float
    eq_after: float
    pnl_dollar: float


def simulate(
    bars: pd.DataFrame,
    trigger: float,
    tp: float,
    sl: float,
    cost: CostModel,
    capital: float = 10_000.0,
) -> Tuple[List[Trade], float]:
    ret = intraday_returns(bars["Close"])
    sig = ret.shift(1)  # signal = previous COMPLETED bar's return (known at bar open)
    et_date = bars.index.tz_convert(ET).date
    trades: List[Trade] = []
    eq = capital

    for i in range(1, len(bars)):
        prev_r = sig.iloc[i]
        if pd.isna(prev_r) or abs(prev_r) < trigger:
            continue
        if et_date[i] != et_date[i - 1]:
            # signal bar belongs to the previous trading day: stale, skip
            continue

        direction = 1 if prev_r > 0 else -1
        raw_open = float(bars["Open"].iloc[i])
        entry_px = entry_fill(raw_open, direction, cost)
        tp_px = entry_px * (1 + tp * direction)
        sl_px = entry_px * (1 - sl * direction)

        res = settle_bracket(bars.iloc[i : i + 1], direction, entry_px, tp_px, sl_px, cost)

        eq_before = eq
        eq_after = eq * (1 + res.ret)
        pnl_dollar = eq_after - eq_before
        eq = eq_after
        trades.append(
            Trade(
                entry_time=bars.index[i],
                exit_time=res.exit_time,
                direction=direction,
                entry=entry_px,
                exit=res.exit_px,
                ret=res.ret,
                hit=res.hit,
                eq_before=eq_before,
                eq_after=eq_after,
                pnl_dollar=pnl_dollar,
            )
        )
    total_return = eq / capital - 1
    return trades, total_return


def simulate_dyn(
    bars: pd.DataFrame,
    trigger_k: float,
    std_window: int,
    tp: float,
    sl: float,
    cost: CostModel,
    capital: float = 10_000.0,
) -> Tuple[List[Trade], float]:
    ret = intraday_returns(bars["Close"])
    sig = ret.shift(1)
    # min_periods: intraday_returns leaves a NaN on each day's first bar, so a
    # full-window requirement would never be met; std over available obs only.
    vol = ret.rolling(std_window, min_periods=max(2, std_window // 2)).std().shift(1)
    et_date = bars.index.tz_convert(ET).date
    trades: List[Trade] = []
    eq = capital

    for i in range(1, len(bars)):
        prev_r = sig.iloc[i]
        thr = trigger_k * vol.iloc[i]
        if pd.isna(prev_r) or pd.isna(thr) or abs(prev_r) < thr:
            continue
        if et_date[i] != et_date[i - 1]:
            continue

        direction = 1 if prev_r > 0 else -1
        raw_open = float(bars["Open"].iloc[i])
        entry_px = entry_fill(raw_open, direction, cost)
        tp_px = entry_px * (1 + tp * direction)
        sl_px = entry_px * (1 - sl * direction)

        res = settle_bracket(bars.iloc[i : i + 1], direction, entry_px, tp_px, sl_px, cost)

        eq_before = eq
        eq_after = eq * (1 + res.ret)
        pnl_dollar = eq_after - eq_before
        eq = eq_after
        trades.append(
            Trade(
                entry_time=bars.index[i],
                exit_time=res.exit_time,
                direction=direction,
                entry=entry_px,
                exit=res.exit_px,
                ret=res.ret,
                hit=res.hit,
                eq_before=eq_before,
                eq_after=eq_after,
                pnl_dollar=pnl_dollar,
            )
        )

    total_return = eq / capital - 1
    return trades, total_return


def grid_search(bars: pd.DataFrame, triggers, tps, sls, cost: CostModel) -> pd.DataFrame:
    rows = []
    for trig in triggers:
        for tp in tps:
            for sl in sls:
                trades, tot = simulate(bars, trig, tp, sl, cost)
                rows.append(
                    {
                        "trigger": trig,
                        "tp": tp,
                        "sl": sl,
                        "trades": len(trades),
                        "total_return": tot,
                        "hit_tp_pct": np.mean([t.hit == "tp" for t in trades]) if trades else 0,
                        "hit_sl_pct": np.mean([t.hit == "sl" for t in trades]) if trades else 0,
                        "avg_ret": np.mean([t.ret for t in trades]) if trades else 0,
                    }
                )
    return pd.DataFrame(rows)


def walk_forward(
    bars: pd.DataFrame,
    triggers,
    tps,
    sls,
    cost: CostModel,
    train_days: int = 40,
    test_days: int = 20,
    min_trades: int = 5,
    capital: float = 10_000.0,
) -> Tuple[pd.DataFrame, List[Trade], float]:
    """Rolling walk-forward: pick params on `train_days` trading days, apply
    them untouched to the next `test_days`, roll by `test_days`. Only the
    out-of-sample (test) results are reported. Best-by-return selection is
    gated on `min_trades` in-sample trades so a 2-trade fluke can't win."""
    days = sorted(pd.unique(bars.index.tz_convert(ET).date))
    if len(days) <= train_days:
        raise ValueError(
            f"walk-forward needs > {train_days} trading days, got {len(days)} — "
            "fetch deeper history or lower --train_days"
        )
    et_date = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)

    folds = []
    oos_trades: List[Trade] = []
    eq = capital
    start = 0
    fold_no = 0
    while start + train_days < len(days):
        train_set = set(days[start : start + train_days])
        test_set = set(days[start + train_days : start + train_days + test_days])
        train_bars = bars[et_date.isin(train_set).values]
        test_bars = bars[et_date.isin(test_set).values]
        if test_bars.empty:
            break
        fold_no += 1

        grid = grid_search(train_bars, triggers, tps, sls, cost)
        eligible = grid[grid["trades"] >= min_trades]
        pick_from = eligible if not eligible.empty else grid
        best = pick_from.sort_values("total_return", ascending=False).iloc[0]

        trades, test_ret = simulate(
            test_bars, best.trigger, best.tp, best.sl, cost, capital=eq
        )
        eq *= 1 + test_ret
        oos_trades.extend(trades)
        folds.append(
            {
                "fold": fold_no,
                "train_start": min(train_set),
                "test_start": min(test_set),
                "test_end": max(test_set),
                "trigger": best.trigger,
                "tp": best.tp,
                "sl": best.sl,
                "train_return": best.total_return,
                "train_trades": int(best.trades),
                "test_return": test_ret,
                "test_trades": len(trades),
            }
        )
        start += test_days

    oos_total = eq / capital - 1
    return pd.DataFrame(folds), oos_trades, oos_total


def save_trades(trades: List[Trade], path: Path):
    if not trades:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(
        [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry": t.entry,
                "exit": t.exit,
                "ret": t.ret,
                "hit": t.hit,
                "eq_before": t.eq_before,
                "eq_after": t.eq_after,
                "pnl_dollar": t.pnl_dollar,
            }
            for t in trades
        ]
    )
    df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")


def make_report(
    trades: List[Trade],
    total_return: float,
    capital: float,
    params: dict | None = None,
    cost: CostModel | None = None,
) -> str:
    if not trades:
        return "No trades generated."

    rets = np.array([t.ret for t in trades])
    tp_rate = np.mean([t.hit == "tp" for t in trades])
    sl_rate = np.mean([t.hit == "sl" for t in trades])
    win_rate = np.mean(rets > 0)
    cum_eq = np.array([t.eq_after for t in trades])
    roll_max = np.maximum.accumulate(cum_eq)
    max_dd = np.min((cum_eq - roll_max) / roll_max)

    lines = []
    lines.append("1H 实盘可执行信号回测报告（样本内选参，含交易成本）")
    if params:
        lines.append(f"参数: trigger={params['trigger']:.3f}, tp={params['tp']:.3f}, sl={params['sl']:.3f}")
    if cost:
        lines.append(f"成本假设: 单边手续费 {cost.fee_bp:.1f}bp + 不利滑点 {cost.slippage_bp:.1f}bp（入场与止损出场）")
    lines.append(f"起始本金: ${capital:,.0f}")
    lines.append(f"交易笔数: {len(trades)}")
    lines.append(f"总收益: {total_return*100:.2f}%  (期末本金: ${capital*(1+total_return):,.0f})")
    lines.append(f"胜率: {win_rate*100:.1f}%, TP命中: {tp_rate*100:.1f}%, SL命中: {sl_rate*100:.1f}%")
    lines.append(f"单笔平均/中位收益: {rets.mean()*100:.2f}% / {np.median(rets)*100:.2f}%")
    lines.append(f"最大单笔涨跌: {rets.max()*100:.2f}% / {rets.min()*100:.2f}%")
    lines.append(f"最大回撤: {max_dd*100:.2f}%")
    lines.append(f"单笔等价资金: 全仓 ${capital:,.0f}，逐笔复利滚动；单笔盈亏中位 ${np.median([t.pnl_dollar for t in trades]):,.0f}")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="Executable 1H large-move capture backtest")
    ap.add_argument("--input_csv", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--trigger", type=float, default=None, help="Single trigger; if omitted run grid.")
    ap.add_argument("--tp", type=float, default=None, help="TP in fraction; if omitted run grid.")
    ap.add_argument("--sl", type=float, default=None, help="SL in fraction; if omitted run grid.")
    ap.add_argument("--output_dir", default="outputs/hourly_signal")
    ap.add_argument("--capital", type=float, default=10_000.0, help="Starting capital for PnL calc (default 10k)")
    ap.add_argument("--trigger_std", type=float, default=None, help="k * rolling std of 1H returns as dynamic trigger.")
    ap.add_argument("--std_window", type=int, default=20, help="Rolling window for std-based trigger.")
    ap.add_argument("--fee_bp", type=float, default=1.0, help="Per-side fee in basis points (default 1).")
    ap.add_argument("--slippage_bp", type=float, default=2.0, help="Adverse slippage in basis points (default 2).")
    ap.add_argument("--walkforward", action="store_true",
                    help="Rolling walk-forward: select params on train window, report test window only.")
    ap.add_argument("--train_days", type=int, default=40, help="Walk-forward train window (trading days).")
    ap.add_argument("--test_days", type=int, default=20, help="Walk-forward test window (trading days).")
    ap.add_argument("--min_trades", type=int, default=5, help="Min in-sample trades for a param set to be selectable.")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df5 = load_bars(args.input_csv)
    bars = resample_bars(df5, 60)
    cost = CostModel(fee_bp=args.fee_bp, slippage_bp=args.slippage_bp)

    if args.walkforward:
        triggers = [0.01, 0.015, 0.02]
        tps = [0.01, 0.015, 0.02]
        sls = [0.005, 0.0075, 0.01]
        folds, oos_trades, oos_total = walk_forward(
            bars, triggers, tps, sls, cost,
            train_days=args.train_days, test_days=args.test_days,
            min_trades=args.min_trades, capital=args.capital,
        )
        folds.to_csv(outdir / "wf_folds.csv", index=False)
        save_trades(oos_trades, outdir / "trades_wf_oos.csv")
        report = make_report(oos_trades, oos_total, args.capital, cost=cost)
        report = report.replace("样本内选参", "walk-forward 样本外")
        (outdir / "report_wf.txt").write_text(report)
        print(folds.to_string(index=False))
        print(f"Out-of-sample total return: {oos_total:.2%}, trades: {len(oos_trades)}")
    elif args.trigger is not None and args.tp is not None and args.sl is not None:
        trades, tot = simulate(bars, args.trigger, args.tp, args.sl, cost, capital=args.capital)
        save_trades(trades, outdir / "trades.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(make_report(trades, tot, args.capital, cost=cost))
        print(f"Trades: {len(trades)}, total return: {tot:.2%}")
    elif args.trigger_std is not None:
        # dynamic trigger based on rolling std
        trades, tot = simulate_dyn(
            bars, args.trigger_std, args.std_window, tp=0.025, sl=0.01, cost=cost, capital=args.capital
        )
        save_trades(trades, outdir / "trades_dyn.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(
            make_report(trades, tot, args.capital, {"trigger": args.trigger_std, "tp": 0.025, "sl": 0.01}, cost=cost)
        )
        print(f"Trades: {len(trades)}, total return: {tot:.2%}")
    else:
        triggers = [0.01, 0.015, 0.02]
        tps = [0.01, 0.015, 0.02]
        sls = [0.005, 0.0075, 0.01]
        grid = grid_search(bars, triggers, tps, sls, cost)
        grid = grid.sort_values("total_return", ascending=False)
        grid.to_csv(outdir / "grid_results.csv", index=False)
        best = grid.iloc[0]
        trades, tot = simulate(bars, best.trigger, best.tp, best.sl, cost, capital=args.capital)
        save_trades(trades, outdir / "trades_best.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(make_report(trades, tot, args.capital, best.to_dict(), cost=cost))
        print("Best params:", best.to_dict())
        print(f"Total return: {tot:.2%}, trades: {len(trades)}")


if __name__ == "__main__":
    main()
