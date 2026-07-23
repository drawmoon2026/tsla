"""
Executable 1H breakout-follow strategy (no lookahead on direction):
- Resample 5m to 1H aligned to 9:30 ET (offset 30min).
- Signal: if previous 1H close-to-close return magnitude >= trigger, enter next bar open in same direction.
- Exit: TP/SL intrabar using that bar's high/low. If both hit, assume worst-case (stop first) to avoid optimism.
- If neither TP nor SL hits, exit at bar close.
- Grid search over trigger / TP / SL; report best equity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_5m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    out = df.resample("1h", offset="30min", label="right", closed="right").agg(agg)
    return out.dropna()


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


def simulate(bars: pd.DataFrame, trigger: float, tp: float, sl: float, capital: float = 10_000.0) -> Tuple[List[Trade], float]:
    ret_prev = bars["Close"].pct_change()
    trades: List[Trade] = []
    eq = capital

    for i in range(1, len(bars)):
        prev_r = ret_prev.iloc[i]
        if pd.isna(prev_r):
            continue
        if abs(prev_r) < trigger:
            continue

        direction = 1 if prev_r > 0 else -1
        bar = bars.iloc[i]
        entry = bar["Open"]
        tp_price = entry * (1 + tp * direction)
        sl_price = entry * (1 - sl * direction)

        # Determine exit conservatively: if both TP & SL in range, assume SL hits first
        high, low, close = bar["High"], bar["Low"], bar["Close"]
        hit = "close"
        exit_price = close

        if direction == 1:
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price
        else:
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

        if tp_hit and sl_hit:
            # worst-case: stop first
            exit_price = sl_price
            hit = "sl"
        elif tp_hit:
            exit_price = tp_price
            hit = "tp"
        elif sl_hit:
            exit_price = sl_price
            hit = "sl"

        ret_trade = direction * (exit_price - entry) / entry
        eq_before = eq
        eq_after = eq * (1 + ret_trade)
        pnl_dollar = eq_after - eq_before
        eq = eq_after
        trades.append(
            Trade(
                entry_time=bars.index[i],
                exit_time=bars.index[i],
                direction=direction,
                entry=entry,
                exit=exit_price,
                ret=ret_trade,
                hit=hit,
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
    capital: float = 10_000.0,
) -> Tuple[List[Trade], float]:
    ret_prev = bars["Close"].pct_change()
    vol = ret_prev.rolling(std_window).std()
    trades: List[Trade] = []
    eq = capital

    for i in range(1, len(bars)):
        prev_r = ret_prev.iloc[i]
        thr = trigger_k * vol.iloc[i]
        if pd.isna(prev_r) or pd.isna(thr) or abs(prev_r) < thr:
            continue

        direction = 1 if prev_r > 0 else -1
        bar = bars.iloc[i]
        entry = bar["Open"]
        tp_price = entry * (1 + tp * direction)
        sl_price = entry * (1 - sl * direction)

        high, low, close = bar["High"], bar["Low"], bar["Close"]
        hit = "close"
        exit_price = close

        if direction == 1:
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price
        else:
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

        if tp_hit and sl_hit:
            exit_price = sl_price
            hit = "sl"
        elif tp_hit:
            exit_price = tp_price
            hit = "tp"
        elif sl_hit:
            exit_price = sl_price
            hit = "sl"

        ret_trade = direction * (exit_price - entry) / entry
        eq *= (1 + ret_trade)
        trades.append(
            Trade(
                entry_time=bars.index[i],
                exit_time=bars.index[i],
                direction=direction,
                entry=entry,
                exit=exit_price,
                ret=ret_trade,
                hit=hit,
                eq_before=eq / (1 + ret_trade),
                eq_after=eq,
                pnl_dollar=capital * ret_trade,
            )
        )

    total_return = eq / capital - 1
    return trades, total_return


def grid_search(bars: pd.DataFrame, triggers, tps, sls) -> pd.DataFrame:
    rows = []
    for trig in triggers:
        for tp in tps:
            for sl in sls:
                trades, tot = simulate(bars, trig, tp, sl)
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


def make_report(trades: List[Trade], total_return: float, capital: float, params: dict | None = None) -> str:
    if not trades:
        return "No trades generated."
    import numpy as np

    rets = np.array([t.ret for t in trades])
    tp_rate = np.mean([t.hit == "tp" for t in trades])
    sl_rate = np.mean([t.hit == "sl" for t in trades])
    win_rate = np.mean(rets > 0)
    cum_eq = np.array([t.eq_after for t in trades])
    roll_max = np.maximum.accumulate(cum_eq)
    max_dd = np.min((cum_eq - roll_max) / roll_max)

    lines = []
    lines.append("1H 实盘可执行信号回测报告")
    if params:
        lines.append(f"参数: trigger={params['trigger']:.3f}, tp={params['tp']:.3f}, sl={params['sl']:.3f}")
    lines.append(f"起始本金: ${capital:,.0f}")
    lines.append(f"交易笔数: {len(trades)}")
    lines.append(f"总收益: {total_return*100:.2f}%  (期末本金: ${capital*(1+total_return):,.0f})")
    lines.append(f"胜率: {win_rate*100:.1f}%, TP命中: {tp_rate*100:.1f}%, SL命中: {sl_rate*100:.1f}%")
    lines.append(f"单笔平均/中位收益: {rets.mean()*100:.2f}% / {np.median(rets)*100:.2f}%")
    lines.append(f"最大单笔涨跌: {rets.max()*100:.2f}% / {rets.min()*100:.2f}%")
    lines.append(f"最大回撤: {max_dd*100:.2f}%")
    lines.append(f"单笔等价资金: 全仓 ${capital:,.0f}，逐笔复利滚动；单笔盈亏中位 ${np.median([t.pnl_dollar for t in trades]):,.0f}")
    lines.append(f"情境化举例：\n  若起始 $10,000，首笔盈亏后滚动，最终约 ${capital*(1+total_return):,.0f}；\n  TP 一笔赚约 ${capital*0.02:,.0f}，SL 一笔亏约 ${capital*0.01:,.0f}（以当笔入场本金计）。")
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
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df5 = load_5m(args.input_csv)
    bars = resample_1h(df5)

    if args.trigger and args.tp and args.sl:
        trades, tot = simulate(bars, args.trigger, args.tp, args.sl, capital=args.capital)
        save_trades(trades, outdir / "trades.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(make_report(trades, tot, args.capital))
        print(f"Trades: {len(trades)}, total return: {tot:.2%}")
    elif args.trigger_std:
        # dynamic trigger based on rolling std
        trades, tot = simulate_dyn(bars, args.trigger_std, args.std_window, tp=0.025, sl=0.01, capital=args.capital)
        save_trades(trades, outdir / "trades_dyn.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(make_report(trades, tot, args.capital, {"trigger": args.trigger_std, "tp": 0.025, "sl": 0.01}))
        print(f"Trades: {len(trades)}, total return: {tot:.2%}")
    else:
        triggers = [0.01, 0.015, 0.02]
        tps = [0.01, 0.015, 0.02]
        sls = [0.005, 0.0075, 0.01]
        grid = grid_search(bars, triggers, tps, sls)
        grid = grid.sort_values("total_return", ascending=False)
        grid.to_csv(outdir / "grid_results.csv", index=False)
        best = grid.iloc[0]
        trades, tot = simulate(bars, best.trigger, best.tp, best.sl, capital=args.capital)
        save_trades(trades, outdir / "trades_best.csv")
        report_path = outdir / "report.txt"
        report_path.write_text(make_report(trades, tot, args.capital, best.to_dict()))
        print("Best params:", best.to_dict())
        print(f"Total return: {tot:.2%}, trades: {len(trades)}")


if __name__ == "__main__":
    main()
