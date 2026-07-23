"""
1-minute intrabar replay for the 1H breakout-follow strategy (no lookahead on direction).

Differences vs hourly_signal_backtest:
- Uses real 1m path inside each 1H bar to decide TP/SL touch order.
- Data fetch: yfinance 1m (Yahoo限制约30天)，默认 TSLA period=30d.
- Signal: previous 1H close-to-close abs return >= trigger -> enter next bar at its opening minute.
- Exit: scan 1m bars within the holding 1H window in time order; first touch of TP/SL exits; else exit at that hour's close.

Outputs: trades CSV, report.txt, equity.png in outputs/hourly_signal_1m/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry: float
    exit: float
    ret: float
    hit: str
    eq_before: float
    eq_after: float
    pnl_dollar: float


def fetch_1m(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1m", auto_adjust=True, progress=False, threads=True)
    if df.empty:
        raise RuntimeError("No 1m data fetched; Yahoo 1m仅支持约30天，请调整period。")
    # flatten possible MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns.to_flat_index()]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def resample_1h(df1m: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    bars = df1m.resample("1h", offset="30min", label="right", closed="right").agg(agg)
    return bars.dropna()


def simulate_intrabar(
    bars1h: pd.DataFrame,
    df1m: pd.DataFrame,
    trigger: float,
    tp: float,
    sl: float,
    capital: float = 10_000.0,
) -> Tuple[List[Trade], float]:
    ret_prev = bars1h["Close"].pct_change()
    trades: List[Trade] = []
    eq = capital

    for i in range(1, len(bars1h)):
        prev_r = ret_prev.iloc[i]
        if pd.isna(prev_r) or abs(prev_r) < trigger:
            continue

        direction = 1 if prev_r > 0 else -1
        bar_end = bars1h.index[i]
        bar_start = bar_end - pd.Timedelta(hours=1) + pd.Timedelta(minutes=1)  # first minute after previous bar end

        bar1m = df1m.loc[(df1m.index >= bar_start) & (df1m.index <= bar_end)]
        if bar1m.empty:
            continue

        entry = bar1m["Open"].iloc[0]
        tp_price = entry * (1 + tp * direction)
        sl_price = entry * (1 - sl * direction)

        exit_price = bar1m["Close"].iloc[-1]
        exit_time = bar1m.index[-1]
        hit = "close"

        # step through 1m bars to find first touch
        for _, row in bar1m.iterrows():
            high = row["High"]
            low = row["Low"]
            ts = row.name
            if direction == 1:
                tp_hit = high >= tp_price
                sl_hit = low <= sl_price
            else:
                tp_hit = low <= tp_price
                sl_hit = high >= sl_price

            if sl_hit and tp_hit:
                exit_price = sl_price
                hit = "sl"
                exit_time = ts
                break
            elif sl_hit:
                exit_price = sl_price
                hit = "sl"
                exit_time = ts
                break
            elif tp_hit:
                exit_price = tp_price
                hit = "tp"
                exit_time = ts
                break

        ret_trade = direction * (exit_price - entry) / entry
        eq_before = eq
        eq_after = eq * (1 + ret_trade)
        pnl_dollar = eq_after - eq_before
        eq = eq_after

        trades.append(
            Trade(
                entry_time=bar1m.index[0],
                exit_time=exit_time,
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


def make_report(trades: List[Trade], total_return: float, capital: float, params: dict) -> str:
    if not trades:
        return "无交易生成。"
    rets = np.array([t.ret for t in trades])
    tp_rate = np.mean([t.hit == "tp" for t in trades])
    sl_rate = np.mean([t.hit == "sl" for t in trades])
    win_rate = np.mean(rets > 0)
    cum_eq = np.array([t.eq_after for t in trades])
    roll_max = np.maximum.accumulate(cum_eq)
    max_dd = np.min((cum_eq - roll_max) / roll_max)

    lines = []
    lines.append("1m 重播版本（1H 信号，1m 执行）回测报告")
    lines.append(f"参数: trigger={params['trigger']:.3f}, tp={params['tp']:.3f}, sl={params['sl']:.3f}")
    lines.append(f"起始本金: ${capital:,.0f}")
    lines.append(f"交易笔数: {len(trades)}")
    lines.append(f"总收益: {total_return*100:.2f}%  (期末本金: ${capital*(1+total_return):,.0f})")
    lines.append(f"胜率: {win_rate*100:.1f}%, TP命中: {tp_rate*100:.1f}%, SL命中: {sl_rate*100:.1f}%")
    lines.append(f"单笔平均/中位收益: {rets.mean()*100:.2f}% / {np.median(rets)*100:.2f}%")
    lines.append(f"最大单笔涨跌: {rets.max()*100:.2f}% / {rets.min()*100:.2f}%")
    lines.append(f"最大回撤: {max_dd*100:.2f}%")
    lines.append(f"单笔等价资金: 全仓 ${capital:,.0f}，逐笔复利；单笔盈亏中位 ${np.median([t.pnl_dollar for t in trades]):,.0f}")
    return "\n".join(lines)


def plot_equity(trades: List[Trade], outpath: Path):
    if not trades:
        return
    eq = [t.eq_after for t in trades]
    ts = [t.exit_time for t in trades]
    plt.figure(figsize=(10, 4))
    plt.plot(ts, eq)
    plt.title("Equity curve (1m intrabar)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="1m intrabar replay for 1H signal strategy")
    ap.add_argument("--symbol", default="TSLA")
    ap.add_argument("--period", default="7d", help="yfinance period for 1m (<=8d recommended by Yahoo)")
    ap.add_argument("--trigger", type=float, default=0.01)
    ap.add_argument("--tp", type=float, default=0.02)
    ap.add_argument("--sl", type=float, default=0.01)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--output_dir", default="outputs/hourly_signal_1m")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df1m = fetch_1m(args.symbol, args.period)
    bars1h = resample_1h(df1m)

    trades, tot = simulate_intrabar(bars1h, df1m, args.trigger, args.tp, args.sl, capital=args.capital)

    trades_path = outdir / "trades_1m.csv"
    report_path = outdir / "report.txt"
    equity_path = outdir / "equity.png"

    pd.DataFrame([t.__dict__ for t in trades]).to_csv(trades_path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    report_path.write_text(
        make_report(
            trades,
            tot,
            args.capital,
            {"trigger": args.trigger, "tp": args.tp, "sl": args.sl},
        ),
        encoding="utf-8",
    )
    plot_equity(trades, equity_path)

    print(f"Trades: {len(trades)}, total return: {tot:.2%}")
    print(f"Files written to {outdir}")


if __name__ == "__main__":
    main()
