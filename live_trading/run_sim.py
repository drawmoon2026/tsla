"""
End-to-end simulation wiring using existing 5m dataset as feed:
- Build 1H bars from 5m via src.common.data_io (bucket-START labels, no
  cross-day buckets, partial buckets dropped).
- Generate signals from the previous 1H intraday close-to-close move
  (overnight gap excluded).
- Execute inside the NEXT 1H bucket's 5m bars; TP/SL settle through the
  shared pessimistic bracket model (gap-through stops, strict-cross TP,
  SL priority, per-side fees).
This is a dry-run scaffold for real-time deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# ensure project root on path when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_trading.bar_builder import resample_1h
from live_trading.config import Config, load_config
from live_trading.signal import compute_trigger
from src.common.data_io import load_bars
from src.common.execution import CostModel, entry_fill, settle_bracket

BUCKET = pd.Timedelta(minutes=60)


def run_sim(cfg: Config, feed_path: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_bars(feed_path)
    bars1h = resample_1h(df)  # bucket-START labelled, Close_ret is overnight-free
    bars1h["ret_prev"] = bars1h["Close_ret"]
    # ret_prev is NaN on each day's first bucket (overnight gap removed); allow
    # the rolling std to tolerate those NaNs instead of returning all-NaN
    bars1h["ret_std"] = bars1h["ret_prev"].rolling(
        cfg.std_window, min_periods=max(2, int(cfg.std_window * 0.7))
    ).std()

    cost = CostModel(fee_bp=cfg.fee_bp, slippage_bp=cfg.slip_bp)

    trades = []
    equity = cfg.capital
    peak_equity = equity
    tz = cfg.tz

    # account-level risk state
    cur_day = None
    day_start_equity = equity
    day_blocked = False
    global_stop_hit = False

    # iterate over 1H buckets: signal = bucket i-1, execution = bucket i's 5m bars
    for i in range(1, len(bars1h)):
        prev = bars1h.iloc[i - 1]
        if cfg.trigger_std is not None:
            thr = cfg.trigger_std * prev["ret_std"]
            direction = 0 if pd.isna(thr) or pd.isna(prev["ret_prev"]) or abs(prev["ret_prev"]) < thr \
                else (1 if prev["ret_prev"] > 0 else -1)
        else:
            direction = compute_trigger(prev, cfg.trigger)
        if direction == 0:
            continue

        window_start = bars1h.index[i]
        # execution bucket must immediately follow the signal bucket (same day)
        if window_start - bars1h.index[i - 1] != BUCKET:
            continue

        # time filter: ET hour of the execution bucket START
        et_start = window_start.tz_convert(tz)
        if et_start.hour not in cfg.allowed_hours:
            continue

        # account risk: reset daily tracking on new ET day
        if et_start.date() != cur_day:
            cur_day = et_start.date()
            day_start_equity = equity
            day_blocked = False

        # pre-entry checks: global drawdown stop, then daily loss stop
        if (equity - peak_equity) / peak_equity <= -cfg.global_stop:
            global_stop_hit = True
            break
        if day_blocked:
            continue

        # execution window: all 5m bars inside the next 1H bucket
        window = df.loc[(df.index >= window_start) & (df.index < window_start + BUCKET)]
        if window.empty:
            continue

        entry_price = entry_fill(float(window.iloc[0]["Open"]), direction, cost)
        tp_px = entry_price * (1 + cfg.tp * direction)
        sl_px = entry_price * (1 - cfg.sl * direction)

        res = settle_bracket(window, direction, entry_price, tp_px, sl_px, cost)

        ret = res.ret
        eq_before = equity
        equity *= (1 + ret)
        peak_equity = max(peak_equity, equity)
        if (equity - day_start_equity) / day_start_equity <= -cfg.daily_loss_stop:
            day_blocked = True
        trades.append(
            {
                "entry_time": window.index[0],
                "exit_time": res.exit_time,
                "direction": direction,
                "hit": res.hit,
                "entry": entry_price,
                "exit": res.exit_px,
                "ret": ret,
                "eq_before": eq_before,
                "eq_after": equity,
            }
        )

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(outdir / "sim_trades.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    if not trades_df.empty:
        total_return = equity / cfg.capital - 1
        summary = (
            f"Trades: {len(trades_df)}, total return: {total_return:.2%}, "
            f"win rate: {(trades_df['ret']>0).mean():.2%}"
        )
    else:
        total_return = 0.0
        summary = "No trades generated."
    if global_stop_hit:
        summary += " | GLOBAL STOP HIT"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return trades_df, total_return


def parse_args():
    ap = argparse.ArgumentParser(description="Simulate live trading scaffold on historical feed.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv", help="Lower timeframe feed (e.g., 5m CSV).")
    ap.add_argument("--output_dir", default="outputs/live_sim")
    ap.add_argument("--trigger", type=float, default=None, help="Override trigger (abs 1H ret).")
    ap.add_argument("--tp", type=float, default=None, help="Override TP fraction.")
    ap.add_argument("--sl", type=float, default=None, help="Override SL fraction.")
    ap.add_argument("--slip_bp", type=float, default=None, help="Override slippage bp.")
    ap.add_argument("--fee_bp", type=float, default=None, help="Override fee bp per side.")
    ap.add_argument("--trigger_std", type=float, default=None, help="Use k * rolling std of 1H returns.")
    ap.add_argument("--std_window", type=int, default=None, help="Rolling window for std trigger.")
    ap.add_argument("--no_report", action="store_true", help="Skip writing detailed report.txt")
    return ap.parse_args()


def write_report(trades_df: pd.DataFrame, capital: float, outdir: Path):
    if trades_df.empty:
        (outdir / "report.txt").write_text("No trades generated.", encoding="utf-8")
        return
    trades_df = trades_df.copy()
    trades_df["cum_equity"] = (1 + trades_df["ret"]).cumprod() * capital
    cum = trades_df["cum_equity"]
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max

    stats = {
        "trades": len(trades_df),
        "win_rate": (trades_df["ret"] > 0).mean(),
        "tp_rate": (trades_df["hit"] == "tp").mean(),
        "sl_rate": (trades_df["hit"] == "sl").mean(),
        "avg_ret": trades_df["ret"].mean(),
        "median_ret": trades_df["ret"].median(),
        "max_gain": trades_df["ret"].max(),
        "max_loss": trades_df["ret"].min(),
        "total_return": trades_df["cum_equity"].iloc[-1] / capital - 1,
        "max_dd": dd.min(),
    }
    trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
    monthly = trades_df.groupby("month")["ret"].apply(lambda x: (1 + x).prod() - 1)

    lines = []
    lines.append(f"起始本金: ${capital:,.0f}")
    lines.append(f"交易笔数: {stats['trades']}")
    lines.append(f"总收益: {stats['total_return']*100:.2f}% (期末: ${trades_df['cum_equity'].iloc[-1]:,.0f})")
    lines.append(f"胜率: {stats['win_rate']*100:.1f}% | TP率: {stats['tp_rate']*100:.1f}% | SL率: {stats['sl_rate']*100:.1f}%")
    lines.append(f"单笔平均/中位收益: {stats['avg_ret']*100:.2f}% / {stats['median_ret']*100:.2f}%")
    lines.append(f"最大单笔涨跌: {stats['max_gain']*100:.2f}% / {stats['max_loss']*100:.2f}%")
    lines.append(f"最大回撤: {stats['max_dd']*100:.2f}%")
    lines.append("\n月度收益:")
    lines.append(monthly.to_string())

    (outdir / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    # also export stakes/pnl for便于查看投入
    trades_df["stake"] = trades_df["eq_before"]
    trades_df["pnl_dollar"] = trades_df["eq_after"] - trades_df["eq_before"]
    trades_df["pnl_pct"] = trades_df["ret"] * 100
    trades_df[["entry_time", "entry", "exit", "stake", "pnl_dollar", "pnl_pct", "hit"]].to_csv(
        outdir / "trades_with_stake.csv", index=False, date_format="%Y-%m-%d %H:%M:%S"
    )


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    if args.trigger is not None:
        cfg.trigger = args.trigger
    if args.tp is not None:
        cfg.tp = args.tp
    if args.sl is not None:
        cfg.sl = args.sl
    if args.slip_bp is not None:
        cfg.slip_bp = args.slip_bp
    if args.fee_bp is not None:
        cfg.fee_bp = args.fee_bp
    if args.trigger_std is not None:
        cfg.trigger_std = args.trigger_std
    if args.std_window is not None:
        cfg.std_window = args.std_window
    cfg.validate()
    trades_df, _ = run_sim(cfg, args.feed, outdir)
    if not args.no_report:
        write_report(trades_df, cfg.capital, outdir)


if __name__ == "__main__":
    main()
