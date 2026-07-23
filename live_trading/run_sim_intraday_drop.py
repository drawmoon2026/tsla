"""
Intraday drawdown trigger:
- Track each RTH day (ET 9:30-16:00). When price falls >= drop_pct from day's high-so-far, enter long at next bar open.
- Exit with TP / SL (pct of entry) or at day close if neither hit.
- Uses 5m bars from CSV feed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_trading.config import load_config, Config
from src.common.execution import CostModel, entry_fill, settle_bracket


def load_feed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def run(cfg: Config, feed: str, outdir: Path, drop_pct: float):
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_feed(feed)
    # RTH filter 9:30-16:00 ET
    et = df.index.tz_convert(cfg.tz)
    rth = (et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))
    rth &= (et.hour < 16)
    df = df[rth]
    et = df.index.tz_convert(cfg.tz)

    trades = []
    equity = cfg.capital
    cost = CostModel(fee_bp=cfg.fee_bp, slippage_bp=cfg.slip_bp)

    for date, day in df.groupby(et.date):
        high_so_far = None
        pending_entry = False
        for i, (ts, row) in enumerate(day.iterrows()):
            close = row["Close"]
            # track the intraday high with bar Highs, not Closes
            high_so_far = row["High"] if high_so_far is None else max(high_so_far, row["High"])
            draw = (high_so_far - close) / high_so_far if high_so_far else 0
            if draw >= drop_pct:
                pending_entry = True
            if pending_entry and i < len(day) - 1:
                # enter next bar open
                nxt = day.iloc[i + 1]
                entry_time = day.index[i + 1]
                entry = entry_fill(float(nxt["Open"]), 1, cost)
                tp_px = entry * (1 + cfg.tp)
                sl_px = entry * (1 - cfg.sl)
                # settle TP/SL via the shared pessimistic bracket model
                # (gap-through stops fill at the open, strict-cross TP, SL priority)
                res = settle_bracket(day.iloc[i + 1:], 1, entry, tp_px, sl_px, cost)
                hit = "day_close" if res.hit == "close" else res.hit
                exit_px = res.exit_px
                exit_time = res.exit_time
                ret = res.ret
                eq_before = equity
                equity *= (1 + ret)
                trades.append(
                    {
                        "date": date,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry": entry,
                        "exit": exit_px,
                        "ret": ret,
                        "hit": hit,
                        "eq_before": eq_before,
                        "eq_after": equity,
                    }
                )
                break  # one trade per day

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(outdir / "trades_intraday_drop.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    total_return = equity / cfg.capital - 1
    summary = f"Trades: {len(trades_df)}, total return: {total_return:.2%}"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def parse_args():
    ap = argparse.ArgumentParser(description="Intraday drop-buy strategy on 5m bars.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--output_dir", default="outputs/live_sim_intraday_drop")
    ap.add_argument("--drop_pct", type=float, default=0.02, help="Trigger drop from day high")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    outdir = Path(args.output_dir)
    run(cfg, args.feed, outdir, args.drop_pct)


if __name__ == "__main__":
    main()
