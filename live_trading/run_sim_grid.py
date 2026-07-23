"""
Grid + Martingale simulator on 5m feed, using 1H signal trigger.
- Signal: previous 1H |ret| >= trigger -> direction of that return.
- Execution window: the next 1H sliced into 5m bars; grid adds on price steps.
- TP: exit all when avg_cost moves +tp_pct in direction.
- SL: exit all when price moves -stop_pct from avg_cost.
- Timeout: after timeout_bars since first fill, exit at bar close.

Configurable in live_trading/config.py (grid_* fields).
Outputs: trades CSV and summary.txt under chosen output_dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_trading.bar_builder import resample_1h
from live_trading.config import Config, load_config
from live_trading.signal import compute_trigger


def load_feed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def simulate_grid(window: pd.DataFrame, direction: int, cfg: Config, capital: float, equity: float):
    """
    Simulate grid fills inside a 1H window of 5m bars.
    Returns trade dict or None if no fill.
    """
    steps = cfg.grid_steps
    mults = cfg.grid_mults
    tp_pct = cfg.grid_tp
    stop_pct = cfg.grid_stop
    timeout = cfg.grid_timeout_bars
    slip = cfg.slip_bp / 10_000

    anchor = window.iloc[0]["Open"]
    levels = [anchor * (1 + s * direction) for s in steps]

    size = 0.0
    cash = 0.0
    first_fill_idx = None
    filled_levels: List[int] = []

    for bar_idx, (ts, row) in enumerate(window.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]

        # check new fills
        for li, price in enumerate(levels):
            if li in filled_levels:
                continue
            hit = (l <= price <= h) if direction == 1 else (l <= price <= h)
            if hit:
                # leverage check: (current notional + new) / capital <= max_leverage
                prospective_notional = (cash + mults[li] * price) / capital
                if prospective_notional > cfg.max_leverage:
                    continue
                fill_price = price * (1 + slip * direction)
                qty = mults[li]
                cash += qty * fill_price
                size += qty
                filled_levels.append(li)
                if first_fill_idx is None:
                    first_fill_idx = bar_idx

        if size == 0:
            continue

        avg_cost = cash / size
        tp_price = avg_cost * (1 + tp_pct * direction)
        sl_price = avg_cost * (1 - stop_pct * direction)

        # evaluate intrabar TP/SL conservatively (SL first)
        tp_hit = h >= tp_price if direction == 1 else l <= tp_price
        sl_hit = l <= sl_price if direction == 1 else h >= sl_price
        if sl_hit:
            exit_px = sl_price
            pnl = direction * (exit_px - avg_cost) / avg_cost
            return {
                "hit": "sl",
                "entry_price": avg_cost,
                "exit_price": exit_px,
                "ret": pnl,
                "bars_held": bar_idx - (first_fill_idx or 0) + 1,
            }
        if tp_hit:
            exit_px = tp_price
            pnl = direction * (exit_px - avg_cost) / avg_cost
            return {
                "hit": "tp",
                "entry_price": avg_cost,
                "exit_price": exit_px,
                "ret": pnl,
                "bars_held": bar_idx - (first_fill_idx or 0) + 1,
            }

        # timeout check at bar close
        if first_fill_idx is not None and (bar_idx - first_fill_idx + 1) >= timeout:
            exit_px = c
            pnl = direction * (exit_px - avg_cost) / avg_cost
            return {
                "hit": "timeout",
                "entry_price": avg_cost,
                "exit_price": exit_px,
                "ret": pnl,
                "bars_held": bar_idx - first_fill_idx + 1,
            }

    # end of window: flat at last close
    if size > 0:
        avg_cost = cash / size
        exit_px = window["Close"].iloc[-1]
        pnl = direction * (exit_px - avg_cost) / avg_cost
        return {
            "hit": "close",
            "entry_price": avg_cost,
            "exit_price": exit_px,
            "ret": pnl,
            "bars_held": len(window) - (first_fill_idx or 0),
        }
    return None


def run(cfg: Config, feed_path: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_feed(feed_path)
    bars1h = resample_1h(df, offset_minutes=cfg.offset_minutes)

    trades = []
    equity = cfg.capital
    tz = cfg.tz

    for i in range(1, len(bars1h)):
        prev = bars1h.iloc[i - 1]
        direction = compute_trigger(prev, cfg.trigger)
        if direction == 0:
            continue
        bar_end = bars1h.index[i]
        hour_et = bar_end.tz_convert(tz).hour
        if cfg.allowed_hours and hour_et not in cfg.allowed_hours:
            continue

        window_start = bars1h.index[i - 1]
        window_end = bars1h.index[i]
        window = df.loc[(df.index > window_start) & (df.index <= window_end)]
        if window.empty:
            continue

        fill = simulate_grid(window, direction, cfg, cfg.capital, equity)
        if not fill:
            continue
        ret = fill["ret"]
        eq_before = equity
        equity *= (1 + ret)
        trades.append(
            {
                "entry_time": window.index[0],
                "exit_time": window.index[min(len(window) - 1, fill["bars_held"] - 1 if fill["bars_held"] else 0)],
                "direction": direction,
                "hit": fill["hit"],
                "ret": ret,
                "eq_before": eq_before,
                "eq_after": equity,
                "bars_held": fill["bars_held"],
            }
        )
        # global stop: if drawdown exceeds limit, flatten and stop
        dd = (equity - max(cfg.capital, max(t["eq_after"] for t in trades))) / max(cfg.capital, max(t["eq_after"] for t in trades))
        if dd <= -cfg.global_stop:
            break

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(outdir / "trades_grid.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    total_return = equity / cfg.capital - 1
    summary = f"Trades: {len(trades_df)}, total return: {total_return:.2%}"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def parse_args():
    ap = argparse.ArgumentParser(description="Grid/Martingale simulator on 5m feed with 1H signals.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--output_dir", default="outputs/live_sim_grid_martin")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    outdir = Path(args.output_dir)
    run(cfg, args.feed, outdir)


if __name__ == "__main__":
    main()
