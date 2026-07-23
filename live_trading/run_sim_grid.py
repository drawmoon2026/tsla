"""
Grid + Martingale simulator on 5m feed, using 1H signal trigger.
- Signal: previous 1H intraday |ret| >= trigger -> direction of that return.
- Execution window: the NEXT 1H bucket sliced into 5m bars; grid adds on
  adverse price steps, each level sized in DOLLARS
  (grid_base_dollars * grid_mults[i]); shares and cost tracked per fill.
- TP: exit all when price moves +tp_pct from average cost (strict cross).
- SL: exit all when price moves -stop_pct from average cost (stop-market:
  gap-through opens fill at the open, adverse slippage on top).
- Timeout: after timeout_bars since first fill, exit at bar close.
- PnL settles in dollars (fees per side on notional) and updates equity.
- --pessimistic: within a bar, check SL against the PRE-fill average cost
  first (worst-case ordering of fills vs. stop).

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
from src.common.data_io import load_bars
from src.common.execution import CostModel

BUCKET = pd.Timedelta(minutes=60)


def worst_case_loss(cfg: Config, cost: CostModel) -> float:
    """Worst single-trade dollar loss: full ladder filled, stopped at avg-cost
    stop, plus round-trip slippage and fees on the full notional."""
    full_invest = cfg.grid_base_dollars * sum(cfg.grid_mults)
    return full_invest * (cfg.grid_stop + 2 * cost.slip + 2 * cost.fee)


def simulate_grid(
    window: pd.DataFrame,
    direction: int,
    cfg: Config,
    equity: float,
    cost: CostModel,
    pessimistic: bool = False,
):
    """
    Simulate dollar-sized grid fills inside a 1H window of 5m bars.
    Returns trade dict (dollar PnL) or None if no fill.
    """
    steps = cfg.grid_steps  # adverse-move fractions for a long; sign flips for shorts
    mults = cfg.grid_mults
    base = cfg.grid_base_dollars
    tp_pct = cfg.grid_tp
    stop_pct = cfg.grid_stop
    timeout = cfg.grid_timeout_bars

    anchor = float(window.iloc[0]["Open"])
    # adverse move is DOWN for longs, UP for shorts: flip step sign via direction
    levels = [anchor * (1 + s * direction) for s in steps]

    shares = 0.0
    cost_dollars = 0.0  # sum of shares_i * fill_px
    invested = 0.0      # committed notional in dollars
    fees = 0.0
    first_fill_idx = None
    first_fill_ts = None
    filled_levels: List[int] = []

    def close_out(tag: str, exit_px: float, ts, bar_idx: int):
        exit_fees = shares * exit_px * cost.fee
        avg_cost = cost_dollars / shares
        pnl = direction * (exit_px - avg_cost) * shares - fees - exit_fees
        return {
            "hit": tag,
            "entry_price": avg_cost,
            "exit_price": exit_px,
            "pnl": pnl,
            "invested": invested,
            "shares": shares,
            "entry_time": first_fill_ts,
            "exit_time": ts,
            "bars_held": bar_idx - first_fill_idx + 1,
        }

    for bar_idx, (ts, row) in enumerate(window.iterrows()):
        o, h, l, c = (float(row[k]) for k in ("Open", "High", "Low", "Close"))

        # pessimistic ordering: stop is judged on the PRE-fill average cost
        # before this bar's grid adds are allowed to drag the average down
        if pessimistic and shares > 0:
            avg_cost = cost_dollars / shares
            sl_price = avg_cost * (1 - stop_pct * direction)
            if direction == 1:
                sl_hit = o <= sl_price or l <= sl_price
                sl_fill = min(o, sl_price) * (1 - cost.slip)
            else:
                sl_hit = o >= sl_price or h >= sl_price
                sl_fill = max(o, sl_price) * (1 + cost.slip)
            if sl_hit:
                return close_out("sl", sl_fill, ts, bar_idx)

        # check new fills (resting limit adds at grid levels)
        for li, price in enumerate(levels):
            if li in filled_levels:
                continue
            touched = (l <= price) if direction == 1 else (h >= price)
            if not touched:
                continue
            add_dollars = base * mults[li]
            # leverage check: (already invested + this add) / CURRENT equity
            if (invested + add_dollars) / equity > cfg.max_leverage:
                continue
            fill_px = price * (1 + cost.slip * direction)
            qty = add_dollars / fill_px
            shares += qty
            cost_dollars += qty * fill_px
            invested += add_dollars
            fees += add_dollars * cost.fee
            filled_levels.append(li)
            if first_fill_idx is None:
                first_fill_idx = bar_idx
                first_fill_ts = ts

        if shares == 0:
            continue

        avg_cost = cost_dollars / shares
        tp_price = avg_cost * (1 + tp_pct * direction)
        sl_price = avg_cost * (1 - stop_pct * direction)

        # pessimistic bracket on this bar: gap-through stop at open, strict
        # crossing for TP, SL priority when both touch
        if direction == 1:
            sl_hit = o <= sl_price or l <= sl_price
            tp_hit = o >= tp_price or h > tp_price
            sl_fill = min(o, sl_price) * (1 - cost.slip)
            tp_fill = max(o, tp_price)
        else:
            sl_hit = o >= sl_price or h >= sl_price
            tp_hit = o <= tp_price or l < tp_price
            sl_fill = max(o, sl_price) * (1 + cost.slip)
            tp_fill = min(o, tp_price)

        if sl_hit:
            return close_out("sl", sl_fill, ts, bar_idx)
        if tp_hit:
            return close_out("tp", tp_fill, ts, bar_idx)

        # timeout check at bar close (market exit, adverse slippage)
        if (bar_idx - first_fill_idx + 1) >= timeout:
            return close_out("timeout", c * (1 - cost.slip * direction), ts, bar_idx)

    # end of window: flat at last close (market exit, adverse slippage)
    if shares > 0:
        exit_px = float(window["Close"].iloc[-1]) * (1 - cost.slip * direction)
        return close_out("close", exit_px, window.index[-1], len(window) - 1)
    return None


def run(cfg: Config, feed_path: str, outdir: Path, pessimistic: bool = False):
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_bars(feed_path)
    bars1h = resample_1h(df)  # bucket-START labelled, Close_ret is overnight-free

    cost = CostModel(fee_bp=cfg.fee_bp, slippage_bp=cfg.slip_bp)

    # worst-case single-trade loss assertion (full ladder + avg-cost stop)
    wc_loss = worst_case_loss(cfg, cost)
    if wc_loss > cfg.capital * 0.5:
        print(
            "!" * 70 + "\n"
            f"WARNING: worst-case single-trade loss ${wc_loss:,.0f} exceeds 50% of "
            f"capital (${cfg.capital:,.0f}). Martingale ladder is oversized.\n" + "!" * 70
        )

    trades = []
    equity = cfg.capital
    peak_equity = equity
    global_stop_hit = False
    tz = cfg.tz

    for i in range(1, len(bars1h)):
        prev = bars1h.iloc[i - 1]
        direction = compute_trigger(prev, cfg.trigger)
        if direction == 0:
            continue

        window_start = bars1h.index[i]
        # execution bucket must immediately follow the signal bucket (same day)
        if window_start - bars1h.index[i - 1] != BUCKET:
            continue
        hour_et = window_start.tz_convert(tz).hour
        if cfg.allowed_hours and hour_et not in cfg.allowed_hours:
            continue

        # global drawdown circuit breaker: PRE-entry check
        if (equity - peak_equity) / peak_equity <= -cfg.global_stop:
            global_stop_hit = True
            break

        window = df.loc[(df.index >= window_start) & (df.index < window_start + BUCKET)]
        if window.empty:
            continue

        fill = simulate_grid(window, direction, cfg, equity, cost, pessimistic=pessimistic)
        if not fill:
            continue
        eq_before = equity
        equity += fill["pnl"]
        peak_equity = max(peak_equity, equity)
        trades.append(
            {
                "entry_time": fill["entry_time"],
                "exit_time": fill["exit_time"],
                "direction": direction,
                "hit": fill["hit"],
                "avg_cost": fill["entry_price"],
                "exit_price": fill["exit_price"],
                "invested": fill["invested"],
                "shares": fill["shares"],
                "pnl": fill["pnl"],
                "ret_on_equity": fill["pnl"] / eq_before,
                "eq_before": eq_before,
                "eq_after": equity,
                "bars_held": fill["bars_held"],
            }
        )

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(outdir / "trades_grid.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    total_return = equity / cfg.capital - 1
    summary = (
        f"Trades: {len(trades_df)}, total return: {total_return:.2%} | "
        f"mode: {'pessimistic' if pessimistic else 'optimistic'} intrabar ordering | "
        f"worst-case single-trade loss: ${wc_loss:,.0f} "
        f"({wc_loss / cfg.capital:.1%} of capital)"
    )
    if global_stop_hit:
        summary += " | GLOBAL STOP HIT"
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return trades_df, total_return


def parse_args():
    ap = argparse.ArgumentParser(description="Grid/Martingale simulator on 5m feed with 1H signals.")
    ap.add_argument("--feed", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--output_dir", default="outputs/live_sim_grid_martin")
    ap.add_argument(
        "--pessimistic",
        action="store_true",
        help="Within a bar, judge SL on the pre-fill average cost first (worst-case ordering).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    outdir = Path(args.output_dir)
    run(cfg, args.feed, outdir, pessimistic=args.pessimistic)


if __name__ == "__main__":
    main()
