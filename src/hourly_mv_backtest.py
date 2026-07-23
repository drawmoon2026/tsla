"""
Capture-only backtest on 1H bars to verify large-move payoffs.

*** LOOK-AHEAD UPPER BOUND — NOT TRADABLE ***
This script assumes the direction of every large 1H move is known in advance
and captured in full (close-to-close). It is a theoretical ceiling for what a
perfect predictor could earn, NOT an executable strategy. Compare against
hourly_signal_backtest.py for the executable (costed, no-lookahead) version.

Logic:
- Load/resample via src.common.data_io: 5m -> 1H buckets anchored to each ET
  trading day's 9:30 open, labelled by bucket START, partial buckets dropped.
- Compute intraday close-to-close returns (overnight gaps excluded); select
  bars with abs(ret) >= threshold (default 2%).
- For each selected bar, assume perfect capture of that move in the correct
  direction (long on up bars, short on down bars) with full capital.
- Metrics: count, signed sum, abs sum, compounded abs return, equity curve.
- Outputs: trades CSV, equity PNG, return histogram PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# ensure project root on path when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import intraday_returns, load_bars, resample_bars  # noqa: E402

WARNING_BANNER = "LOOK-AHEAD UPPER BOUND — NOT TRADABLE"


def backtest(df1: pd.DataFrame, threshold: float, bars: int | None, capital: float = 10_000.0):
    ret = intraday_returns(df1["Close"]).dropna()
    if bars:
        ret = ret.iloc[-bars:]

    mask = ret.abs() >= threshold
    big = ret[mask]

    # equity on captured absolute moves
    equity = (1 + big.abs()).cumprod() * capital
    equity.index.name = "Datetime"

    # trades table
    trades = pd.DataFrame(
        {
            "return": big,
            "abs_return": big.abs(),
            "direction": big.apply(lambda x: "long" if x > 0 else "short"),
            "equity_after": equity,
        }
    )

    stats = {
        "bars_total": len(ret),
        "bars_selected": len(big),
        "signed_sum": big.sum(),
        "abs_sum": big.abs().sum(),
        "compounded_abs": equity.iloc[-1] / capital - 1 if not equity.empty else 0.0,
        "capital": capital,
    }
    return trades, equity, stats


def plot_equity(equity: pd.Series, outpath: Path):
    if equity.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values)
    plt.title(f"Equity curve (capture abs moves) — {WARNING_BANNER}")
    plt.ylabel("Equity")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_hist(trades: pd.DataFrame, outpath: Path):
    if trades.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(trades["return"], bins=40)
    plt.title(f"1H returns distribution (selected) — {WARNING_BANNER}")
    plt.xlabel("Return")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_trades(trades: pd.DataFrame, path: Path):
    # First line marks the file itself as a non-tradable upper bound
    # (readable back with pd.read_csv(path, comment="#")).
    with open(path, "w") as f:
        f.write(f"# {WARNING_BANNER}: perfect-foresight capture, no costs, not an executable strategy\n")
        trades.to_csv(f, date_format="%Y-%m-%dT%H:%M:%S%z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"1H large-move capture backtest ({WARNING_BANNER})")
    p.add_argument("--input_csv", default="data/TSLA_5m_60d.csv")
    p.add_argument("--threshold", type=float, default=0.02, help="Abs return threshold (default 0.02 = 2%)")
    p.add_argument("--bars", type=int, default=None, help="Use last N 1H bars (e.g., 30). None = all.")
    p.add_argument("--output_dir", default="outputs/hourly_mv")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df5 = load_bars(args.input_csv)
    df1 = resample_bars(df5, 60)

    trades, equity, stats = backtest(df1, args.threshold, args.bars)

    trades_path = outdir / "trades_hourly.csv"
    equity_path = outdir / "equity_hourly.png"
    hist_path = outdir / "ret_hist_hourly.png"
    save_trades(trades, trades_path)
    plot_equity(equity, equity_path)
    plot_hist(trades, hist_path)

    print(f"*** {WARNING_BANNER} ***")
    print("(perfect-foresight capture of every large move, no costs; theoretical ceiling only)")
    print(f"Bars total {stats['bars_total']}, selected >= {args.threshold:.2%}: {stats['bars_selected']}")
    print(f"Signed sum: {stats['signed_sum']:.4f}, Abs sum: {stats['abs_sum']:.4f}")
    print(f"Compounded abs return: {stats['compounded_abs']:.2%} (vs earlier target ~89.6% sum, ~141.6% compounded)")
    print(f"Trades saved: {trades_path}")
    print(f"Equity plot: {equity_path}")
    print(f"Hist plot: {hist_path}")
    print(f"*** {WARNING_BANNER} ***")


if __name__ == "__main__":
    main()
