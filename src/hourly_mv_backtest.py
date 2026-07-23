"""
Capture-only backtest on 1H bars to verify large-move payoffs.

Logic:
- Resample 5m data to 1H aligned to 9:30 ET buckets (offset=30min).
- Compute close-to-close returns; select bars with abs(ret) >= threshold (default 2%).
- For each selected bar, assume perfect capture of that move in the correct direction
  (long on up bars, short on down bars) with full capital.
- Metrics: count, signed sum, abs sum, compounded abs return, equity curve.
- Outputs: trades CSV, equity PNG, return histogram PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_5m(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    # Align buckets to start at 9:30 (offset=30min) to match RTH hour blocks.
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    df1 = df.resample("1h", offset="30min", label="right", closed="right").agg(agg)
    return df1.dropna()


def backtest(df1: pd.DataFrame, threshold: float, bars: int | None, capital: float = 10_000.0):
    ret = df1["Close"].pct_change().dropna()
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
    plt.title("Equity curve (capture abs moves)")
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
    plt.title("1H returns distribution (selected)")
    plt.xlabel("Return")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1H large-move capture backtest")
    p.add_argument("--input_csv", default="data/TSLA_5m_60d.csv")
    p.add_argument("--threshold", type=float, default=0.02, help="Abs return threshold (default 0.02 = 2%)")
    p.add_argument("--bars", type=int, default=None, help="Use last N 1H bars (e.g., 30). None = all.")
    p.add_argument("--output_dir", default="outputs/hourly_mv")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df5 = load_5m(args.input_csv)
    df1 = resample_1h(df5)

    trades, equity, stats = backtest(df1, args.threshold, args.bars)

    trades_path = outdir / "trades_hourly.csv"
    equity_path = outdir / "equity_hourly.png"
    hist_path = outdir / "ret_hist_hourly.png"
    trades.to_csv(trades_path, date_format="%Y-%m-%dT%H:%M:%S%z")
    plot_equity(equity, equity_path)
    plot_hist(trades, hist_path)

    print(f"Bars total {stats['bars_total']}, selected >= {args.threshold:.2%}: {stats['bars_selected']}")
    print(f"Signed sum: {stats['signed_sum']:.4f}, Abs sum: {stats['abs_sum']:.4f}")
    print(f"Compounded abs return: {stats['compounded_abs']:.2%} (vs earlier target ~89.6% sum, ~141.6% compounded)")
    print(f"Trades saved: {trades_path}")
    print(f"Equity plot: {equity_path}")
    print(f"Hist plot: {hist_path}")


if __name__ == "__main__":
    main()
