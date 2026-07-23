import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf


def build_output_path(symbol: str, interval: str, period: str) -> Path:
    """Create a predictable output path like data/TSLA_5m_60d.csv."""
    safe_symbol = symbol.upper()
    safe_interval = interval.replace(" ", "")
    safe_period = period.replace(" ", "")
    return Path("data") / f"{safe_symbol}_{safe_interval}_{safe_period}.csv"


def normalize_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ensure columns are a flat index with OHLCV for the requested symbol.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer selecting by symbol level if present.
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0, drop_level=True)
        else:
            # Fallback: flatten all levels into single strings.
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.to_flat_index()]
    return df


def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance.")

    df = normalize_columns(df, symbol)

    # Ensure timezone-aware index; yfinance typically provides UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols)

    # Add Eastern time column for grouping by trading day.
    eastern = pytz.timezone("America/New_York")
    df["time_et"] = df.index.tz_convert(eastern)
    return df


def load_cached(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if "time_et" not in df.columns:
        df["time_et"] = df.index.tz_convert("America/New_York")
    return df


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, date_format="%Y-%m-%dT%H:%M:%S%z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TSLA 5m data and cache to CSV.")
    parser.add_argument("--symbol", default="TSLA", help="Ticker symbol (default: TSLA)")
    parser.add_argument("--period", default="60d", help='Period to request, e.g., "60d"')
    parser.add_argument("--interval", default="5m", help='Bar interval, e.g., "5m"')
    parser.add_argument("--refresh", action="store_true", help="Force re-download even if CSV exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_output_path(args.symbol, args.interval, args.period)

    if not args.refresh:
        cached = load_cached(output_path)
    else:
        cached = None

    if cached is not None:
        df = cached
        source = "cache"
    else:
        df = fetch_data(args.symbol, args.period, args.interval)
        save_data(df, output_path)
        source = "download"

    print(f"Source: {source}")
    print(f"Rows: {len(df)}")
    print(f"Index timezone: {df.index.tz}")
    print(f"UTC range: {df.index.min()} -> {df.index.max()}")
    print(f"ET range: {df['time_et'].min()} -> {df['time_et'].max()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
