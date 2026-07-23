"""Print basic statistics for the cached/downloaded TSLA 5m data.

Thin wrapper over data_fetch so the whole project shares one fetch convention
(America/New_York, auto_adjust=True) instead of a second divergent copy.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetch import build_output_path, fetch_data, load_cached, save_data


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[numeric_cols].describe().T


def main() -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)

    symbol, period, interval = "TSLA", "60d", "5m"
    path = build_output_path(symbol, interval, period)
    df = load_cached(path, max_age_days=3.0)
    if df is None:
        print(f"Downloading {symbol} {interval} data ({period})...")
        df = fetch_data(symbol, period, interval)
        save_data(df, path)

    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"Time range (ET): {df['time_et'].min()} -> {df['time_et'].max()}")

    print("\nSummary statistics (5m bars):")
    print(basic_stats(df).round(2))

    latest_close = float(df["Close"].iloc[-1])
    daily_volume = df["Volume"].groupby(df["time_et"].dt.date).sum()

    print(f"\nLatest close: {latest_close:.2f}")
    print("Volume by day (shares):")
    print(daily_volume)


if __name__ == "__main__":
    main()
