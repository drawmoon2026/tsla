import pandas as pd
import pytz
import yfinance as yf


def fetch_intraday(ticker: str = "TSLA", period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    Download intraday data for the given ticker.
    5m bars are limited by Yahoo to recent history; default 5d keeps it small and reliable.
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",  # keep flat columns even if one ticker
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Try a shorter period or check ticker.")

    # yfinance provides a timezone-aware index; convert to US/Eastern for readability.
    eastern = pytz.timezone("US/Eastern")
    if df.index.tz is None:
        df.index = df.index.tz_localize(pytz.utc).tz_convert(eastern)
    else:
        df.index = df.index.tz_convert(eastern)

    # If columns are multi-index (e.g., when multiple tickers are passed),
    # keep only the requested ticker and drop the ticker level.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1, drop_level=True)

    return df


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for key numeric columns.
    """
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[numeric_cols].describe().T


def main() -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)

    print("Downloading TSLA 5-minute data (last 5 days)...")
    df = fetch_intraday()
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    print(f"Time range (US/Eastern): {df.index.min()} -> {df.index.max()}")

    stats = basic_stats(df)
    print("\nSummary statistics (5m bars):")
    print(stats.round(2))

    # Simple additional aggregates
    latest_close = float(df["Close"].iloc[-1])
    daily_volume = df["Volume"].groupby(df.index.date).sum()

    print(f"\nLatest close: {latest_close:.2f}")
    print("Volume by day (shares):")
    print(daily_volume)


if __name__ == "__main__":
    main()
