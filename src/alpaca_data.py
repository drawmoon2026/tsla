"""Fetch multi-year historical minute bars from Alpaca Market Data v2.

Purpose: break the yfinance 60-day ceiling (Phase 2 of docs/roadmap.md).
Free accounts work: SIP historical data older than 15 minutes is available;
falls back to the IEX feed automatically if SIP is rejected.

Credentials (either naming scheme, env vars or a project-root .env file):
    ALPACA_KEY_ID / ALPACA_SECRET_KEY        (this project's config.py names)
    APCA_API_KEY_ID / APCA_API_SECRET_KEY    (Alpaca SDK names)

Usage:
    .venv/bin/python src/alpaca_data.py --years 2
    -> data/TSLA_5m_alpaca.csv  (RTH only, UTC timestamps, bar-start,
       split-adjusted; drop-in compatible with src.common.data_io.load_bars)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET

DATA_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
PAGE_LIMIT = 10_000


def load_keys() -> tuple[str, str]:
    env = dict(os.environ)
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    key = env.get("ALPACA_KEY_ID") or env.get("APCA_API_KEY_ID")
    secret = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca API keys not found.\n"
            "Set them in your shell:\n"
            "    export ALPACA_KEY_ID=...\n"
            "    export ALPACA_SECRET_KEY=...\n"
            "or put those two lines (KEY=value) in a project-root .env file\n"
            "(.env is gitignored — never commit keys)."
        )
    return key, secret


def fetch_bars(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp,
    timeframe: str, key: str, secret: str, feed: str = "sip",
) -> pd.DataFrame:
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": PAGE_LIMIT,
        "adjustment": "split",
        "feed": feed,
        "sort": "asc",
    }
    rows: list[dict] = []
    page = 0
    while True:
        r = requests.get(DATA_URL.format(symbol=symbol), headers=headers, params=params, timeout=30)
        if r.status_code == 403 and feed == "sip":
            print("SIP feed rejected for this account — falling back to IEX feed.")
            return fetch_bars(symbol, start, end, timeframe, key, secret, feed="iex")
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("bars") or [])
        page += 1
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        if page % 10 == 0:
            print(f"  ... {len(rows):,} bars fetched")

    if not rows:
        raise SystemExit("Alpaca returned no bars — check symbol/date range/feed.")
    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "Datetime", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    print(f"Fetched {len(df):,} raw bars via feed={feed}")
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars only (ET 9:30 <= t < 16:00) so downstream
    day-anchored resampling keeps its 9:30 anchor."""
    et = df.index.tz_convert(ET)
    mask = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    return df[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch historical minute bars from Alpaca.")
    ap.add_argument("--symbol", default="TSLA")
    ap.add_argument("--years", type=float, default=2.0, help="How far back to fetch (default 2).")
    ap.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "1h"],
                    help="Bar size (default 5m).")
    ap.add_argument("--out", default=None, help="Output CSV (default data/{SYMBOL}_{interval}_alpaca.csv)")
    args = ap.parse_args()

    timeframe = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}[args.interval]
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)  # free tier: data older than 15 min
    start = end - pd.Timedelta(days=int(args.years * 365.25))
    out = Path(args.out) if args.out else Path("data") / f"{args.symbol.upper()}_{args.interval}_alpaca.csv"

    key, secret = load_keys()
    print(f"Fetching {args.symbol} {timeframe} bars {start.date()} -> {end.date()} ...")
    df = fetch_bars(args.symbol.upper(), start, end, timeframe, key, secret)
    df = filter_rth(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index_label="Datetime", date_format="%Y-%m-%dT%H:%M:%S%z")
    et = df.index.tz_convert(ET)
    n_days = len(pd.unique(et.date))
    print(f"Saved {len(df):,} RTH bars across {n_days} trading days -> {out}")
    print(f"Range (ET): {et.min()} -> {et.max()}")
    print("Use it anywhere the yfinance CSV is used, e.g.:")
    print(f"  .venv/bin/python src/hourly_signal_backtest.py --walkforward --input_csv {out}")


if __name__ == "__main__":
    main()
