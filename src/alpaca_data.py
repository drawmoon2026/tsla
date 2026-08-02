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
            # P0-B: fallback is no longer silent — the recursive call stamps the
            # actually-used feed into df.attrs["feed"], which callers persist.
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
    df.attrs["feed"] = feed  # P0-B: record the feed actually used (sip may fall back to iex)
    print(f"Fetched {len(df):,} raw bars via feed={feed}")
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars only (ET 9:30 <= t < 16:00) so downstream
    day-anchored resampling keeps its 9:30 anchor."""
    et = df.index.tz_convert(ET)
    mask = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    out = df[mask]
    out.attrs = dict(df.attrs)  # keep feed provenance through filtering
    return out


def write_sidecar_meta(csv_path: Path, df: pd.DataFrame, *, symbol: str,
                       timeframe: str, requested_feed: str = "sip") -> Path:
    """P0-B 数据口径披露：CSV 旁车 meta json（不动 CSV 本体，load_bars 兼容）.

    记录实际使用的 feed（sip 被 403 拒后自动回落 iex——回落不再只留 stdout）。
    """
    import json
    from datetime import datetime, timezone

    meta = {
        "source": "alpaca",
        "feed": df.attrs.get("feed"),
        "feed_requested": requested_feed,
        "feed_note": "sip=全市场合并带（历史，免费层 >15min 延迟可用）；"
                     "iex=IEX 单一交易所（sip 被 403 拒时自动回落）",
        "symbol": symbol,
        "timeframe": timeframe,
        "adjustment": "split",
        "rth_only": True,
        "n_bars": int(len(df)),
        "range_utc": [df.index.min().isoformat(), df.index.max().isoformat()] if len(df) else None,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = csv_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch historical minute bars from Alpaca.")
    ap.add_argument("--symbol", default="TSLA")
    ap.add_argument("--years", type=float, default=2.0, help="How far back to fetch (default 2).")
    ap.add_argument("--start", default=None,
                    help="Explicit range start (YYYY-MM-DD, UTC). Overrides --years.")
    ap.add_argument("--end", default=None,
                    help="Explicit range end (YYYY-MM-DD, UTC). Requires/implies --start; "
                         "defaults to now-16min when only --start is given.")
    ap.add_argument("--interval", default="5m", choices=["1m", "5m", "15m", "1h"],
                    help="Bar size (default 5m).")
    ap.add_argument("--out", default=None, help="Output CSV (default data/{SYMBOL}_{interval}_alpaca.csv)")
    args = ap.parse_args()

    timeframe = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}[args.interval]
    now_cap = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)  # free tier: data older than 15 min
    if args.end and not args.start:
        raise SystemExit("--end requires --start")
    if args.start:
        start = pd.Timestamp(args.start, tz="UTC")
        end = min(pd.Timestamp(args.end, tz="UTC"), now_cap) if args.end else now_cap
    else:
        end = now_cap
        start = end - pd.Timedelta(days=int(args.years * 365.25))
    out = Path(args.out) if args.out else Path("data") / f"{args.symbol.upper()}_{args.interval}_alpaca.csv"

    key, secret = load_keys()
    print(f"Fetching {args.symbol} {timeframe} bars {start.date()} -> {end.date()} ...")
    df = fetch_bars(args.symbol.upper(), start, end, timeframe, key, secret)
    df = filter_rth(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index_label="Datetime", date_format="%Y-%m-%dT%H:%M:%S%z")
    meta_path = write_sidecar_meta(out, df, symbol=args.symbol.upper(), timeframe=timeframe)
    et = df.index.tz_convert(ET)
    n_days = len(pd.unique(et.date))
    print(f"Saved {len(df):,} RTH bars across {n_days} trading days -> {out}")
    print(f"Feed provenance -> {meta_path} (feed={df.attrs.get('feed')})")
    print(f"Range (ET): {et.min()} -> {et.max()}")
    print("Use it anywhere the yfinance CSV is used, e.g.:")
    print(f"  .venv/bin/python src/hourly_signal_backtest.py --walkforward --input_csv {out}")


if __name__ == "__main__":
    main()
