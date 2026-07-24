"""FINRA 空头利益批量采集（N5 跨标的验证用）.

复用 intel.collectors.finra_short 的 API 口径与发布时刻近似（结算 + 9 交易日 16:00 ET，
与 N2/N4 相同），对 E8 池 25 只标的逐只拉全历史，落
data/intel/pool_short/{SYMBOL}.csv（N1 schema，与 data/intel/finra_short.csv 同构）。
TSLA 不在此拉取——复用既有 data/intel/finra_short.csv。

用法： .venv/bin/python -m intel.collectors.finra_short_pool
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from intel.collectors.base import UA
from intel.collectors.finra_short import API, _payload, approx_pub_time_utc

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "intel" / "pool_short"

POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "CRM", "COST", "JPM", "V", "UNH", "XOM", "WMT", "DIS", "BA", "CAT", "GS",
    "INTC", "MU", "QCOM", "PLTR", "COIN",
]


def fetch_symbol(symbol: str, limit: int = 1000) -> list[dict]:
    body = {
        "limit": limit,
        "compareFilters": [
            {"fieldName": "symbolCode", "fieldValue": symbol, "compareType": "EQUAL"}
        ],
    }
    for attempt in range(5):
        r = requests.post(
            API,
            headers={"User-Agent": UA, "Accept": "application/json",
                     "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{symbol}: 429 after retries")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in POOL:
        out = OUT_DIR / f"{sym}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip {sym} (exists)")
            continue
        rows = fetch_symbol(sym)
        recs = [
            {
                "event_time_utc": approx_pub_time_utc(r["settlementDate"]),
                "source": "finra_short",
                "type": "short_interest",
                "payload": json.dumps(_payload(r), ensure_ascii=False),
            }
            for r in rows
        ]
        df = pd.DataFrame(recs).sort_values("event_time_utc")
        df.to_csv(out, index=False)
        rng = (f"[{df['event_time_utc'].min()} -> {df['event_time_utc'].max()}]"
               if len(df) else "[empty]")
        print(f"{sym}: {len(df)} rows {rng}")
        time.sleep(2)


if __name__ == "__main__":
    main()
