"""FINRA 空头利益批量采集（N5 跨标的验证 / E12 特征用）.

复用 intel.collectors.finra_short 的 API 口径与发布时刻近似（结算 + 9 交易日 16:00 ET，
与 N2/N4 相同），对 E8 池 25 只标的逐只拉全历史，落
data/intel/pool_short/{SYMBOL}.csv（N1 schema，与 data/intel/finra_short.csv 同构）。
TSLA 不在此拉取——复用既有 data/intel/finra_short.csv。

正式口径（2026-07-24 起，出处 N5/N6）：
- 拆股折算：落盘前按 intel/splits.py 拆股表重算跨拆股行 change_pct
  （原值留 change_pct_raw、split_adjusted 记账）；折算后仍 >= SPLIT_GUARD_PCT
  的跳变打印 [SPLIT-ALERT]（疑似未登记拆股，需人工核对拆股表），不静默落盘。
- META symbolCode 清洗：FINRA symbolCode 有复用污染——META 代码在 2021-07→2022-01
  被 Roundhill Ball Metaverse ETF 占用、FB 代码 2025-06 起被 ProShares ETF 复用。
  META 系列由 FB(Facebook/Meta Platforms 更名期) + META(Meta Platforms) 两个
  symbolCode 按 issueName 白名单过滤后拼接，剔除 ETF 行。

用法： .venv/bin/python -m intel.collectors.finra_short_pool          # 缺的补拉
       .venv/bin/python -m intel.collectors.finra_short_pool --force  # 全量重拉
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from intel.collectors.base import UA
from intel.collectors.finra_short import (
    API,
    adjusted_payloads,
    approx_pub_time_utc,
)
from intel.splits import SPLIT_GUARD_PCT

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "intel" / "pool_short"

POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "CRM", "COST", "JPM", "V", "UNH", "XOM", "WMT", "DIS", "BA", "CAT", "GS",
    "INTC", "MU", "QCOM", "PLTR", "COIN",
]

# META symbolCode 清洗（N5 副产物）：issueName 前缀白名单，命中才保留
_META_ISSUE_PREFIXES = ("facebook", "meta platforms")


def _query(symbol_code: str, limit: int = 1000) -> list[dict]:
    body = {
        "limit": limit,
        "compareFilters": [
            {"fieldName": "symbolCode", "fieldValue": symbol_code,
             "compareType": "EQUAL"}
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
    raise RuntimeError(f"{symbol_code}: 429 after retries")


def fetch_symbol(symbol: str, limit: int = 1000) -> list[dict]:
    """单标的原始行；META 走 FB+META 双 symbolCode 拼接 + issueName 清洗."""
    if symbol != "META":
        return _query(symbol, limit)
    rows = _query("FB", limit) + _query("META", limit)
    kept, dropped = [], set()
    seen: set[str] = set()
    for r in sorted(rows, key=lambda r: r["settlementDate"]):
        name = (r.get("issueName") or "").strip().lower()
        if not name.startswith(_META_ISSUE_PREFIXES):
            dropped.add(r.get("issueName"))
            continue
        if r["settlementDate"] in seen:  # FB/META 换代边界防重
            continue
        seen.add(r["settlementDate"])
        kept.append(r)
    if dropped:
        print(f"  META 清洗：剔除 symbolCode 复用污染行 issueName={sorted(dropped)}")
    return kept


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in POOL:
        out = OUT_DIR / f"{sym}.csv"
        if not force and out.exists() and out.stat().st_size > 0:
            print(f"skip {sym} (exists)")
            continue
        payloads, fixes, alerts = adjusted_payloads(fetch_symbol(sym), sym)
        recs = [
            {
                "event_time_utc": approx_pub_time_utc(p["settlement_date"]),
                "source": "finra_short",
                "type": "short_interest",
                "payload": json.dumps(p, ensure_ascii=False),
            }
            for p in payloads
        ]
        df = pd.DataFrame(recs).sort_values("event_time_utc")
        df.to_csv(out, index=False)
        rng = (f"[{df['event_time_utc'].min()} -> {df['event_time_utc'].max()}]"
               if len(df) else "[empty]")
        print(f"{sym}: {len(df)} rows {rng}")
        for f in fixes:
            print(f"  拆股折算 {f['settlement_date']}: change_pct "
                  f"{f['change_pct_raw']:+.2f}% -> {f['change_pct_corrected']:+.2f}% "
                  f"(factor {f['factor_prev']:.0f}->{f['factor_cur']:.0f})")
        for a in alerts:
            print(f"  [SPLIT-ALERT] {sym} {a['settlement_date']}: change_pct "
                  f"{a['change_pct']:+.2f}% >= +{SPLIT_GUARD_PCT:.0f}% 拆股表无法解释"
                  "——疑似未登记拆股或极端真实跳变，人工核对 intel/splits.py")
        time.sleep(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="已存在的 CSV 也重拉覆盖")
    args = p.parse_args()
    main(force=args.force)
