"""FINRA 空头利益采集器：TSLA 双周合并空头利益（T1，布局痕迹层）.

通道：FINRA Query API 公开数据集 consolidatedShortInterest（免 key，POST JSON）。
    https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
实测（2026-07）：symbolCode=TSLA 一次拿全 2017-12-29 → 最新，约 205 行。

时间口径（诚实声明）：数据集只有 settlementDate（结算日），没有官方发布时刻字段。
FINRA 的公开发布日程约为结算日后 9 个交易日（盘后）。event_time_utc 取
settlementDate + 9 个交易日的 16:00 ET 作为**近似发布时刻**，payload 里
approx_publication=true 记账。observed_time_utc（哨兵首见）永远是准的；
首采历史行的 lag 是回填口径。

用法： .venv/bin/python -m intel.collectors.finra_short --once
       .venv/bin/python -m intel.collectors.finra_short --backfill  # → data/intel/finra_short.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

from intel.collectors.base import UA, Collector

API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "intel"
PUBLICATION_LAG_BDAYS = 9  # FINRA 公布日程近似：结算日 + 9 个交易日


def _fetch_rows(limit: int = 1000) -> list[dict]:
    body = {
        "limit": limit,
        "compareFilters": [
            {"fieldName": "symbolCode", "fieldValue": "TSLA", "compareType": "EQUAL"}
        ],
    }
    r = requests.post(
        API,
        headers={"User-Agent": UA, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def approx_pub_time_utc(settlement_date: str) -> str:
    """结算日 + 9 交易日 16:00 ET → ISO8601 UTC（近似发布时刻，不含交易所假日修正）."""
    pub = np.busday_offset(settlement_date, PUBLICATION_LAG_BDAYS, roll="forward")
    d = datetime.strptime(str(pub), "%Y-%m-%d").replace(
        hour=16, minute=0, tzinfo=ZoneInfo("America/New_York")
    )
    return d.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")


def _payload(row: dict) -> dict:
    return {
        "settlement_date": row["settlementDate"],
        "short_interest": row["currentShortPositionQuantity"],
        "prev_short_interest": row["previousShortPositionQuantity"],
        "change": row["changePreviousNumber"],
        "change_pct": row["changePercent"],
        "avg_daily_volume": row["averageDailyVolumeQuantity"],
        "days_to_cover": row["daysToCoverQuantity"],
        "revision": row.get("revisionFlag"),
        "approx_publication": True,
        "publication_rule": f"settlement + {PUBLICATION_LAG_BDAYS} bdays 16:00 ET",
    }


class FinraShortCollector(Collector):
    SOURCE = {
        "source_id": "finra_short",
        "name": "FINRA consolidated short interest TSLA (biweekly)",
        "tier": "T1",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.6,
        "notes": "双周口径天然钝；event_time 为近似发布时刻（结算+9交易日16:00ET），"
                 "approx_publication 记账",
    }

    def fetch(self):
        return _fetch_rows()

    def normalize(self, raw) -> list[dict]:
        events = []
        for row in raw:
            sd = row["settlementDate"]
            events.append(
                {
                    "dedupe_key": f"si_{sd}",
                    "event_time_utc": approx_pub_time_utc(sd),
                    "symbol": "TSLA",
                    "type": "short_interest",
                    "title": (f"TSLA short interest {row['currentShortPositionQuantity']:,} "
                              f"({row['changePercent']:+.1f}%) @ {sd}"),
                    "url": "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files",
                    "payload": _payload(row),
                }
            )
        return events


def backfill() -> None:
    """全历史 → data/intel/finra_short.csv（N1 schema）."""
    import pandas as pd

    rows = _fetch_rows()
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "finra_short.csv", index=False)
    print(f"finra_short.csv: {len(df)} rows "
          f"[{df['event_time_utc'].min()} -> {df['event_time_utc'].max()}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    if args.backfill:
        backfill()
    else:
        FinraShortCollector().run_once()
