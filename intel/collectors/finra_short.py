"""FINRA 空头利益采集器：TSLA 双周合并空头利益（T1，布局痕迹层）.

通道：FINRA Query API 公开数据集 consolidatedShortInterest（免 key，POST JSON）。
    https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
实测（2026-07）：symbolCode=TSLA 一次拿全 2017-12-29 → 最新，约 205 行。

时间口径（诚实声明）：数据集只有 settlementDate（结算日），没有官方发布时刻字段。
FINRA 的公开发布日程约为结算日后 9 个交易日（盘后）。event_time_utc 取
settlementDate + 9 个交易日的 16:00 ET 作为**近似发布时刻**，payload 里
approx_publication=true 记账。observed_time_utc（哨兵首见）永远是准的；
首采历史行的 lag 是回填口径。

拆股折算（正式口径，intel/splits.py，2026-07-24 起）：short_interest 为未复权
股数，跨拆股行的 change_pct 在入库/落盘前按拆股表折算（原值留 change_pct_raw、
split_adjusted 记账）；折算后仍 >= SPLIT_GUARD_PCT 的跳变发 si_split_alert
告警事件（疑似未登记拆股，需人工把拆股表补进 intel/splits.py），不静默入库。

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
from intel.splits import SPLIT_GUARD_PCT, adjust_change_pct, unexplained_jumps

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


def adjusted_payloads(rows: list[dict], symbol: str) -> tuple[list[dict], list, list]:
    """原始行 → 按结算日升序的 payload 列表，入库前拆股折算 + 跳变侦测.

    返回 (payloads, 折算明细, 未解释跳变告警)。正式口径见 intel/splits.py。
    """
    payloads = [_payload(r) for r in sorted(rows, key=lambda r: r["settlementDate"])]
    fixes = adjust_change_pct(payloads, symbol)
    alerts = unexplained_jumps(payloads, symbol)
    return payloads, fixes, alerts


class FinraShortCollector(Collector):
    SOURCE = {
        "source_id": "finra_short",
        "name": "FINRA consolidated short interest TSLA (biweekly)",
        "tier": "T0",
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
        payloads, _fixes, alerts = adjusted_payloads(raw, "TSLA")
        events = []
        for p in payloads:
            sd = p["settlement_date"]
            events.append(
                {
                    "dedupe_key": f"si_{sd}",
                    "event_time_utc": approx_pub_time_utc(sd),
                    "symbol": "TSLA",
                    "type": "short_interest",
                    "title": (f"TSLA short interest {p['short_interest']:,} "
                              f"({p['change_pct']:+.1f}%) @ {sd}"
                              + ("（拆股折算）" if p.get("split_adjusted") else "")),
                    "url": "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files",
                    "payload": p,
                }
            )
        # 跳变侦测告警：拆股表无法解释的 >= SPLIT_GUARD_PCT 跳变，不静默入库
        for a in alerts:
            events.append(
                {
                    "dedupe_key": f"si_split_alert_{a['settlement_date']}",
                    "event_time_utc": approx_pub_time_utc(a["settlement_date"]),
                    "symbol": "TSLA",
                    "type": "si_split_alert",
                    "title": (f"空头跳变告警：TSLA change_pct {a['change_pct']:+.1f}% >= "
                              f"+{SPLIT_GUARD_PCT:.0f}%（结算 {a['settlement_date']}）"
                              "且拆股表无法解释——疑似未登记拆股，需人工核对 "
                              "intel/splits.py（真实跳变则由 detector SPLIT_GUARD 拦截）"),
                    "payload": a,
                }
            )
        return events


def backfill() -> None:
    """全历史 → data/intel/finra_short.csv（N1 schema，拆股折算后落盘）."""
    import pandas as pd

    payloads, fixes, alerts = adjusted_payloads(_fetch_rows(), "TSLA")
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "finra_short.csv", index=False)
    print(f"finra_short.csv: {len(df)} rows "
          f"[{df['event_time_utc'].min()} -> {df['event_time_utc'].max()}]")
    for f in fixes:
        print(f"  拆股折算 {f['settlement_date']}: change_pct "
              f"{f['change_pct_raw']:+.2f}% -> {f['change_pct_corrected']:+.2f}% "
              f"(factor {f['factor_prev']:.0f}->{f['factor_cur']:.0f})")
    for a in alerts:
        print(f"  [SPLIT-ALERT] {a['settlement_date']}: change_pct "
              f"{a['change_pct']:+.2f}% >= +{SPLIT_GUARD_PCT:.0f}% 拆股表无法解释")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    if args.backfill:
        backfill()
    else:
        FinraShortCollector().run_once()
