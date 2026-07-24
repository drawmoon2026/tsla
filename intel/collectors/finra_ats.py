"""FINRA ATS（暗池）+ 非 ATS 场外周报采集器：TSLA 场外成交占比（T1，布局痕迹层）.

通道：FINRA Query API 公开数据集 weeklySummary（OTC Transparency，免 key）。
    https://api.finra.org/data/group/otcMarket/name/weeklySummary
每周每标的的行结构：per-firm 行（ATS_W_SMBL_FIRM / OTC_W_SMBL_FIRM）+ 两条聚合行
（ATS_W_SMBL = 全部暗池合计，OTC_W_SMBL = 非 ATS 场外做市合计）。本采集器只取聚合行。

时间口径：行内自带 initialPublishedDate（FINRA 首次发布日，Tier1 约滞后 2-3 周），
event_time_utc 取该日 12:00 ET（精确到日，payload date_precision 记账）。

场外占比：off_exchange_pct = (ATS + OTC 做市) 周成交股数 / 全市场周成交量。
分母用 yfinance 日线周合计（best effort；拉不到则为 null，分子照存）。

用法： .venv/bin/python -m intel.collectors.finra_ats --once
       .venv/bin/python -m intel.collectors.finra_ats --backfill  # 全历史 → data/intel/finra_ats.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from intel.collectors.base import UA, Collector

API = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "intel"
AGG_CODES = {"ATS_W_SMBL", "OTC_W_SMBL"}
RECENT_LIMIT = 60  # 前向模式只看最近 60 条聚合行（≈30 周，够覆盖发布滞后）


def _post(body: dict) -> list[dict]:
    r = requests.post(
        API,
        headers={"User-Agent": UA, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_agg_rows(all_history: bool = False) -> list[dict]:
    """TSLA 的聚合行（ATS_W_SMBL / OTC_W_SMBL）；all_history 时 offset 翻页拿全."""
    rows, offset = [], 0
    while True:
        batch = _post(
            {
                "limit": 1000,
                "offset": offset,
                "compareFilters": [
                    {"fieldName": "issueSymbolIdentifier", "fieldValue": "TSLA",
                     "compareType": "EQUAL"}
                ],
                "domainFilters": [
                    {"fieldName": "summaryTypeCode", "values": sorted(AGG_CODES)}
                ],
            }
        )
        rows.extend(batch)
        if len(batch) < 1000 or not all_history:
            break
        offset += 1000
    rows = [r for r in rows if r.get("summaryTypeCode") in AGG_CODES]
    rows.sort(key=lambda r: (r["weekStartDate"], r["summaryTypeCode"]))
    return rows if all_history else rows[-RECENT_LIMIT:]


def _pub_time_utc(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=12, minute=0, tzinfo=ZoneInfo("America/New_York")
    )
    return d.astimezone(ZoneInfo("UTC")).isoformat(timespec="seconds")


def weekly_market_volume(week_starts: list[str]) -> dict[str, float]:
    """yfinance 日线 → {weekStartDate(周一): 全市场周成交量}；失败返回空 dict."""
    try:
        import pandas as pd
        import yfinance as yf

        lo = min(week_starts)
        df = yf.download("TSLA", start=lo, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return {}
        vol = df["Volume"]
        if hasattr(vol, "columns"):  # 多层列
            vol = vol.iloc[:, 0]
        wk = vol.groupby(vol.index.to_period("W").start_time).sum()
        return {k.strftime("%Y-%m-%d"): float(v) for k, v in wk.items()}
    except Exception:  # noqa: BLE001  best effort，占比可为空
        return {}


def build_week_events(rows: list[dict]) -> list[dict]:
    """聚合行 → 每周一条事件（ATS+OTC 合并，含 off_exchange_pct）."""
    by_week: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_week[r["weekStartDate"]][r["summaryTypeCode"]] = r
    mkt_vol = weekly_market_volume(sorted(by_week)) if by_week else {}

    events = []
    for week, d in sorted(by_week.items()):
        ats, otc = d.get("ATS_W_SMBL"), d.get("OTC_W_SMBL")
        pub_dates = [x["initialPublishedDate"] for x in d.values() if x.get("initialPublishedDate")]
        if not pub_dates:
            continue
        ats_sh = float(ats["totalWeeklyShareQuantity"]) if ats else None
        otc_sh = float(otc["totalWeeklyShareQuantity"]) if otc else None
        off_sh = (ats_sh or 0.0) + (otc_sh or 0.0)
        mv = mkt_vol.get(week)
        pct = round(off_sh / mv * 100, 2) if mv else None
        events.append(
            {
                "dedupe_key": f"ats_{week}",
                "event_time_utc": _pub_time_utc(max(pub_dates)),
                "symbol": "TSLA",
                "type": "ats_weekly",
                "title": (f"TSLA off-exchange week {week}: ATS {ats_sh or 0:,.0f} + "
                          f"OTC {otc_sh or 0:,.0f} sh"
                          + (f" ({pct}% of consolidated)" if pct is not None else "")),
                "url": "https://otctransparency.finra.org/",
                "payload": {
                    "week_start": week,
                    "ats_shares": ats_sh,
                    "ats_trades": ats.get("totalWeeklyTradeCount") if ats else None,
                    "otc_shares": otc_sh,
                    "otc_trades": otc.get("totalWeeklyTradeCount") if otc else None,
                    "off_exchange_shares": off_sh,
                    "market_week_volume": mv,
                    "off_exchange_pct": pct,
                    "published": max(pub_dates),
                    "date_precision": "day",
                },
            }
        )
    return events


class FinraAtsCollector(Collector):
    SOURCE = {
        "source_id": "finra_ats",
        "name": "FINRA OTC transparency TSLA ATS/OTC weekly",
        "tier": "T0",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.6,
        "notes": "周口径，Tier1 发布滞后 2-3 周；event_time=initialPublishedDate 12:00ET"
                 "（精确到日）；占比分母 yfinance best-effort",
    }

    def fetch(self):
        return fetch_agg_rows(all_history=False)

    def normalize(self, raw) -> list[dict]:
        return build_week_events(raw)


def backfill() -> None:
    """全历史聚合行 → data/intel/finra_ats.csv（N1 schema）."""
    import pandas as pd

    rows = fetch_agg_rows(all_history=True)
    events = build_week_events(rows)
    recs = [
        {
            "event_time_utc": ev["event_time_utc"],
            "source": "finra_ats",
            "type": ev["type"],
            "payload": json.dumps(ev["payload"], ensure_ascii=False),
        }
        for ev in events
    ]
    df = pd.DataFrame(recs).sort_values("event_time_utc")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "finra_ats.csv", index=False)
    n_pct = sum(1 for ev in events if ev["payload"]["off_exchange_pct"] is not None)
    print(f"finra_ats.csv: {len(df)} weeks "
          f"[{df['event_time_utc'].min()} -> {df['event_time_utc'].max()}], "
          f"off_exchange_pct 可算 {n_pct}/{len(df)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    if args.backfill:
        backfill()
    else:
        FinraAtsCollector().run_once()
