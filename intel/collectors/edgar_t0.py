"""EDGAR T0 采集器：TSLA 的 SC 13D/13G（含 /A）与 Form 144（布局痕迹层，T1）.

通道：与 edgar.py 相同的 data.sec.gov submissions API（免 key，acceptanceDateTime 口径）。
13D/G 与 Form 144 都以 TSLA 为 subject company，出现在 TSLA 的 submissions 列表里
（实测 2026-07：recent 段含 SC 13G ×6、SC 13G/A ×27、SCHEDULE 13G(/A) ×5、144 ×75）。
EDGAR 2024 起把新申报表名从 "SC 13D/G" 改为 "SCHEDULE 13D/G"，两套命名都收。

event_time_utc = acceptanceDateTime（公众理论最早可见时刻）；限速合规由 base.http_get
统一处理（全局 ~6.7 req/s < SEC 红线 10）。

已知边界（诚实声明）：
- Form 144 电子申报 2023-04 起才强制；此前绝大多数是纸质件，不进 EDGAR 全文库——
  历史回填的 144 覆盖实际从 2023-04 开始，更早的空白是制度性缺口，不是采集缺陷。
- 13D（积极举牌）在 TSLA 上极罕见：机构对 TSLA 的持仓披露几乎全走 13G（被动）。

用法：
    .venv/bin/python -m intel.collectors.edgar_t0 --once       # 哨兵前向一轮
    .venv/bin/python -m intel.collectors.edgar_t0 --backfill   # 2018 起历史 → data/intel/*.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intel.collectors.base import Collector, http_get

CIK = "0001318605"
URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
FORMS_13DG = {
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
    "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A",
}
FORMS_144 = {"144", "144/A"}
FORMS = FORMS_13DG | FORMS_144
FIRST_FILL_DAYS = 90
BACKFILL_SINCE = "2018-01-01"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "intel"


def _acc_to_utc(acceptance: str) -> str:
    dt = datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _etype(form: str) -> str:
    base = form.replace("SCHEDULE ", "SC ")  # 新旧命名归一
    return {
        "SC 13D": "sc13d", "SC 13D/A": "sc13d_amend",
        "SC 13G": "sc13g", "SC 13G/A": "sc13g_amend",
        "144": "form144", "144/A": "form144_amend",
    }[base]


def _rows_from_recent(rec: dict, since: str) -> list[dict]:
    """submissions 段（recent 或归档分片，同构 dict of lists）→ 事件 dict 列表."""
    n = len(rec["form"])
    items_col = rec.get("items", [""] * n)
    desc_col = rec.get("primaryDocDescription", [""] * n)
    out = []
    for i in range(n):
        form = rec["form"][i]
        if form not in FORMS or rec["filingDate"][i] < since:
            continue
        acc = rec["accessionNumber"][i]
        doc = rec["primaryDocument"][i]
        out.append(
            {
                "dedupe_key": acc,
                "event_time_utc": _acc_to_utc(rec["acceptanceDateTime"][i]),
                "symbol": "TSLA",
                "type": _etype(form),
                "title": f"{form} {desc_col[i] or acc}",
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc.replace('-', '')}/{doc}",
                "payload": {
                    "form": form,
                    "accession": acc,
                    "filing_date": rec["filingDate"][i],
                    "report_date": rec["reportDate"][i],
                    "size": rec["size"][i],
                    "_items": items_col[i],
                },
            }
        )
    return out


class EdgarT0Collector(Collector):
    SOURCE = {
        "source_id": "edgar_t0",
        "name": "SEC EDGAR TSLA SC13D/G + Form144 (submissions API)",
        "tier": "T1",
        "method": "api",
        "poll_interval_s": 300,
        "cost": "free",
        "weight_source": 0.9,
        "notes": "布局痕迹层：举牌/被动大持仓披露 + 内部人拟售登记；"
                 "144 电子申报 2023-04 起强制，更早为纸质缺口",
    }

    def fetch(self):
        return http_get(URL).json()

    def normalize(self, raw) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=FIRST_FILL_DAYS)).strftime(
            "%Y-%m-%d"
        )
        events = _rows_from_recent(raw["filings"]["recent"], since=cutoff)
        for ev in events:
            ev["payload"].pop("_items", None)
        return events


# ---------------------------------------------------------------- backfill (Part B)


def backfill() -> None:
    """2018 起 13D/G 与 144 全历史 → data/intel/edgar_13dg.csv / edgar_144.csv.

    输出与 N1 历史管线同 schema：event_time_utc, source, type, payload。
    """
    import pandas as pd

    main = http_get(URL).json()
    segments = [main["filings"]["recent"]]
    for extra in main["filings"].get("files", []):
        if extra["filingTo"] >= BACKFILL_SINCE:
            segments.append(http_get(f"https://data.sec.gov/submissions/{extra['name']}").json())
    rows = []
    for seg in segments:
        rows.extend(_rows_from_recent(seg, since=BACKFILL_SINCE))

    recs = []
    for ev in rows:
        p = ev["payload"]
        recs.append(
            {
                "event_time_utc": ev["event_time_utc"],
                "source": "edgar_13dg" if not p["form"].startswith("144") else "edgar_144",
                "type": ev["type"],
                "payload": json.dumps(
                    {k: v for k, v in p.items() if k != "_items"}, ensure_ascii=False
                ),
            }
        )
    df = pd.DataFrame(recs).sort_values("event_time_utc")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src, fname in [("edgar_13dg", "edgar_13dg.csv"), ("edgar_144", "edgar_144.csv")]:
        sub = df[df["source"] == src]
        sub.to_csv(DATA_DIR / fname, index=False)
        span = (sub["event_time_utc"].min(), sub["event_time_utc"].max()) if len(sub) else ("-", "-")
        print(f"{fname}: {len(sub)} rows  [{span[0]} -> {span[1]}]")
        if src == "edgar_144" and len(sub):
            print("  注意：144 电子申报 2023-04 起才强制，更早的纸质件不在 EDGAR，属制度性缺口")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="哨兵前向跑一轮")
    p.add_argument("--backfill", action="store_true", help="2018 起历史 → data/intel/ CSV")
    args = p.parse_args()
    if args.backfill:
        backfill()
    else:
        EdgarT0Collector().run_once()
