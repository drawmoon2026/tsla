"""EDGAR 采集器：TSLA 的 Form 4 / 8-K 实时申报流（T1）.

通道：data.sec.gov submissions API（官方 JSON，含 acceptanceDateTime，免 key）。
    https://data.sec.gov/submissions/CIK0001318605.json
recent 段覆盖最近约 1000 张申报，增量轮询绰绰有余（TSLA 年均申报数百张）。
event_time_utc = acceptanceDateTime（SEC 接收申报时刻 = 公众理论最早可见时刻）；
交易发生日只进 payload。首轮回填最近 FIRST_FILL_DAYS 天，之后靠 accession 去重增量。

注意：SEC 要求 User-Agent 带联系方式、≤10 req/s——base.http_get 已统一处理。
历史全量（2018 起）另见旧模块 intel/edgar.py（CSV 管线，与本采集器互不干扰）。

用法： .venv/bin/python -m intel.collectors.edgar --once
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intel.collectors.base import Collector, cli, http_get

CIK = "0001318605"
URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
FORMS = {"4", "4/A", "8-K", "8-K/A"}
FIRST_FILL_DAYS = 90


def _acc_to_utc(acceptance: str) -> str:
    # "2026-07-23T20:15:12.000Z" -> ISO8601 +00:00
    dt = datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class EdgarCollector(Collector):
    SOURCE = {
        "source_id": "edgar",
        "name": "SEC EDGAR TSLA Form4/8-K (submissions API)",
        "tier": "T1",
        "method": "api",
        "poll_interval_s": 300,
        "cost": "free",
        "weight_source": 1.0,
        "notes": "acceptanceDateTime 为公众最早可见时刻；EDGAR 接收窗口工作日 6:00-22:00 ET",
    }

    def fetch(self):
        return http_get(URL).json()

    def normalize(self, raw) -> list[dict]:
        rec = raw["filings"]["recent"]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=FIRST_FILL_DAYS)).strftime(
            "%Y-%m-%d"
        )
        events = []
        for i in range(len(rec["form"])):
            form = rec["form"][i]
            if form not in FORMS or rec["filingDate"][i] < cutoff:
                continue
            acc = rec["accessionNumber"][i]
            acc_nodash = acc.replace("-", "")
            doc = rec["primaryDocument"][i]
            items = [s.strip() for s in (rec.get("items", [""] * len(rec["form"]))[i] or "").split(",") if s.strip()]
            if form.startswith("4"):
                etype = "form4"
            else:
                etype = "8k_items_" + "_".join(sorted(items)) if items else "8k"
            events.append(
                {
                    "dedupe_key": acc,
                    "event_time_utc": _acc_to_utc(rec["acceptanceDateTime"][i]),
                    "symbol": "TSLA",
                    "type": etype,
                    "title": f"{form} {rec.get('primaryDocDescription', [''] * len(rec['form']))[i] or acc}",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc_nodash}/{doc}",
                    "payload": {
                        "form": form,
                        "accession": acc,
                        "filing_date": rec["filingDate"][i],
                        "report_date": rec["reportDate"][i],
                        "items": items,
                        "size": rec["size"][i],
                    },
                }
            )
        return events


if __name__ == "__main__":
    cli(EdgarCollector)
