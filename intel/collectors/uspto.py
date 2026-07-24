"""USPTO 专利采集器：PatentsView Search API 查 Tesla 新授权/新申请（T2）.

通道：https://search.patentsview.org/api/v1/{patent,publication}/ （免费但需 API key，
在 https://patentsview-support.atlassian.net/servicedesk 申请后放环境变量
PATENTSVIEW_API_KEY 或 data/intel/patentsview.key 文件）。
旧版 api.patentsview.org 已于 2025-02 退役，无 key 时本渠道降级（poll_log 记失败）。
2026-07-24 实测：search.patentsview.org 从本网络直接连接超时（keyless/带 key 均未验通），
渠道暂降级；备选：USPTO Open Data Portal https://api.uspto.gov （MyUSPTO 免费 key，
申请专利搜索端点 /api/v1/patent/applications/search，实测 401 = 通但要 key）。

event_time_utc = 公开日（授权 patent_date / 申请公开 publication_date，日粒度，
补 00:00 UTC）——专利授权每周二批量公开，本渠道天然是慢通道（天级时延）。
过滤：assignee 组织名以 "Tesla" 开头（排除 Nikola/Tesla 命名的无关公司需人工复核）。

用法： .venv/bin/python -m intel.collectors.uspto --once
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intel.collectors.base import Collector, cli, http_get

BASE = "https://search.patentsview.org/api/v1"
KEY_FILE = Path(__file__).resolve().parents[2] / "data" / "intel" / "patentsview.key"
LOOKBACK_DAYS = 90
_TESLA_RE = re.compile(r"^tesla\b", re.IGNORECASE)


def _api_key() -> str | None:
    key = os.environ.get("PATENTSVIEW_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    return None


class UsptoCollector(Collector):
    SOURCE = {
        "source_id": "uspto",
        "name": "USPTO PatentsView: Tesla grants & applications",
        "tier": "T2",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free (API key required)",
        "weight_source": 0.6,
        "notes": "降级中(2026-07-24)：需 PATENTSVIEW_API_KEY，且 search.patentsview.org "
                 "当前网络连接超时；备选 USPTO ODP api.uspto.gov（MyUSPTO 免费 key）。"
                 "授权每周二批量公开，日粒度慢通道",
    }

    def fetch(self):
        key = self._key = _api_key()
        if not key:
            raise RuntimeError(
                "no PatentsView API key (set PATENTSVIEW_API_KEY or "
                f"{KEY_FILE}); channel degraded — 见 sources.notes"
            )
        since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime(
            "%Y-%m-%d"
        )
        out = {}
        specs = {
            "patent": {
                "date_field": "patent_date",
                "q": {"_and": [
                    {"_contains": {"assignees.assignee_organization": "Tesla"}},
                    {"_gte": {"patent_date": since}},
                ]},
                "f": ["patent_id", "patent_title", "patent_date",
                      "assignees.assignee_organization"],
            },
            "publication": {
                "date_field": "publication_date",
                "q": {"_and": [
                    {"_contains": {"assignees.assignee_organization": "Tesla"}},
                    {"_gte": {"publication_date": since}},
                ]},
                "f": ["document_number", "publication_title", "publication_date",
                      "assignees.assignee_organization"],
            },
        }
        for name, spec in specs.items():
            resp = http_get(
                f"{BASE}/{name}/",
                headers={"X-Api-Key": key},
                params={
                    "q": json.dumps(spec["q"]),
                    "f": json.dumps(spec["f"]),
                    "o": json.dumps({"size": 500}),
                    "s": json.dumps([{spec["date_field"]: "desc"}]),
                },
            )
            out[name] = resp.json()
        return out

    def normalize(self, raw) -> list[dict]:
        events = []
        for pat in raw.get("patent", {}).get("patents") or []:
            orgs = [a.get("assignee_organization") or ""
                    for a in pat.get("assignees") or []]
            if not any(_TESLA_RE.match(o) for o in orgs):
                continue
            pid = pat["patent_id"]
            events.append({
                "dedupe_key": f"grant:{pid}",
                "event_time_utc": f"{pat['patent_date']}T00:00:00+00:00",
                "symbol": "TSLA",
                "type": "patent_grant",
                "title": pat.get("patent_title"),
                "url": f"https://patents.google.com/patent/US{pid}",
                "payload": {"patent_id": pid, "assignees": orgs,
                            "date": pat["patent_date"]},
            })
        for pub in raw.get("publication", {}).get("publications") or []:
            orgs = [a.get("assignee_organization") or ""
                    for a in pub.get("assignees") or []]
            if not any(_TESLA_RE.match(o) for o in orgs):
                continue
            doc = pub["document_number"]
            events.append({
                "dedupe_key": f"pub:{doc}",
                "event_time_utc": f"{pub['publication_date']}T00:00:00+00:00",
                "symbol": "TSLA",
                "type": "patent_application",
                "title": pub.get("publication_title"),
                "url": f"https://patents.google.com/patent/US{doc}A1",
                "payload": {"document_number": doc, "assignees": orgs,
                            "date": pub["publication_date"]},
            })
        return events


if __name__ == "__main__":
    cli(UsptoCollector)
