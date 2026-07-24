"""Polymarket 采集器：Tesla/Musk 相关预测市场的日度赔率快照（T3，预测市场）.

通道：Polymarket Gamma 公开 API（免 key）。
    https://gamma-api.polymarket.com/public-search?q=<kw>&events_status=active
实测（2026-07-24）：存在大量 TSLA 相关市场（周度/月度价格区间、"Musk out as
Tesla CEO"、发推数区间等）——渠道有货，正常采集。

口径：这是**快照型**渠道——赔率随时在变，没有"发布时刻"概念。
event_time_utc = 抓取时刻；每市场每 ET 日一条（dedupe_key = market_id|date），
即日度赔率序列。payload 存 outcome 价格/成交量/流动性/截止日。

过滤：event/market 标题匹配 tesla|tsla|musk 且 active、未 closed；
按 24h 成交量排序取前 MAX_MARKETS，过滤零流动性壳市场。

用法： .venv/bin/python -m intel.collectors.polymarket --once
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from intel.collectors.base import Collector, cli, http_get

SEARCH = "https://gamma-api.polymarket.com/public-search"
KEYWORDS = ["tesla", "tsla", "musk"]
RELEVANT_RE = re.compile(r"tesla|tsla|musk", re.I)
MAX_MARKETS = 120
MIN_LIQUIDITY = 100.0  # 美元，滤壳市场


def _floats(s: str | None) -> list[float]:
    try:
        return [float(x) for x in json.loads(s or "[]")]
    except (ValueError, TypeError):
        return []


class PolymarketCollector(Collector):
    SOURCE = {
        "source_id": "polymarket",
        "name": "Polymarket Tesla/Musk markets odds snapshot",
        "tier": "T0",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.4,
        "notes": "快照型渠道：event_time=抓取时刻，每市场每日一条赔率快照；"
                 "关键词 tesla/tsla/musk，流动性>=100 USD；"
                 "归 T0：预测市场赔率视为资金布局痕迹（非法定披露，注意口径差）",
    }

    def fetch(self):
        events = {}
        for kw in KEYWORDS:
            d = http_get(
                SEARCH,
                params={"q": kw, "limit_per_type": 50, "events_status": "active"},
            ).json()
            for ev in d.get("events", []):
                events[ev.get("id")] = ev
        return list(events.values())

    def normalize(self, raw) -> list[dict]:
        now = datetime.now(ZoneInfo("UTC"))
        now_iso = now.isoformat(timespec="seconds")
        et_date = now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        cands = []
        for ev in raw:
            ev_title = ev.get("title") or ""
            for m in ev.get("markets", []):
                q = m.get("question") or ""
                if not (RELEVANT_RE.search(ev_title) or RELEVANT_RE.search(q)):
                    continue
                if m.get("closed") or not m.get("active"):
                    continue
                liq = float(m.get("liquidity") or 0)
                vol = float(m.get("volume") or 0)
                if liq < MIN_LIQUIDITY:
                    continue
                cands.append((vol, ev_title, ev.get("id"), m))
        cands.sort(key=lambda t: -t[0])
        out = []
        for vol, ev_title, ev_id, m in cands[:MAX_MARKETS]:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = _floats(m.get("outcomePrices"))
            out.append(
                {
                    "dedupe_key": f"{m['id']}|{et_date}",
                    "event_time_utc": now_iso,
                    "symbol": "TSLA",
                    "type": "polymarket_odds",
                    "title": (m.get("question") or ev_title)[:200],
                    "url": f"https://polymarket.com/market/{m.get('slug', '')}",
                    "payload": {
                        "market_id": m["id"],
                        "event_id": ev_id,
                        "event_title": ev_title,
                        "question": m.get("question"),
                        "outcomes": outcomes,
                        "prices": prices,
                        "volume_usd": vol,
                        "liquidity_usd": float(m.get("liquidity") or 0),
                        "end_date": m.get("endDate"),
                        "snapshot_et_date": et_date,
                    },
                }
            )
        return out


if __name__ == "__main__":
    cli(PolymarketCollector)
