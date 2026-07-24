"""YouTube 频道采集器：高影响频道 RSS（T3）.

通道：https://www.youtube.com/feeds/videos.xml?channel_id=<ID>（免 key，每频道
最近 15 条视频）。event_time_utc = <published>（视频发布时刻，秒级）。
频道清单在 CHANNELS 配置化，加频道 = 加一行 (channel_id, 标签)。
频道 ID 获取：频道页源码搜 "channelId" 或 externalId。

用法： .venv/bin/python -m intel.collectors.youtube --once
"""

from __future__ import annotations

import feedparser

from intel.collectors.base import Collector, cli, http_get, struct_time_to_utc_iso

FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"

# (channel_id, 标签, 是否只要含 Tesla/TSLA 关键词的标题)
CHANNELS = [
    ("UC5WjFrtBdufl6CZojX3D8dQ", "tesla_official", False),   # Tesla 官方
    ("UCvJJ_dzjViJCoLf5uKUTwoA", "cnbc", True),              # CNBC 主频道
    ("UCrp_UI8XtuYfpiqluWLD7Lw", "cnbc_tv", True),           # CNBC Television
    ("UCIALMKvObZNtJ6AmdCLP7Lg", "bloomberg_tv", True),      # Bloomberg Television
    ("UCEAZeUIeJs0IjQiqTCdVSIg", "yahoo_finance", True),     # Yahoo Finance
]
KEYWORDS = ("tesla", "tsla", "musk")


class YoutubeCollector(Collector):
    SOURCE = {
        "source_id": "youtube",
        "name": "YouTube channel RSS (Tesla official + 财经媒体)",
        "tier": "T3",
        "method": "rss",
        "poll_interval_s": 900,
        "cost": "free",
        "weight_source": 0.3,
        "notes": "官方频道全收；媒体频道标题含 tesla/tsla/musk 才入库；每频道仅最近15条",
    }

    def fetch(self):
        out = []
        for cid, label, kw_only in CHANNELS:
            resp = http_get(FEED.format(cid=cid))
            out.append((label, kw_only, feedparser.parse(resp.content)))
        return out

    def normalize(self, raw) -> list[dict]:
        events = []
        for label, kw_only, feed in raw:
            for e in feed.entries:
                title = e.get("title", "")
                if kw_only and not any(k in title.lower() for k in KEYWORDS):
                    continue
                vid = e.get("yt_videoid") or e.get("id", "")
                published = struct_time_to_utc_iso(e.get("published_parsed"))
                if not published or not vid:
                    continue
                events.append({
                    "dedupe_key": f"{label}:{vid}",
                    "event_time_utc": published,
                    "symbol": "TSLA" if (label == "tesla_official" or not kw_only
                                         or any(k in title.lower() for k in KEYWORDS))
                              else None,
                    "type": f"youtube_{label}",
                    "title": title,
                    "url": e.get("link"),
                    "payload": {"channel": label, "video_id": vid,
                                "author": e.get("author", "")},
                })
        return events


if __name__ == "__main__":
    cli(YoutubeCollector)
