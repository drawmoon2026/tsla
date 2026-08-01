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
    # P1-7：官方频道 feed 无关键词过滤恒回最近 15 条，持续 seen=0 = 渠道级异常；
    # 15 分钟一轮，连续 8 轮（约 2 小时）零产出即告警
    ZERO_SEEN_ALERT_N = 8

    SOURCE = {
        "source_id": "youtube",
        "name": "YouTube channel RSS (Tesla official + 财经媒体)",
        "tier": "T3",
        "method": "rss",
        "poll_interval_s": 900,
        "cost": "free",
        "weight_source": 0.3,
        "notes": "官方频道全收；媒体频道标题含 tesla/tsla/musk 才入库；每频道仅最近15条；"
                 "单频道失败不拖垮整渠道（2026-08-01 查证：Tesla 官方 channel_id 有效，"
                 "但 YouTube feed 端点存在间歇性 404，非 ID 失效）",
    }

    def fetch(self):
        # 单频道容错（P2-11）：一个频道 404/超时不拖垮其余频道；全军覆没才算渠道失败。
        # 2026-08-01 查证：UC5WjFrtBdufl6CZojX3D8dQ（Tesla 官方）仍有效，404 为
        # YouTube 端间歇性抽风——按单频道降级处理并打印，不改 channel_id。
        out = []
        errors = []
        for cid, label, kw_only in CHANNELS:
            try:
                resp = http_get(FEED.format(cid=cid))
                out.append((label, kw_only, feedparser.parse(resp.content)))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}({cid}): {type(e).__name__}: {e}")
        if not out:
            raise RuntimeError("; ".join(errors))
        if errors:
            print(f"  [youtube] partial fail: {'; '.join(errors)}")
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
