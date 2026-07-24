"""财经新闻 RSS 聚合采集器（T3）.

FEEDS 配置化列表；标题/摘要含 Tesla/TSLA/Musk 关键词才入库（symbol 专属 feed 全收）。
event_time_utc = RSS <pubDate>（发布时刻）。Reuters 公共 RSS 已于 2020 关闭，不列；
Google News RSS 是聚合器（转发源文章，时延=源发布→Google 收录）。

用法： .venv/bin/python -m intel.collectors.news_rss --once
"""

from __future__ import annotations

import feedparser

from intel.collectors.base import Collector, cli, http_get, struct_time_to_utc_iso

# (feed_id, url, 是否 TSLA 专属（专属则不做关键词过滤）)
FEEDS = [
    ("yahoo_tsla",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
     True),
    ("cnbc_top",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
     False),
    ("cnbc_markets",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
     False),
    ("marketwatch_top",
     "https://feeds.content.dowjones.io/public/rss/mw_topstories",
     False),
    ("google_news_tsla",
     "https://news.google.com/rss/search?q=Tesla%20OR%20TSLA%20when:1d&hl=en-US&gl=US&ceid=US:en",
     True),
]
KEYWORDS = ("tesla", "tsla", "musk")


class NewsRssCollector(Collector):
    SOURCE = {
        "source_id": "news_rss",
        "name": "财经 RSS 聚合 (Yahoo/CNBC/MarketWatch/GoogleNews)",
        "tier": "T3",
        "method": "rss",
        "poll_interval_s": 300,
        "cost": "free",
        "weight_source": 0.3,
        "notes": "通用 feed 按 tesla/tsla/musk 关键词过滤；google_news 为二手聚合",
    }

    def fetch(self):
        out = []
        for fid, url, tsla_only in FEEDS:
            try:
                resp = http_get(url)
                out.append((fid, tsla_only, feedparser.parse(resp.content), None))
            except Exception as e:  # 单 feed 挂掉不拖垮整个渠道
                out.append((fid, tsla_only, None, f"{type(e).__name__}: {e}"))
        return out

    def normalize(self, raw) -> list[dict]:
        events = []
        dead = []
        for fid, tsla_only, feed, err in raw:
            if feed is None:
                dead.append(f"{fid}: {err}")
                continue
            for e in feed.entries:
                title = e.get("title", "")
                summary = e.get("summary", "")
                text = f"{title} {summary}".lower()
                if not tsla_only and not any(k in text for k in KEYWORDS):
                    continue
                published = struct_time_to_utc_iso(
                    e.get("published_parsed") or e.get("updated_parsed"))
                link = e.get("link", "")
                if not published:
                    continue
                events.append({
                    "dedupe_key": f"{fid}:{e.get('id') or link or title}",
                    "event_time_utc": published,
                    "symbol": "TSLA",
                    "type": f"news_{fid}",
                    "title": title,
                    "url": link,
                    "payload": {"feed": fid, "summary": summary[:1000],
                                "source_attr": (e.get("source") or {}).get("title", "")},
                })
        if dead:
            print(f"  [news_rss] dead feeds this poll: {'; '.join(dead)}")
        return events


if __name__ == "__main__":
    cli(NewsRssCollector)
