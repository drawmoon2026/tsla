"""X/Twitter 采集器：Nitter 镜像 RSS（T3，脆弱通道，付费备选已登记）.

2026-07-24 实测结论（详见 intel/README.md）：
- nitter.net/<user>/rss  ✅ 可用，返回最近 20 帖，时延分钟级——本模块用它
- syndication.twitter.com timeline-profile  ✗ 返回空壳 HTML，无时间线数据
- nitter.poast.org / lightbrd.com / twiiit.com  ✗ 403 反爬墙
Nitter 实例历史上反复死亡，此通道随时可能失效——poll_log 连续失败即为信号，
届时启用付费通道（sources 表已登记 x_api_paid：X API Basic 档 ~$200/月）。

event_time_utc = 发帖时刻（pubDate，发帖即公开）。RT 的 pubDate 是原帖时间，
type 区分 *_rt。只抓不硬爬：单账号单请求，轮询间隔 ≥5 分钟。

用法： .venv/bin/python -m intel.collectors.x_nitter --once
"""

from __future__ import annotations

import feedparser

from intel.collectors.base import Collector, cli, http_get, struct_time_to_utc_iso

BASE = "https://nitter.net/{user}/rss"
ACCOUNTS = ["elonmusk", "Tesla"]
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# 付费备选，由 run_sentinel 登记进 sources 表
PAID_FALLBACK = {
    "source_id": "x_api_paid",
    "name": "X API v2 (付费备选，未启用)",
    "tier": "T3",
    "method": "paid_pending",
    "poll_interval_s": 0,
    "cost": "Basic ~$200/月(读额度~1.5万帖/月) / Pro ~$5000/月；价格以 developer.x.com 现价为准",
    "weight_source": 0.3,
    "notes": "nitter 通道死亡时启用；Free 档只写不读，不可用于采集",
}


class XNitterCollector(Collector):
    SOURCE = {
        "source_id": "x_nitter",
        "name": "X via nitter.net RSS (elonmusk + Tesla)",
        "tier": "T3",
        "method": "rss",
        "poll_interval_s": 300,
        "cost": "free",
        "weight_source": 0.3,
        "notes": "脆弱通道：nitter 实例随时可能死；每账号仅最近20帖；RT 时间为原帖时间",
    }

    def fetch(self):
        out = []
        errors = []
        failed_users = set()
        for user in ACCOUNTS:
            try:
                resp = http_get(BASE.format(user=user),
                                headers={"User-Agent": _BROWSER_UA})
                feed = feedparser.parse(resp.content)
                if not feed.entries:
                    raise RuntimeError("empty feed (可能被反爬/实例降级)")
                out.append((user, feed))
            except Exception as e:  # noqa: BLE001
                failed_users.add(user)
                errors.append(f"{user}: {type(e).__name__}: {e}")
        # P0-1a：elonmusk 账号是探测器腿 B 唯一前向数据源——它失败即整渠道失败
        # （poll_log ok=0、连败计数生效、detector 的 blind 判定得以感知），
        # 即便 Tesla 账号成功也不算本轮成功（其事件下轮 recent-20 会补上）。
        if "elonmusk" in failed_users:
            raise RuntimeError("elonmusk feed 失败（腿 B 唯一数据源，按整渠道失败处理）: "
                               + "; ".join(errors))
        if not out:
            raise RuntimeError("; ".join(errors))
        if errors:
            print(f"  [x_nitter] partial fail: {'; '.join(errors)}")
        return out

    def normalize(self, raw) -> list[dict]:
        events = []
        for user, feed in raw:
            for e in feed.entries:
                published = struct_time_to_utc_iso(e.get("published_parsed"))
                link = e.get("link", "")
                if not published or not link:
                    continue
                creator = (e.get("author") or "").lstrip("@")
                is_rt = creator.lower() != user.lower()
                base = "musk" if user == "elonmusk" else "tesla_co"
                events.append({
                    "dedupe_key": link.split("#")[0],
                    "event_time_utc": published,
                    "symbol": "TSLA",
                    "type": f"x_{base}_rt" if is_rt else f"x_{base}_post",
                    "title": (e.get("title") or "")[:300],
                    "url": link.replace("nitter.net", "x.com"),
                    "payload": {"account": user, "author": creator, "is_rt": is_rt,
                                "text": (e.get("description") or "")[:2000]},
                })
        return events


if __name__ == "__main__":
    cli(XNitterCollector)
