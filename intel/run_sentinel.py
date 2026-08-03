"""哨兵总入口：跑一轮所有渠道采集器.

用法：
    .venv/bin/python -m intel.run_sentinel --once          # 全渠道跑一轮
    .venv/bin/python -m intel.run_sentinel --once --auto   # launchd 模式（见下）
    .venv/bin/python -m intel.run_sentinel --status        # 只打印库内统计

--auto（配合 launchd 每 5 分钟触发）：
    盘中（美东工作日 09:30-16:00）每次触发都跑；
    盘外只有当分钟落在 [0,5) 或 [30,35) 才跑（≈每 30 分钟一轮），其余触发直接退出。
慢渠道（fed/uspto，poll_interval_s=86400）自带节流：距上次成功轮询不足
poll_interval_s 的渠道本轮跳过（--force 忽略节流）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from intel import store
from intel.collectors.earnings_cal import EarningsCalCollector
from intel.collectors.edgar import EdgarCollector
from intel.collectors.edgar_t0 import EdgarT0Collector
from intel.collectors.fed import FedCollector
from intel.collectors.finra_ats import FinraAtsCollector
from intel.collectors.finra_short import FinraShortCollector
from intel.collectors.macro_fred import MacroFredCollector
from intel.collectors.news_rss import NewsRssCollector
from intel.collectors.options_snapshot import OptionsSnapshotCollector
from intel.collectors.polymarket import PolymarketCollector
from intel.collectors.uspto import UsptoCollector
from intel.collectors.x_nitter import PAID_FALLBACK, XNitterCollector
from intel.collectors.youtube import YoutubeCollector
from intel.detector import CausalDetector

COLLECTORS = [
    EdgarCollector,
    EdgarT0Collector,
    FinraShortCollector,
    FinraAtsCollector,
    OptionsSnapshotCollector,
    FedCollector,
    EarningsCalCollector,
    MacroFredCollector,
    UsptoCollector,
    YoutubeCollector,
    NewsRssCollector,
    XNitterCollector,
    PolymarketCollector,
    CausalDetector,   # 值班组件（非采集渠道）：必须排在所有采集器之后跑状态机
]


def _market_open_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= hm < 16 * 60


def _should_run_auto() -> bool:
    if _market_open_now():
        return True
    return datetime.now(timezone.utc).minute % 30 < 5


def _throttled(conn, source_id: str, interval_s: int) -> bool:
    """距上次轮询尝试（无论成败）不足 interval_s 则节流.

    按"尝试"而非"成功"计时：持续降级的渠道（如缺 key 的 uspto）按自身
    poll_interval_s 重试，而不是每 5 分钟刷一行失败 poll_log。
    """
    row = conn.execute(
        "SELECT MAX(poll_time_utc) t FROM poll_log WHERE source_id=?",
        (source_id,),
    ).fetchone()
    if not row or not row["t"]:
        return False
    last = datetime.fromisoformat(row["t"])
    return (datetime.now(timezone.utc) - last).total_seconds() < interval_s * 0.9


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="跑一轮所有渠道")
    p.add_argument("--auto", action="store_true", help="launchd 模式：按盘中/盘后节奏决定本次是否执行")
    p.add_argument("--force", action="store_true", help="忽略慢渠道节流")
    p.add_argument("--status", action="store_true", help="只打印数据库统计")
    args = p.parse_args()

    conn = store.connect()
    store.upsert_source(conn, PAID_FALLBACK)  # 登记付费备选通道（不采集）

    if args.status or not args.once:
        print(f"db: {store.DB_PATH}")
        print(store.summary(conn))
        conn.close()
        return 0

    if args.auto and not _should_run_auto():
        conn.close()
        return 0

    print(f"=== sentinel run {store.utcnow_iso()} ===")
    results = []
    for cls in COLLECTORS:
        sid = cls.SOURCE["source_id"]
        if not args.force and _throttled(conn, sid, cls.SOURCE["poll_interval_s"]):
            print(f"[{sid}] skipped (within poll_interval)")
            continue
        results.append(cls().run_once())
    conn.close()

    n_fail = sum(1 for r in results if not r["ok"])
    n_new = sum(r["n_new"] for r in results)
    print(f"=== done: {len(results)} channels polled, {n_new} new events, "
          f"{n_fail} failures ===")
    return 1 if n_fail == len(results) and results else 0


if __name__ == "__main__":
    sys.exit(main())
