"""采集器统一基类：fetch → normalize → dedupe → store，附限速 HTTP 与 CLI.

子类只需定义：
    SOURCE      渠道注册表行（dict，进 sources 表）
    fetch()     发网络请求，返回原始数据（任意类型）
    normalize(raw) -> list[dict]，每条含 dedupe_key / event_time_utc / type，
                  可选 symbol / title / url / payload
去重与入库由基类走 intel.store（INSERT OR IGNORE by event_id），
每次运行写一行 poll_log。

单渠道运行： .venv/bin/python -m intel.collectors.<name> --once
"""

from __future__ import annotations

import argparse
import calendar
import time
import traceback
from datetime import datetime, timezone

import requests

from intel import store

UA = "TSLA-research sentinel tom (drawmoon2026@gmail.com)"
_MIN_INTERVAL = 0.15  # 全局 ~6.7 req/s，低于 SEC 的 10 req/s 红线
_last_req = [0.0]


def http_get(url: str, timeout: int = 30, retries: int = 3, headers: dict | None = None,
             params: dict | None = None) -> requests.Response:
    """带 User-Agent、全局限速、简单重试的 GET。"""
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            resp = requests.get(url, headers=h, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def struct_time_to_utc_iso(st) -> str | None:
    """feedparser 的 *_parsed（UTC struct_time）→ ISO8601 UTC。"""
    if not st:
        return None
    return datetime.fromtimestamp(
        calendar.timegm(st), tz=timezone.utc
    ).isoformat(timespec="seconds")


class Collector:
    SOURCE: dict = {}  # 子类覆盖

    def fetch(self):
        raise NotImplementedError

    def normalize(self, raw) -> list[dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------ run

    def run_once(self, verbose: bool = True) -> dict:
        sid = self.SOURCE["source_id"]
        conn = store.connect()
        store.upsert_source(conn, self.SOURCE)
        t0 = time.time()
        try:
            raw = self.fetch()
            events = self.normalize(raw)
            n_new = store.insert_events(conn, sid, events)
            dur = int((time.time() - t0) * 1000)
            store.record_poll(conn, sid, ok=True, n_seen=len(events), n_new=n_new,
                              duration_ms=dur)
            stats = {"source_id": sid, "ok": True, "n_seen": len(events),
                     "n_new": n_new, "duration_ms": dur, "error": None}
        except Exception as e:  # noqa: BLE001
            dur = int((time.time() - t0) * 1000)
            err = f"{type(e).__name__}: {e}"
            store.record_poll(conn, sid, ok=False, duration_ms=dur, error=err)
            stats = {"source_id": sid, "ok": False, "n_seen": 0, "n_new": 0,
                     "duration_ms": dur, "error": err}
            if verbose:
                traceback.print_exc()
        finally:
            conn.close()
        if verbose:
            if stats["ok"]:
                print(f"[{sid}] ok  seen={stats['n_seen']} new={stats['n_new']} "
                      f"({stats['duration_ms']}ms)")
            else:
                print(f"[{sid}] FAIL {stats['error']} ({stats['duration_ms']}ms)")
        return stats


def cli(collector_cls) -> None:
    """`python -m intel.collectors.<name> --once` 的统一入口。"""
    p = argparse.ArgumentParser(description=collector_cls.__doc__)
    p.add_argument("--once", action="store_true", help="跑一轮（当前唯一模式）")
    p.parse_args()
    collector_cls().run_once()
