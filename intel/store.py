"""哨兵（Sentinel）双时间戳事件库 — SQLite 存储层.

设计要点（防前视第一原则的落地）：
- 每条事件带两个时间戳：
    event_time_utc     信息发生/公开发布的时刻（源头声称的时刻）
    observed_time_utc  哨兵看到（入库）的时刻——特征只能在此之后可用
  二者之差 = 该渠道的真实时延，v_latency 视图按渠道统计分布。
- event_id = sha256(source_id | dedupe_key) 前 16 hex，INSERT OR IGNORE 去重，
  同一事件被多次轮询看到只保留首次 observed_time_utc（首见时刻）。
- poll_log 记录每次轮询（成功/失败/新增数），做渠道健康监控。

用法：
    from intel.store import connect, upsert_source, insert_events, record_poll
    python -m intel.store          # 初始化 + 打印当前统计
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "intel" / "sentinel.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    tier             TEXT NOT NULL CHECK (tier IN ('T0','T1','T2','T3')),
    method           TEXT NOT NULL,            -- rss / api / scrape / paid_pending
    poll_interval_s  INTEGER NOT NULL,
    cost             TEXT NOT NULL DEFAULT 'free',
    weight_source    REAL NOT NULL DEFAULT 0.5, -- 身位分初值 0-1
    notes            TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,        -- sha256(source_id|dedupe_key)[:16]
    source_id         TEXT NOT NULL REFERENCES sources(source_id),
    event_time_utc    TEXT NOT NULL,           -- 信息发生/发布时刻 ISO8601 UTC
    observed_time_utc TEXT NOT NULL,           -- 哨兵首次看到时刻 ISO8601 UTC
    symbol            TEXT,                    -- 可空；TSLA 等
    type              TEXT NOT NULL,
    title             TEXT,
    url               TEXT,
    payload_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_source_time ON events (source_id, event_time_utc);
CREATE INDEX IF NOT EXISTS idx_events_observed    ON events (observed_time_utc);

CREATE TABLE IF NOT EXISTS poll_log (
    poll_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL,
    poll_time_utc TEXT NOT NULL,
    ok           INTEGER NOT NULL,             -- 1 成功 0 失败
    n_seen       INTEGER NOT NULL DEFAULT 0,   -- 本次抓到的条目数
    n_new        INTEGER NOT NULL DEFAULT 0,   -- 其中新入库数
    duration_ms  INTEGER,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_poll_source_time ON poll_log (source_id, poll_time_utc);

-- 渠道时延视图：observed - event 的秒级分布（负值=日历类"预告未来"事件，属正常）
DROP VIEW IF EXISTS v_latency;
CREATE VIEW v_latency AS
SELECT source_id,
       COUNT(*)                    AS n_events,
       ROUND(MIN(lag_s), 1)        AS lag_min_s,
       ROUND(AVG(lag_s), 1)        AS lag_avg_s,
       ROUND(MAX(lag_s), 1)        AS lag_max_s
FROM (
    SELECT source_id,
           (julianday(observed_time_utc) - julianday(event_time_utc)) * 86400.0 AS lag_s
    FROM events
)
GROUP BY source_id;
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate_sources_tier_t0(conn: sqlite3.Connection) -> None:
    """兼容迁移：旧库 sources.tier CHECK 不含 T0 → 重建表放开（保留数据）.

    SQLite 改 CHECK 只能重建表。events 的外键按表名引用 sources，
    先建新表→搬数据→删旧表→改名，数据与外键引用均不受影响。
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'sources' AND type = 'table'"
    ).fetchone()
    if row is None or "'T0'" in (row[0] or ""):
        return  # 表不存在（executescript 会建新版）或已是新版
    conn.executescript(
        """
        PRAGMA foreign_keys=OFF;
        BEGIN;
        CREATE TABLE sources_new (
            source_id        TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            tier             TEXT NOT NULL CHECK (tier IN ('T0','T1','T2','T3')),
            method           TEXT NOT NULL,
            poll_interval_s  INTEGER NOT NULL,
            cost             TEXT NOT NULL DEFAULT 'free',
            weight_source    REAL NOT NULL DEFAULT 0.5,
            notes            TEXT DEFAULT ''
        );
        INSERT INTO sources_new SELECT * FROM sources;
        DROP TABLE sources;
        ALTER TABLE sources_new RENAME TO sources;
        COMMIT;
        """
    )


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    _migrate_sources_tier_t0(conn)
    conn.executescript(_SCHEMA)
    return conn


def make_event_id(source_id: str, dedupe_key: str) -> str:
    return hashlib.sha256(f"{source_id}|{dedupe_key}".encode()).hexdigest()[:16]


def upsert_source(conn: sqlite3.Connection, s: dict) -> None:
    """注册/更新渠道。weight_source 只在首插时生效（后续人工调权不被覆盖）。"""
    conn.execute(
        """INSERT INTO sources (source_id, name, tier, method, poll_interval_s, cost,
                                weight_source, notes)
           VALUES (:source_id, :name, :tier, :method, :poll_interval_s, :cost,
                   :weight_source, :notes)
           ON CONFLICT(source_id) DO UPDATE SET
               name=excluded.name, tier=excluded.tier, method=excluded.method,
               poll_interval_s=excluded.poll_interval_s, cost=excluded.cost,
               notes=excluded.notes""",
        {"cost": "free", "weight_source": 0.5, "notes": "", **s},
    )
    conn.commit()


def insert_events(conn: sqlite3.Connection, source_id: str, events: list[dict]) -> int:
    """批量入库，去重后返回真正新增数.

    每个 event dict 需要：dedupe_key, event_time_utc, type；
    可选：symbol, title, url, payload（dict，自动转 JSON）。
    """
    now = utcnow_iso()
    n_new = 0
    for ev in events:
        payload = ev.get("payload")
        cur = conn.execute(
            """INSERT OR IGNORE INTO events
               (event_id, source_id, event_time_utc, observed_time_utc,
                symbol, type, title, url, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                make_event_id(source_id, str(ev["dedupe_key"])),
                source_id,
                ev["event_time_utc"],
                now,
                ev.get("symbol"),
                ev["type"],
                (ev.get("title") or "")[:500] or None,
                ev.get("url"),
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
            ),
        )
        n_new += cur.rowcount
    conn.commit()
    return n_new


def record_poll(
    conn: sqlite3.Connection,
    source_id: str,
    ok: bool,
    n_seen: int = 0,
    n_new: int = 0,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO poll_log (source_id, poll_time_utc, ok, n_seen, n_new,
                                 duration_ms, error)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (source_id, utcnow_iso(), int(ok), n_seen, n_new, duration_ms,
         (error or "")[:1000] or None),
    )
    conn.commit()


def latency_stats(conn: sqlite3.Connection) -> list[dict]:
    """v_latency 视图 + Python 算 p50（SQLite 无内置中位数）。"""
    out = []
    for r in conn.execute("SELECT * FROM v_latency ORDER BY source_id"):
        row = dict(r)
        lags = [
            x[0]
            for x in conn.execute(
                """SELECT (julianday(observed_time_utc) - julianday(event_time_utc))
                          * 86400.0
                   FROM events WHERE source_id = ? ORDER BY 1""",
                (row["source_id"],),
            )
        ]
        row["lag_p50_s"] = round(lags[len(lags) // 2], 1) if lags else None
        out.append(row)
    return out


def summary(conn: sqlite3.Connection) -> str:
    lines = ["== sources =="]
    for r in conn.execute("SELECT * FROM sources ORDER BY tier, source_id"):
        lines.append(
            f"  {r['source_id']:<14} {r['tier']} {r['method']:<12} "
            f"poll={r['poll_interval_s']}s cost={r['cost']} w={r['weight_source']}"
        )
    lines.append("== events by source ==")
    for r in conn.execute(
        """SELECT source_id, COUNT(*) n, MAX(event_time_utc) latest
           FROM events GROUP BY source_id ORDER BY source_id"""
    ):
        lines.append(f"  {r['source_id']:<14} {r['n']:>5}  latest_event={r['latest']}")
    lines.append("== latency (observed - event, seconds) ==")
    for row in latency_stats(conn):
        lines.append(
            f"  {row['source_id']:<14} n={row['n_events']:>5}  "
            f"min={row['lag_min_s']}  p50={row['lag_p50_s']}  "
            f"avg={row['lag_avg_s']}  max={row['lag_max_s']}"
        )
    lines.append("== last polls ==")
    for r in conn.execute(
        """SELECT source_id, MAX(poll_time_utc) t, ok, n_seen, n_new, error
           FROM poll_log GROUP BY source_id ORDER BY source_id"""
    ):
        lines.append(
            f"  {r['source_id']:<14} {r['t']} ok={r['ok']} "
            f"seen={r['n_seen']} new={r['n_new']}"
            + (f" err={r['error'][:60]}" if r["error"] else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    c = connect()
    print(f"db: {DB_PATH}")
    print(summary(c))
