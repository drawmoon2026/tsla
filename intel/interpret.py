"""LLM 晨间情报解读 —— 本机 Claude Code 无头 CLI 对哨兵新事件做语义解读（M2）.

框架依据 docs/intel-framework.md 第三节（LLM 信息处理层）：
- **只做提炼，不做决策**：输出方向/强度/一句话解读，不给买卖建议；
- **防污染声明：历史段标注不可信（模型记得历史），解读只覆盖本模块上线运行日
  之后哨兵新看到的事件，只前向追加，永不回填历史**。干净样本靠前向积累。

数据流：
    events（哨兵入库） → 取自上次成功解读以来的新事件
      · T0/T1/衍生信号：全量（防爆硬上限 40 条）
      · T2 日历/专利：全量（量小）
      · T3 新闻/发帖/视频：标题指纹去重后限量 15 条（防 prompt 爆炸）
    → 紧凑 digest → `claude -p <prompt> --output-format json`（subprocess，300s 超时）
    → 严格 JSON：整体态势一句话 + 逐事件 {event_id, direction, strength, note, surprise}
    → 写入 llm_interpretations（幂等：event_id 主键，同批事件不重复解读）
    → 运行记录写 llm_interpret_runs（ok / absent；absent = 降级「今日解读缺席」+ 原因）

调用频率控制（挂 com.tsla.sentinel 链尾，每 5 分钟被触发一次）：
    --auto 模式下，若「今天（ET）已有成功解读」且「无未解读的 T0/T1/衍生新事件」
    则静默跳过——正常每天实际调用 claude 1-2 次。

成本与失败面：
    使用本机 Claude 订阅额度（Claude Code CLI，默认模型）；claude CLI 不存在 /
    未认证 / 超时 / 返回不可解析 → 优雅降级为 absent 运行记录，不影响哨兵采集链。

用法：
    .venv/bin/python -m intel.interpret --once    # 立即跑一次（幂等）
    .venv/bin/python -m intel.interpret --auto    # 链尾模式（按频率控制跳过）
    .venv/bin/python -m intel.interpret --status  # 打印解读表统计
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from intel import store

ET = ZoneInfo("America/New_York")

CLAUDE_TIMEOUT_S = 300      # claude -p 超时
T3_LIMIT = 15               # T3 新闻去重后限量（防 prompt 爆炸）
HI_LIMIT = 40               # T0/T1/T2/衍生 硬上限（正常远达不到）
FIRST_RUN_WINDOW_H = 24     # 首跑取最近 24h
MAX_WINDOW_D = 7            # 链断多日后回看上限（仍是前向：只解读未解读过的）

T3_TYPE_PREFIXES = ("news_", "youtube", "x_")

# 防污染声明（同时写进表注释与本 docstring）：解读只覆盖运行日之后的新事件，
# 只前向追加，永不回填历史——LLM 记得历史，历史段标注不可信。
_SCHEMA = """
-- llm_interpretations：LLM 对单条事件的语义解读。
-- 防污染：只前向追加（event_id 主键幂等），永不回填历史事件——
-- 模型记得历史，历史段解读不可信；干净样本自本模块上线日起前向积累。
CREATE TABLE IF NOT EXISTS llm_interpretations (
    interp_date TEXT NOT NULL,            -- 解读运行日（ET, YYYY-MM-DD）
    event_id    TEXT NOT NULL PRIMARY KEY REFERENCES events(event_id),
    direction   TEXT NOT NULL CHECK (direction IN ('bullish','bearish','neutral')),
    strength    INTEGER NOT NULL CHECK (strength BETWEEN 1 AND 3),
    note        TEXT NOT NULL,            -- 一句话解读（中文）
    surprise    INTEGER NOT NULL DEFAULT 0,  -- 1 = 偏离市场普遍预期
    model_meta  TEXT,                     -- claude CLI 返回的元信息 JSON
    created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_interp_date ON llm_interpretations (interp_date);

-- llm_interpret_runs：每次实际调用（或降级）的运行记录；静默跳过不记录。
-- status: ok = 成功；absent = 降级「今日解读缺席」（reason 记原因）。
CREATE TABLE IF NOT EXISTS llm_interpret_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date    TEXT NOT NULL,            -- ET 日
    status      TEXT NOT NULL CHECK (status IN ('ok','absent')),
    reason      TEXT,                     -- absent 原因（超时/CLI 缺失/解析失败…）
    overall     TEXT,                     -- 整体态势一句话（ok 时）
    n_events    INTEGER NOT NULL DEFAULT 0,
    model_meta  TEXT,
    created_utc TEXT NOT NULL
);
"""

_PROMPT_HEAD = """你是 TSLA 情报哨兵的解读员。职责是提炼，不是决策：不给任何买卖建议。
下面是哨兵自上次解读以来新采集到的事件（tier 含义：T0 布局痕迹 / T1 法定披露 /
T2 官方日历承诺 / T3 新闻叙事 / D 衍生信号=自家算法读数）。

请只输出一个严格 JSON 对象（不要 markdown 代码块，不要任何多余文字），格式：
{"overall": "整体态势一句话（中文，≤80字）",
 "events": [
   {"event_id": "…", "direction": "bullish|bearish|neutral", "strength": 1,
    "note": "一句话解读（中文，≤60字）", "surprise": false}
 ]}

要求：
- events 必须覆盖下面列出的每一个 event_id，不多不少；
- direction 仅限 bullish / bearish / neutral 三值（对 TSLA 股价的影响方向）；
- strength 为整数 1（弱）/ 2（中）/ 3（强）；
- surprise 表示该事件是否偏离市场普遍预期（意外为 true）；
- 日历预告、例行快照类事件通常 neutral / 强度 1 / 非意外，除非内容本身异常。

事件列表（格式：event_id | tier | 类型 | 事件时刻UTC | 来源 | 标题 | 关键payload）：
"""


def _connect() -> sqlite3.Connection:
    conn = store.connect()
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------- 选事件

def _payload_brief(payload_json: str | None, limit: int = 160) -> str:
    """payload 关键字段摘要：扁平 k=v 拼接，截断防 prompt 爆炸。"""
    if not payload_json:
        return ""
    try:
        d = json.loads(payload_json)
    except ValueError:
        return ""
    if not isinstance(d, dict):
        return str(d)[:limit]
    parts = []
    for k, v in d.items():
        if v is None or isinstance(v, (list, dict)) and not v:
            continue
        s = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
        parts.append(f"{k}={s[:60]}")
    return "; ".join(parts)[:limit]


def _title_key(title: str | None) -> str:
    """T3 去重键：标题小写去标点词序列（粗粒度，与仪表盘指纹同思路的轻量版）。"""
    import re
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))[:120]


def _since_utc(conn: sqlite3.Connection, now: datetime) -> datetime:
    """选事件下界：上次成功解读时刻；首跑取 24h；链断最多回看 MAX_WINDOW_D 天。"""
    row = conn.execute(
        "SELECT MAX(created_utc) FROM llm_interpret_runs WHERE status='ok'"
    ).fetchone()
    if row and row[0]:
        try:
            last = datetime.fromisoformat(row[0])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            return max(last, now - timedelta(days=MAX_WINDOW_D))
        except ValueError:
            pass
    return now - timedelta(hours=FIRST_RUN_WINDOW_H)


def select_events(conn: sqlite3.Connection, now: datetime) -> list[dict]:
    """取自上次解读以来、尚未解读过的新事件（observed 口径，前向）。"""
    since = _since_utc(conn, now).isoformat(timespec="seconds")
    rows = [dict(r) for r in conn.execute(
        """SELECT e.event_id, e.observed_time_utc, e.event_time_utc, e.source_id,
                  e.type, e.title, e.payload_json, s.tier
           FROM events e LEFT JOIN sources s ON s.source_id = e.source_id
           WHERE e.observed_time_utc >= ?
             AND e.event_id NOT IN (SELECT event_id FROM llm_interpretations)
           ORDER BY e.observed_time_utc DESC""",
        (since,),
    )]
    hi, t3, seen_titles = [], [], set()
    for e in rows:
        t = e.get("type") or ""
        if any(t.startswith(p) for p in T3_TYPE_PREFIXES):
            key = _title_key(e.get("title"))
            if key and key in seen_titles:
                continue  # 同题跨源去重
            seen_titles.add(key)
            if len(t3) < T3_LIMIT:
                t3.append(e)
        elif len(hi) < HI_LIMIT:
            hi.append(e)  # T0/T1/T2/衍生：全量（硬上限防爆）
    return hi + t3


def has_new_hi(events: list[dict]) -> bool:
    """未解读事件中是否含 T0/T1/衍生（--auto 频率控制判据）。"""
    return any(
        e.get("tier") in ("T0", "T1") or e.get("source_id") == "detector"
        for e in events
    )


# ---------------------------------------------------------------- 调 claude

def _find_claude() -> str | None:
    path = shutil.which("claude")
    if path:
        return path
    for cand in (os.path.expanduser("~/.local/bin/claude"),
                 os.path.expanduser("~/.claude/local/claude"),
                 "/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if os.path.exists(cand):
            return cand
    return None


def build_prompt(events: list[dict]) -> str:
    lines = []
    for e in events:
        tier = "D" if e.get("source_id") == "detector" else (e.get("tier") or "T3")
        lines.append(
            f'{e["event_id"]} | {tier} | {e.get("type") or "—"} | '
            f'{e.get("event_time_utc") or "—"} | {e.get("source_id")} | '
            f'{(e.get("title") or "")[:120]} | {_payload_brief(e.get("payload_json"))}'
        )
    return _PROMPT_HEAD + "\n".join(lines)


def _extract_json(text: str) -> dict:
    """模型返回文本 → JSON 对象（容忍 markdown 代码块 / 前后杂文字）。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t.split("\n", 1)[1] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("返回中未找到 JSON 对象")
    return json.loads(t[i:j + 1])


def call_claude(prompt: str) -> tuple[dict, dict]:
    """调 claude -p，返回 (解析后的解读 JSON, model_meta)。异常上抛给降级处理。"""
    exe = _find_claude()
    if exe is None:
        raise FileNotFoundError("claude CLI 不在 PATH（未安装或未登录环境）")
    proc = subprocess.run(
        [exe, "-p", prompt, "--output-format", "json"],
        capture_output=True, text=True, timeout=CLAUDE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude 退出码 {proc.returncode}: {(proc.stderr or proc.stdout)[:300]}"
        )
    outer = json.loads(proc.stdout)
    if isinstance(outer, list):  # 兼容 stream-json 形态
        outer = next((o for o in reversed(outer) if o.get("type") == "result"), {})
    if outer.get("is_error"):
        raise RuntimeError(f"claude 返回错误: {str(outer.get('result'))[:300]}")
    meta = {k: outer.get(k) for k in
            ("total_cost_usd", "duration_ms", "num_turns", "session_id")
            if outer.get(k) is not None}
    usage = outer.get("modelUsage")
    if isinstance(usage, dict) and usage:
        meta["model"] = next(iter(usage))
    parsed = _extract_json(str(outer.get("result") or ""))
    return parsed, meta


# ---------------------------------------------------------------- 落库

def _validate(parsed: dict, valid_ids: set[str]) -> tuple[str, list[dict]]:
    overall = str(parsed.get("overall") or "").strip()
    if not overall:
        raise ValueError("缺 overall 整体态势句")
    out = []
    for it in parsed.get("events") or []:
        eid = str(it.get("event_id") or "")
        if eid not in valid_ids:
            continue  # 幻觉 id 丢弃
        d = str(it.get("direction") or "").lower()
        if d not in ("bullish", "bearish", "neutral"):
            continue
        try:
            strength = min(3, max(1, int(it.get("strength") or 1)))
        except (TypeError, ValueError):
            strength = 1
        out.append({
            "event_id": eid, "direction": d, "strength": strength,
            "note": str(it.get("note") or "").strip()[:200] or "（无解读）",
            "surprise": 1 if it.get("surprise") else 0,
        })
    if not out:
        raise ValueError("返回中无任何可用的事件解读")
    return overall, out


def run_interpret(auto: bool = False) -> int:
    now = datetime.now(timezone.utc)
    today_et = str(now.astimezone(ET).date())
    conn = _connect()
    try:
        events = select_events(conn, now)
        if not events:
            print(f"[interpret] 无未解读的新事件，跳过（{today_et}）")
            return 0
        if auto:
            done_today = conn.execute(
                "SELECT 1 FROM llm_interpret_runs WHERE run_date=? AND status='ok'",
                (today_et,),
            ).fetchone()
            if done_today and not has_new_hi(events):
                print(f"[interpret] 今天已解读过且无新 T0/T1，跳过（{len(events)} 条 T2/T3 待明日）")
                return 0

        prompt = build_prompt(events)
        valid_ids = {e["event_id"] for e in events}
        print(f"[interpret] 调 claude 解读 {len(events)} 条新事件"
              f"（prompt {len(prompt)} 字符）…")
        try:
            parsed, meta = call_claude(prompt)
            overall, items = _validate(parsed, valid_ids)
        except (FileNotFoundError, RuntimeError, ValueError, OSError,
                json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            conn.execute(
                """INSERT INTO llm_interpret_runs
                   (run_date, status, reason, n_events, created_utc)
                   VALUES (?, 'absent', ?, 0, ?)""",
                (today_et, reason, store.utcnow_iso()),
            )
            conn.commit()
            print(f"[interpret] 降级：今日解读缺席 —— {reason}")
            return 0  # 降级不算失败，不影响采集链

        meta_json = json.dumps(meta, ensure_ascii=False)
        created = store.utcnow_iso()
        n_new = 0
        for it in items:
            cur = conn.execute(
                """INSERT OR IGNORE INTO llm_interpretations
                   (interp_date, event_id, direction, strength, note, surprise,
                    model_meta, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (today_et, it["event_id"], it["direction"], it["strength"],
                 it["note"], it["surprise"], meta_json, created),
            )
            n_new += cur.rowcount
        conn.execute(
            """INSERT INTO llm_interpret_runs
               (run_date, status, overall, n_events, model_meta, created_utc)
               VALUES (?, 'ok', ?, ?, ?, ?)""",
            (today_et, overall, n_new, meta_json, created),
        )
        conn.commit()
        cost = meta.get("total_cost_usd")
        print(f"[interpret] ok：{n_new} 条解读入库 · 态势「{overall}」"
              + (f" · cost≈${cost:.4f}" if isinstance(cost, (int, float)) else ""))
        return 0
    finally:
        conn.close()


def print_status() -> None:
    conn = _connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM llm_interpretations").fetchone()[0]
        print(f"llm_interpretations: {n} 条")
        for r in conn.execute(
            """SELECT interp_date, COUNT(*) c,
                      SUM(direction='bullish') b, SUM(direction='bearish') s,
                      SUM(surprise) sp
               FROM llm_interpretations GROUP BY interp_date
               ORDER BY interp_date DESC LIMIT 10"""
        ):
            print(f"  {r['interp_date']}: {r['c']} 条 "
                  f"(bull {r['b']} / bear {r['s']} / surprise {r['sp']})")
        print("runs（近 5 次）:")
        for r in conn.execute(
            "SELECT * FROM llm_interpret_runs ORDER BY run_id DESC LIMIT 5"
        ):
            print(f"  #{r['run_id']} {r['run_date']} {r['status']} "
                  f"n={r['n_events']} {r['reason'] or r['overall'] or ''}"[:120])
    finally:
        conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="立即跑一次（幂等）")
    p.add_argument("--auto", action="store_true",
                   help="链尾模式：今天已解读过且无新 T0/T1 则跳过")
    p.add_argument("--status", action="store_true", help="打印解读表统计")
    args = p.parse_args()
    if args.status:
        print_status()
        return 0
    if not (args.once or args.auto):
        print_status()
        return 0
    return run_interpret(auto=args.auto and not args.once)


if __name__ == "__main__":
    sys.exit(main())
