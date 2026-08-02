"""N8 —— LLM 解读前向判分器（挂哨兵链尾，逐日运行）.

框架依据 docs/strategy-lab.md N8 条目（预登记假设与判分口径，冻结于建设日）：

  预登记假设（2026-08-02 冻结，样本够时自动裁决）：
    H1  LLM 对事件的方向解读（bull/bear）与事件后 1 日 / 5 日 TSLA 收益方向
        一致率 > 50%（对照 = 方向随机）
    H2  意外标记事件的次日绝对波动 > 非意外事件

  判分口径（冻结，严格照 N8 条目）：
    锚点     = 解读产生时刻（llm_interpretations.created_utc）——信息可用的
               最早时刻，防前视
    D1       = 锚点之后首个「开盘时刻(09:30 ET) ≥ 锚点」的交易日
               （盘中/盘后产生的解读顺延到下一交易日，不吃当日已走完的行情）
    ret_1d   = close(D1)/open(D1) − 1
    ret_5d   = close(D5)/open(D1) − 1，D5 = D1 起第 5 个交易日（D1 后第 4 个）
    判分     = bullish 且 ret>0 → hit；bearish 且 ret<0 → hit；否则 miss
               （ret=0 平局判 miss，保守）；中性解读不计分（score 置 NULL）
    收益主口径 = TSLA 绝对收益方向；池等权超额为可选口径，v1 未启用（如实记录）
    交易日完整性：仅用已收盘的交易日判分（当日收盘+5min 前不判当日）；且价格
               数据必须抓取于该日收盘定稿之后（fetched_utc ≥ close+5min，
               P1-1 收盘价把关）——宁可晚判（pending）不可错判
    判分不可改写（P0-1）：已判定的 score/ret/D1 锚一经写入即冻结，永不覆盖；
               重算不一致落 interp_score_conflicts 告警行；后续补 5 日分
               沿用存储的 D1 锚，不重算

  自动裁决（「在此之前只积累不判读」）：
    每次运行统计样本量；1 日口径已判分样本 n ≥ 60 时才跑预登记检验：
      H1 一致率 vs 50% 单侧二项检验（1 日 + 5 日两口径）
      H2 意外 vs 非意外事件 |ret_1d| 单侧 Mann-Whitney U
    结果（或积累进度）写 outputs/n8_scoring/verdict.json。

存储：interp_scores 表（event_id 主键；中性行也入库存实际收益，供 H2 波动
检验用，但 score_1d/5d 为 NULL 不计分）。价格走 intel.prices 共享缓存
（data/intel/price_cache.json，5 分钟 TTL + 失败退避，P1-4 限流治理），
取数失败降级读缓存，仍缺则保持 pending 不硬判。每次运行记 interp_scorer_runs
（ok / absent+reason，P1-1）——崩溃不再静默。

用法：
    .venv/bin/python -m intel.interp_scorer --once    # 立即判一轮（幂等）
    .venv/bin/python -m intel.interp_scorer --auto    # 链尾模式（无事速退）
    .venv/bin/python -m intel.interp_scorer --status  # 打印判分表统计
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from intel import market_calendar as mcal  # 统一 NYSE 交易日历（口径修正 2026-08-02）
from intel import store

ET = ZoneInfo("America/New_York")

OHLC_CACHE = store.DB_PATH.parent / "n8_ohlc_cache.json"
VERDICT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "n8_scoring"
VERDICT_PATH = VERDICT_DIR / "verdict.json"

VERDICT_MIN_N_1D = 60        # 1 日口径已判分样本达此数才跑预登记检验
SURPRISE_MIN_N = 5           # H2 检验意外组最少样本
ALPHA = 0.05                 # 预登记显著性水平
MARKET_OPEN = time(9, 30)    # ET
RET_TOL = 1e-9               # P0-1：重算收益与已存值的浮点一致性容差

_SCHEMA = """
-- interp_scores：N8 前向判分（口径冻结见 intel/interp_scorer.py docstring）。
-- 中性解读不计分（score 为 NULL），但仍存实际收益供意外-波动检验（H2）。
CREATE TABLE IF NOT EXISTS interp_scores (
    event_id    TEXT PRIMARY KEY REFERENCES llm_interpretations(event_id),
    interp_date TEXT NOT NULL,
    direction   TEXT NOT NULL,
    strength    INTEGER,
    surprise    INTEGER,
    anchor_utc  TEXT NOT NULL,         -- 解读产生时刻（created_utc，判分锚点）
    d1_date     TEXT,                  -- 锚点后首个开盘≥锚点的交易日（未到=NULL）
    ret_1d      REAL,                  -- close(D1)/open(D1)-1
    ret_5d      REAL,                  -- close(D5)/open(D1)-1
    score_1d    TEXT CHECK (score_1d IN ('hit','miss','pending')),
    score_5d    TEXT CHECK (score_5d IN ('hit','miss','pending')),
    scored_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interp_scores_date ON interp_scores (interp_date);

-- interp_scorer_runs：判分器运行记录面（P1-1）——崩溃不再静默，
-- 仿 interpret 的 runs 口径：ok / absent(+reason)；仪表盘记分牌挂数据龄。
CREATE TABLE IF NOT EXISTS interp_scorer_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_utc    TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('ok','absent')),
    reason     TEXT,
    n_updated  INTEGER,
    n_todo     INTEGER
);

-- interp_score_conflicts：P0-1 判分不可改写——重算值与已存已判值不一致时
-- 落告警行（stored 保持不动，绝不覆盖），供审计缓存切换/数据源事后修正。
CREATE TABLE IF NOT EXISTS interp_score_conflicts (
    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    field       TEXT NOT NULL,
    stored      TEXT,
    recomputed  TEXT,
    noted_utc   TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = store.connect()
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------- OHLC

def _read_cache() -> dict:
    try:
        return json.loads(OHLC_CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_cache(d: dict) -> None:
    try:
        OHLC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        OHLC_CACHE.write_text(json.dumps(d, separators=(",", ":")),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _parse_utc(iso: str | None) -> datetime | None:
    try:
        t = datetime.fromisoformat(iso or "")
    except ValueError:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def load_ohlc(now: datetime
              ) -> tuple[dict[date, tuple[float, float]], str | None,
                         datetime | None]:
    """OHLC → ({date: (o,c)}, error, fetched_utc)——共享 intel.prices 缓存.

    取价统一走 data/intel/price_cache.json（5 分钟 TTL + 失败退避），与仪表盘
    价格上下文同源，不再各自打 yfinance。共享缓存拿不到时降级读本模块旧缓存
    （n8_ohlc_cache.json，只兜底不再写入 OHLC；该文件继续存 last_ok_utc）。

    fetched_utc = 这份数据实际抓取时刻（P1-1 收盘价把关用）；旧缓存兜底段
    没有可信抓取时刻 → None（其数据只能保持 pending，不用于定稿判分）。
    """
    from intel import prices

    out, err, fetched_iso = prices.get_ohlc(now)
    fetched = _parse_utc(fetched_iso)
    if not out:  # 共享缓存与 API 均空 → 旧缓存兜底（无抓取时刻，不可定稿判分）
        fetched = None
        for k, v in (_read_cache().get("daily") or {}).items():
            try:
                out[date.fromisoformat(k)] = (float(v[0]), float(v[1]))
            except (ValueError, TypeError, IndexError):
                continue
    return out, err, fetched


def _session_done_dt(d: date) -> datetime:
    """交易日 d 的日线定稿时刻（ET aware）= 日历收盘 + 5 分钟（半日市 13:05）.

    combine+timedelta 写法（P2-1）：非整点收盘也不会分钟溢出。
    """
    return (datetime.combine(d, mcal.close_time_et(d), tzinfo=ET)
            + timedelta(minutes=5))


def _session_complete(d: date, now: datetime, fetched: datetime | None) -> bool:
    """交易日 d 的日线是否可用于定稿判分（P1-1 收盘价把关）.

    两个条件缺一不可：
      1. 现在已过 d 的日历收盘 +5min（半日市 13:05）；
      2. 这份价格数据抓取于 d 的收盘定稿之后（fetched >= close+5min）——
         否则 d 行的 close 可能是盘中残值/陈旧缓存，宁可晚判（pending）不可错判。
    fetched 未知（旧缓存兜底）一律不定稿。
    """
    done = _session_done_dt(d)
    if now.astimezone(ET) < done:
        return False
    if fetched is None:
        return False
    return fetched >= done


# ---------------------------------------------------------------- 判分

def _d1_index(anchor_utc: str, trade_dates: list[date]) -> int | None:
    """锚点后首个「开盘 ≥ 锚点」交易日在 trade_dates 中的下标；未到返回 None."""
    try:
        anchor = datetime.fromisoformat(anchor_utc)
    except ValueError:
        return None
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    anchor_et = anchor.astimezone(ET)
    for i, d in enumerate(trade_dates):
        if datetime.combine(d, MARKET_OPEN, tzinfo=ET) >= anchor_et:
            return i
    return None


def _judge(direction: str, ret: float) -> str:
    """方向判分（冻结）：严格同号为 hit，ret=0 平局判 miss（保守）."""
    if direction == "bullish":
        return "hit" if ret > 0 else "miss"
    return "hit" if ret < 0 else "miss"


def score_one(direction: str, anchor_utc: str,
              ohlc: dict[date, tuple[float, float]], now: datetime,
              fetched: datetime | None = None,
              frozen_d1: str | None = None) -> dict:
    """单条解读判分 → d1_date/ret_1d/ret_5d/score_1d/score_5d（未到期=pending）.

    frozen_d1（P0-1）：D1 锚一经写入即冻结——已存 d1_date 的行沿用存储的锚，
    不重算 _d1_index（缓存切换/键集漂移不换锚）；冻结锚不在当前日线里时
    本轮不判（返回全 pending，字段留待下轮）。
    fetched（P1-1）：价格数据抓取时刻，_session_complete 收盘价把关用。
    """
    directional = direction in ("bullish", "bearish")
    pend = "pending" if directional else None
    out = {"d1_date": None, "ret_1d": None, "ret_5d": None,
           "score_1d": pend, "score_5d": pend}
    trade_dates = sorted(ohlc)
    if frozen_d1:
        try:
            d1 = date.fromisoformat(frozen_d1)
        except ValueError:
            return out
        out["d1_date"] = frozen_d1
        if d1 not in ohlc:
            return out  # 冻结锚不在当前日线（缓存切换）→ 本轮不判，不换锚
        i = trade_dates.index(d1)
    else:
        i = _d1_index(anchor_utc, trade_dates)
        if i is None:
            return out  # D1 尚未出现在日线里（未来交易日）→ pending
        d1 = trade_dates[i]
        out["d1_date"] = str(d1)
    if _session_complete(d1, now, fetched):
        o1, c1 = ohlc[d1]
        ret1 = c1 / o1 - 1.0
        out["ret_1d"] = ret1
        if directional:
            out["score_1d"] = _judge(direction, ret1)
        if (i + 4 < len(trade_dates)
                and _session_complete(trade_dates[i + 4], now, fetched)):
            c5 = ohlc[trade_dates[i + 4]][1]
            ret5 = c5 / o1 - 1.0
            out["ret_5d"] = ret5
            if directional:
                out["score_5d"] = _judge(direction, ret5)
    return out


_FINAL_SCORES = ("hit", "miss")


def _merge_score(ex: dict, s: dict) -> tuple[dict, list[tuple[str, str, str]]]:
    """P0-1 合并：已判定字段只写一次，返回 (要更新的字段, 冲突清单).

    - d1_date / ret_1d / ret_5d：已存非 NULL 即冻结，只补 NULL；
    - score_1d / score_5d：hit/miss 即冻结；pending 可推进为 hit/miss；
      NULL（中性不计分）永远保持 NULL；
    - 重算值与已存冻结值不一致 → 记冲突（告警行），存值保持不动。
    """
    updates: dict = {}
    conflicts: list[tuple[str, str, str]] = []

    for f in ("d1_date", "ret_1d", "ret_5d"):
        new = s[f]
        old = ex.get(f)
        if old is None:
            if new is not None:
                updates[f] = new
        elif new is not None:
            same = (abs(new - old) <= RET_TOL if f != "d1_date"
                    else str(new) == str(old))
            if not same:
                conflicts.append((f, str(old), str(new)))

    for f in ("score_1d", "score_5d"):
        new = s[f]
        old = ex.get(f)
        if old in _FINAL_SCORES:
            if new in _FINAL_SCORES and new != old:
                conflicts.append((f, str(old), str(new)))
        elif old == "pending" and new in _FINAL_SCORES:
            updates[f] = new
        # old is NULL（中性）→ 保持 NULL，不写
    return updates, conflicts


def run_score(conn: sqlite3.Connection, now: datetime,
              ohlc_override: tuple | None = None) -> dict:
    """对所有需要（新增或未了结）的解读判分，落 interp_scores，返回统计.

    P0-1 判分不可改写：已判定的 score/ret/D1 锚永不覆盖——UPDATE 只补 NULL
    （或推进 pending）字段；重算值与已存值不一致时写 interp_score_conflicts
    告警行而非覆盖；scored_utc 仅在实际补写了字段时刷新。
    ohlc_override=(ohlc, err, fetched)：单测注入合成价格数据用。
    """
    interps = [dict(r) for r in conn.execute(
        """SELECT i.event_id, i.interp_date, i.direction, i.strength,
                  i.surprise, i.created_utc
           FROM llm_interpretations i"""
    )]
    existing = {r["event_id"]: dict(r)
                for r in conn.execute("SELECT * FROM interp_scores")}
    todo = [it for it in interps
            if it["event_id"] not in existing
            or existing[it["event_id"]]["ret_1d"] is None
            or existing[it["event_id"]]["ret_5d"] is None]
    stats = {"n_interps": len(interps), "n_todo": len(todo),
             "n_updated": 0, "n_conflicts": 0, "ohlc_error": None}
    if not todo:
        return stats
    ohlc, err, fetched = (ohlc_override if ohlc_override is not None
                          else load_ohlc(now))
    stats["ohlc_error"] = err
    if not ohlc:
        return stats  # 无任何价格数据：保持现状（pending），不硬判
    ts = store.utcnow_iso()
    for it in todo:
        ex = existing.get(it["event_id"])
        s = score_one(it["direction"], it["created_utc"], ohlc, now, fetched,
                      frozen_d1=(ex or {}).get("d1_date"))
        if ex is None:
            conn.execute(
                """INSERT INTO interp_scores
                   (event_id, interp_date, direction, strength, surprise,
                    anchor_utc, d1_date, ret_1d, ret_5d, score_1d, score_5d,
                    scored_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(event_id) DO NOTHING""",
                (it["event_id"], it["interp_date"], it["direction"],
                 it["strength"], it["surprise"], it["created_utc"],
                 s["d1_date"], s["ret_1d"], s["ret_5d"], s["score_1d"],
                 s["score_5d"], ts),
            )
            stats["n_updated"] += 1
            continue
        updates, conflicts = _merge_score(ex, s)
        for field, old, new in conflicts:
            conn.execute(
                """INSERT INTO interp_score_conflicts
                   (event_id, field, stored, recomputed, noted_utc)
                   VALUES (?, ?, ?, ?, ?)""",
                (it["event_id"], field, old, new, ts))
            stats["n_conflicts"] += 1
        if updates:
            sets = ", ".join(f"{f}=?" for f in updates) + ", scored_utc=?"
            conn.execute(
                f"UPDATE interp_scores SET {sets} WHERE event_id=?",  # noqa: S608
                (*updates.values(), ts, it["event_id"]))
            stats["n_updated"] += 1
    conn.commit()
    return stats


# ---------------------------------------------------------------- 自动裁决

_SPEC = {
    "registered": "docs/strategy-lab.md N8（冻结于建设日 2026-08-02）",
    "h1": "方向解读与事件后 1 日/5 日 TSLA 收益方向一致率 > 50%（单侧二项检验）",
    "h2": "意外标记事件的次日绝对波动 > 非意外事件（单侧 Mann-Whitney U）",
    "anchor": "解读产生时刻（created_utc）之后首个开盘≥锚点的交易日，开盘起算",
    "return_metric": "TSLA 绝对收益方向（主口径）；池等权超额为可选口径，未启用",
    "neutral": "中性解读不计分；平局（ret=0）判 miss（保守）",
    "min_n_1d": VERDICT_MIN_N_1D,
    "alpha": ALPHA,
}


def _binom_greater(hits: int, n: int) -> float:
    """单侧二项检验 P(X >= hits | p=0.5)（scipy 缺席时用精确求和）."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(hits, n, 0.5, alternative="greater").pvalue)
    except ImportError:
        from math import comb
        return sum(comb(n, k) for k in range(hits, n + 1)) / 2.0 ** n


def build_verdict(conn: sqlite3.Connection, now: datetime) -> dict:
    """样本量统计 + （n 够时）预登记检验；n 不够时只积累不判读."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM interp_scores")]
    dir_rows = [r for r in rows if r["direction"] in ("bullish", "bearish")]

    def _cnt(key: str) -> dict:
        c = {"hit": 0, "miss": 0, "pending": 0}
        for r in dir_rows:
            if r[key] in c:
                c[r[key]] += 1
        return c

    c1, c5 = _cnt("score_1d"), _cnt("score_5d")
    sur_ret = [abs(r["ret_1d"]) for r in rows
               if r["surprise"] and r["ret_1d"] is not None]
    non_ret = [abs(r["ret_1d"]) for r in rows
               if not r["surprise"] and r["ret_1d"] is not None]
    n1 = c1["hit"] + c1["miss"]
    out = {
        "generated_utc": now.isoformat(timespec="seconds"),
        "spec": _SPEC,
        "sample": {
            "n_interps_total": conn.execute(
                "SELECT COUNT(*) FROM llm_interpretations").fetchone()[0],
            "n_directional": len(dir_rows),
            "score_1d": c1, "score_5d": c5,
            "n_surprise_with_ret1d": len(sur_ret),
            "n_nonsurprise_with_ret1d": len(non_ret),
        },
        "status": "accumulating",
        "verdict": None,
    }
    if n1 < VERDICT_MIN_N_1D:
        out["note"] = (f"1 日口径已判分 {n1}/{VERDICT_MIN_N_1D}——样本积累中，"
                       "只积累不判读（预登记纪律）")
        return out

    # ---- n 够：跑预登记检验
    out["status"] = "verdict"
    p1 = _binom_greater(c1["hit"], n1)
    v = {"h1_1d": {"n": n1, "hits": c1["hit"],
                   "hit_rate": round(c1["hit"] / n1, 4),
                   "p_binom_one_sided": round(p1, 6),
                   "pass": bool(p1 < ALPHA)}}
    n5 = c5["hit"] + c5["miss"]
    if n5 >= VERDICT_MIN_N_1D:
        p5 = _binom_greater(c5["hit"], n5)
        v["h1_5d"] = {"n": n5, "hits": c5["hit"],
                      "hit_rate": round(c5["hit"] / n5, 4),
                      "p_binom_one_sided": round(p5, 6),
                      "pass": bool(p5 < ALPHA)}
    else:
        v["h1_5d"] = {"n": n5, "note": "5 日口径样本未达门槛，暂不判"}
    if len(sur_ret) >= SURPRISE_MIN_N and len(non_ret) >= SURPRISE_MIN_N:
        try:
            from scipy.stats import mannwhitneyu
            u = mannwhitneyu(sur_ret, non_ret, alternative="greater")
            v["h2_surprise_vol"] = {
                "n_surprise": len(sur_ret), "n_non": len(non_ret),
                "median_abs_ret1d_surprise": round(
                    sorted(sur_ret)[len(sur_ret) // 2], 5),
                "median_abs_ret1d_non": round(
                    sorted(non_ret)[len(non_ret) // 2], 5),
                "p_mwu_one_sided": round(float(u.pvalue), 6),
                "pass": bool(u.pvalue < ALPHA),
            }
        except ImportError:
            v["h2_surprise_vol"] = {"note": "scipy 缺席，H2 未检验"}
    else:
        v["h2_surprise_vol"] = {"n_surprise": len(sur_ret),
                                "note": f"意外组样本 < {SURPRISE_MIN_N}，暂不判"}
    out["verdict"] = v
    return out


def write_verdict(verdict: dict) -> None:
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- auto 节流

def _last_close_utc(now: datetime) -> datetime:
    """最近一个「交易日收盘定稿」时刻（统一 NYSE 日历，半日市 13:05）.

    原口径：工作日固定 16:05 ET（busday 近似，不剔假日——假日多跑一次无害）。
    """
    et = now.astimezone(ET)
    d = et.date()
    while True:
        if mcal.is_trading_day(d):   # 原：d.weekday() < 5
            t = _session_done_dt(d)
            if t <= et:
                return t.astimezone(timezone.utc)
        d -= timedelta(days=1)


def should_run_auto(conn: sqlite3.Connection, now: datetime) -> bool:
    """有新解读未入判分表 → 跑；有未了结行且上次成功在最近收盘前 → 跑；否则跳."""
    new = conn.execute(
        """SELECT 1 FROM llm_interpretations
           WHERE event_id NOT IN (SELECT event_id FROM interp_scores) LIMIT 1"""
    ).fetchone()
    if new:
        return True
    unresolved = conn.execute(
        "SELECT 1 FROM interp_scores WHERE ret_1d IS NULL OR ret_5d IS NULL LIMIT 1"
    ).fetchone()
    if not unresolved:
        return False
    try:
        last_ok = datetime.fromisoformat(_read_cache().get("last_ok_utc", ""))
        if last_ok.tzinfo is None:
            last_ok = last_ok.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return last_ok < _last_close_utc(now)


# ---------------------------------------------------------------- 入口

def _record_run(conn: sqlite3.Connection, status: str, reason: str | None,
                stats: dict | None) -> None:
    """interp_scorer_runs 落一行（P1-1）；记录失败本身只打印，不再抛."""
    try:
        conn.execute(
            """INSERT INTO interp_scorer_runs (run_utc, status, reason,
                                               n_updated, n_todo)
               VALUES (?, ?, ?, ?, ?)""",
            (store.utcnow_iso(), status, reason,
             (stats or {}).get("n_updated"), (stats or {}).get("n_todo")))
        conn.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[interp_scorer] runs 记录失败：{type(e).__name__}: {e}")


def run(auto: bool = False) -> int:
    now = datetime.now(timezone.utc)
    conn = _connect()
    stats: dict | None = None
    try:
        if auto and not should_run_auto(conn, now):
            return 0  # 静默速退：无新解读且无可了结的 pending
        try:
            stats = run_score(conn, now)
            verdict = build_verdict(conn, now)
            write_verdict(verdict)
        except Exception as e:  # noqa: BLE001 —— P1-1：崩溃记 absent，不静默
            _record_run(conn, "absent", f"{type(e).__name__}: {e}", stats)
            raise
        notes = []
        if stats.get("ohlc_error"):
            notes.append(f"取价降级：{stats['ohlc_error']}")
        if stats.get("n_conflicts"):
            notes.append(f"冻结冲突 {stats['n_conflicts']} 条"
                         "（重算≠已存，未覆盖，见 interp_score_conflicts）")
        _record_run(conn, "ok", " · ".join(notes) or None, stats)
        cache = _read_cache()
        cache["last_ok_utc"] = now.isoformat(timespec="seconds")
        _write_cache(cache)
        c1 = verdict["sample"]["score_1d"]
        print(f"[interp_scorer] 判分 {stats['n_updated']}/{stats['n_todo']} 条"
              f"（解读共 {stats['n_interps']}）· 1 日口径 "
              f"hit {c1['hit']} / miss {c1['miss']} / pending {c1['pending']}"
              f" · 状态 {verdict['status']} → {VERDICT_PATH}"
              + (f" · {' · '.join(notes)}" if notes else ""))
        return 0
    finally:
        conn.close()


def print_status() -> None:
    conn = _connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM interp_scores").fetchone()[0]
        print(f"interp_scores: {n} 条")
        for r in conn.execute(
            """SELECT interp_date, direction, score_1d, score_5d, d1_date,
                      ROUND(ret_1d*100,2) r1, ROUND(ret_5d*100,2) r5, surprise
               FROM interp_scores WHERE direction != 'neutral'
               ORDER BY interp_date DESC, event_id LIMIT 30"""
        ):
            print(f"  {r['interp_date']} {r['direction']:<8} D1={r['d1_date'] or '—'} "
                  f"1d={r['score_1d']}({r['r1']}%) 5d={r['score_5d']}({r['r5']}%)"
                  + (" 意外" if r["surprise"] else ""))
        if VERDICT_PATH.exists():
            v = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
            print(f"verdict: status={v['status']} · {v.get('note', '')}")
    finally:
        conn.close()


# ---------------------------------------------------------------- 自测

def _selftest() -> None:
    """P0-1/P1-1 合成场景单测（不碰真实 DB / 不打网络）：
    python -m intel.interp_scorer --selftest
    """
    d5, d6, d7, d8, d9 = (date(2026, 1, i) for i in (5, 6, 7, 8, 9))  # 周一~五

    def _et(d: date, h: int, m: int = 0) -> datetime:
        return datetime.combine(d, time(h, m), tzinfo=ET)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE llm_interpretations (interp_date TEXT, event_id TEXT "
        "PRIMARY KEY, direction TEXT, strength INTEGER, surprise INTEGER, "
        "created_utc TEXT);" + _SCHEMA.replace("REFERENCES llm_interpretations(event_id)", ""))
    anchor = _et(d5, 7).astimezone(timezone.utc).isoformat()  # D1 = 01-05
    conn.execute("INSERT INTO llm_interpretations VALUES ('2026-01-05','ev1',"
                 "'bullish',2,0,?)", (anchor,))
    conn.execute("INSERT INTO llm_interpretations VALUES ('2026-01-05','ev2',"
                 "'neutral',1,0,?)", (anchor,))
    ohlc_a = {d5: (100.0, 110.0), d6: (111.0, 112.0), d7: (112.0, 113.0),
              d8: (113.0, 114.0), d9: (114.0, 120.0)}

    def _row(eid: str = "ev1") -> dict:
        return dict(conn.execute(
            "SELECT * FROM interp_scores WHERE event_id=?", (eid,)).fetchone())

    # 1) P1-1 stale 价拒判：收盘后判分但数据抓于收盘前 → 保持 pending
    run_score(conn, _et(d5, 17), (ohlc_a, None, _et(d5, 15)))
    r = _row()
    assert r["score_1d"] == "pending" and r["ret_1d"] is None, r
    # 2) 数据抓于收盘定稿后 → 判 hit，D1 锚落定
    run_score(conn, _et(d5, 17), (ohlc_a, None, _et(d5, 16, 10)))
    r = _row()
    assert (r["score_1d"], r["d1_date"]) == ("hit", "2026-01-05"), r
    assert abs(r["ret_1d"] - 0.10) < 1e-12, r
    frozen = (r["ret_1d"], r["score_1d"], r["d1_date"], r["scored_utc"])
    # 3) P0-1 价格事后修正（ret 翻面）→ 已判行不变，落冲突告警行
    ohlc_fix = dict(ohlc_a); ohlc_fix[d5] = (100.0, 90.0)
    st = run_score(conn, _et(d6, 17), (ohlc_fix, None, _et(d6, 16, 10)))
    r = _row()
    assert (r["ret_1d"], r["score_1d"], r["d1_date"]) == frozen[:3], r
    ncf_ev1 = conn.execute("SELECT COUNT(*) FROM interp_score_conflicts "
                           "WHERE event_id='ev1'").fetchone()[0]
    assert ncf_ev1 == 2, ncf_ev1  # ret_1d + score_1d 双冲突（ev2 另记 ret 冲突）
    total_cf = conn.execute(
        "SELECT COUNT(*) FROM interp_score_conflicts").fetchone()[0]
    assert total_cf == st["n_conflicts"], (total_cf, st)
    # 4) P0-1 缓存切换键集漂移（D1 从当前日线消失）→ 不换锚、不改行
    ohlc_b = {k: v for k, v in ohlc_a.items() if k != d5}
    run_score(conn, _et(d7, 17), (ohlc_b, None, _et(d7, 16, 10)))
    r = _row()
    assert (r["ret_1d"], r["score_1d"], r["d1_date"]) == frozen[:3], r
    # 5) 冻结锚沿用补 5 日分：ret_5d 按存储 D1 开盘计，1 日字段保持原值
    st = run_score(conn, _et(d9, 17), (ohlc_a, None, _et(d9, 16, 10)))
    r = _row()
    assert (r["ret_1d"], r["score_1d"], r["d1_date"]) == frozen[:3], r
    assert abs(r["ret_5d"] - 0.20) < 1e-12 and r["score_5d"] == "hit", r
    assert r["scored_utc"] is not None
    # 6) 中性行：ret 填充但 score 恒 NULL（不计分）
    r2 = _row("ev2")
    assert r2["score_1d"] is None and r2["score_5d"] is None, r2
    assert r2["ret_1d"] is not None and r2["ret_5d"] is not None, r2
    # 7) score_one 冻结锚优先于重算：frozen_d1 指定 01-06 则用 01-06 行情
    s = score_one("bullish", anchor, ohlc_a, _et(d9, 17), _et(d9, 16, 10),
                  frozen_d1="2026-01-06")
    assert s["d1_date"] == "2026-01-06" and abs(s["ret_1d"] - 1.0 / 111.0) < 1e-12
    conn.close()
    print("interp_scorer selftest: OK（stale 价拒判 / 已判不可变+冲突告警 / "
          "缓存切换不换锚 / 冻结锚补 5 日 / 中性不计分）")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="立即判一轮（幂等）")
    p.add_argument("--auto", action="store_true", help="链尾模式：无事速退")
    p.add_argument("--status", action="store_true", help="打印判分表统计")
    p.add_argument("--selftest", action="store_true",
                   help="P0-1/P1-1 合成场景单测（不碰真实 DB）")
    args = p.parse_args()
    if args.selftest:
        _selftest()
        return 0
    if args.status or not (args.once or args.auto):
        print_status()
        return 0
    return run(auto=args.auto and not args.once)


if __name__ == "__main__":
    sys.exit(main())
