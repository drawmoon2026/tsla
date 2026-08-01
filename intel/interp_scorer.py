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
    交易日完整性：仅用已收盘的交易日判分（当日 16:05 ET 前不判当日）

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

from intel import store

ET = ZoneInfo("America/New_York")

OHLC_CACHE = store.DB_PATH.parent / "n8_ohlc_cache.json"
VERDICT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "n8_scoring"
VERDICT_PATH = VERDICT_DIR / "verdict.json"

VERDICT_MIN_N_1D = 60        # 1 日口径已判分样本达此数才跑预登记检验
SURPRISE_MIN_N = 5           # H2 检验意外组最少样本
ALPHA = 0.05                 # 预登记显著性水平
MARKET_OPEN = time(9, 30)    # ET
SESSION_DONE = time(16, 5)   # ET：此刻之后视当日线为收盘定稿

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


def load_ohlc(now: datetime) -> tuple[dict[date, tuple[float, float]], str | None]:
    """OHLC → ({date: (o,c)}, error)——共享 intel.prices 缓存（P1-4 限流治理）.

    取价统一走 data/intel/price_cache.json（5 分钟 TTL + 失败退避），与仪表盘
    价格上下文同源，不再各自打 yfinance。共享缓存拿不到时降级读本模块旧缓存
    （n8_ohlc_cache.json，只兜底不再写入 OHLC；该文件继续存 last_ok_utc）。
    """
    from intel import prices

    out, err = prices.get_ohlc(now)
    if not out:  # 共享缓存与 API 均空 → 旧缓存兜底
        for k, v in (_read_cache().get("daily") or {}).items():
            try:
                out[date.fromisoformat(k)] = (float(v[0]), float(v[1]))
            except (ValueError, TypeError, IndexError):
                continue
    return out, err


def _session_complete(d: date, now: datetime) -> bool:
    """交易日 d 的日线是否已收盘定稿（当日 16:05 ET 前不算）."""
    now_et = now.astimezone(ET)
    if d < now_et.date():
        return True
    return d == now_et.date() and now_et.time() >= SESSION_DONE


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
              ohlc: dict[date, tuple[float, float]], now: datetime) -> dict:
    """单条解读判分 → d1_date/ret_1d/ret_5d/score_1d/score_5d（未到期=pending）."""
    directional = direction in ("bullish", "bearish")
    pend = "pending" if directional else None
    out = {"d1_date": None, "ret_1d": None, "ret_5d": None,
           "score_1d": pend, "score_5d": pend}
    trade_dates = sorted(ohlc)
    i = _d1_index(anchor_utc, trade_dates)
    if i is None:
        return out  # D1 尚未出现在日线里（未来交易日）→ pending
    d1 = trade_dates[i]
    out["d1_date"] = str(d1)
    if _session_complete(d1, now):
        o1, c1 = ohlc[d1]
        ret1 = c1 / o1 - 1.0
        out["ret_1d"] = ret1
        if directional:
            out["score_1d"] = _judge(direction, ret1)
        if i + 4 < len(trade_dates) and _session_complete(trade_dates[i + 4], now):
            c5 = ohlc[trade_dates[i + 4]][1]
            ret5 = c5 / o1 - 1.0
            out["ret_5d"] = ret5
            if directional:
                out["score_5d"] = _judge(direction, ret5)
    return out


def run_score(conn: sqlite3.Connection, now: datetime) -> dict:
    """对所有需要（新增或未了结）的解读判分，落 interp_scores，返回统计."""
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
             "n_updated": 0, "ohlc_error": None}
    if not todo:
        return stats
    ohlc, err = load_ohlc(now)
    stats["ohlc_error"] = err
    if not ohlc:
        return stats  # 无任何价格数据：保持现状（pending），不硬判
    ts = store.utcnow_iso()
    for it in todo:
        s = score_one(it["direction"], it["created_utc"], ohlc, now)
        conn.execute(
            """INSERT INTO interp_scores
               (event_id, interp_date, direction, strength, surprise, anchor_utc,
                d1_date, ret_1d, ret_5d, score_1d, score_5d, scored_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id) DO UPDATE SET
                 d1_date=excluded.d1_date, ret_1d=excluded.ret_1d,
                 ret_5d=excluded.ret_5d, score_1d=excluded.score_1d,
                 score_5d=excluded.score_5d, scored_utc=excluded.scored_utc""",
            (it["event_id"], it["interp_date"], it["direction"], it["strength"],
             it["surprise"], it["created_utc"], s["d1_date"], s["ret_1d"],
             s["ret_5d"], s["score_1d"], s["score_5d"], ts),
        )
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
    """最近一个「工作日 16:05 ET」时刻（busday 近似，不剔假日——假日多跑一次无害）."""
    et = now.astimezone(ET)
    d = et.date()
    while True:
        if d.weekday() < 5:
            t = datetime.combine(d, SESSION_DONE, tzinfo=ET)
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
        _record_run(conn, "ok",
                    stats.get("ohlc_error") and f"取价降级：{stats['ohlc_error']}",
                    stats)
        cache = _read_cache()
        cache["last_ok_utc"] = now.isoformat(timespec="seconds")
        _write_cache(cache)
        c1 = verdict["sample"]["score_1d"]
        print(f"[interp_scorer] 判分 {stats['n_updated']}/{stats['n_todo']} 条"
              f"（解读共 {stats['n_interps']}）· 1 日口径 "
              f"hit {c1['hit']} / miss {c1['miss']} / pending {c1['pending']}"
              f" · 状态 {verdict['status']} → {VERDICT_PATH}"
              + (f" · 取价降级：{stats['ohlc_error']}" if stats["ohlc_error"] else ""))
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true", help="立即判一轮（幂等）")
    p.add_argument("--auto", action="store_true", help="链尾模式：无事速退")
    p.add_argument("--status", action="store_true", help="打印判分表统计")
    args = p.parse_args()
    if args.status or not (args.once or args.auto):
        print_status()
        return 0
    return run(auto=args.auto and not args.once)


if __name__ == "__main__":
    sys.exit(main())
