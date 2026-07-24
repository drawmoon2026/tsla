"""因果探测器 · 哨兵 — 静态 HTML 仪表盘生成器.

读 data/intel/sentinel.sqlite（只读，mode=ro），渲染完全自包含的
data/intel/dashboard.html：内联全部 CSS/JS，无外部依赖，浏览器直接打开。
视觉语言合并自设计稿 data/intel/dashboard_design.html（v1.0，存档保留）：
暗色默认 + 亮色（尊重 prefers-color-scheme，页内可切换）、宋体衬线板块题
+ 等宽序号、标定环表盘 + 三灯信号组、军事地图旗标、T0-T3 色深+形状双编码。

板块（等宽序号按实际渲染顺序编排）：
  顶栏：TSLA 现价与最近涨跌 / 生成时刻 / 最后事件入库时刻 / 最后轮询时刻
     （>30 分钟标红）+ 主题切换；<meta refresh> 每 5 分钟自动重载
  01 今日合议（每日决策卡：现价 · S2 开关读数（距 252 日高回撤，E11 冻结口径）·
     探测器状态与标定倒计时 · 策略线 shadow 健康 · 规则合成的综合一句话）
  02 态势总览（detector_state：标定环表盘（标定倒计时）+ 态势陈述 +
     两腿读数卡（含数据龄徽章）+ 证据等级行 + 三灯信号组 + 假想单判分；
     表缺失整面板隐藏）
  03 战场走势（标的视图：TSLA 日线收盘（yfinance 增量补到最新，失败降级
     + STALE 徽章）对数坐标折线 + 1/3/8 年时间刷 + 可开关图层——图例分
     「研究回放（事后）」区（避险影线带 / 真坑 / 假坑 / Musk 数据失明期灰底）
     与「前向/事实（当时可知）」区（Musk 菱旗 / 假想单十字准星）；
     坑判据写进 tooltip 与明细表。页面预留多标的标签栏。）
  04 渠道健康（按 T0-T3 分组的渠道卡片 + 衍生信号单列；权重标注人工先验）
  05 最新情报流（最近 50 条事件时间线，行首带层级徽章）
  06 计数与时延（层级双条计数（ET 日口径）+ 每渠道 p50→p90 稳态口径标尺）

容错：任一表/视图/CSV 缺失则跳过或置灰对应板块（图层），不炸。

用法：
    .venv/bin/python -m intel.dashboard              # 生成一次
    .venv/bin/python -m intel.dashboard --out x.html # 自定义输出路径

launchd 定时重生成模板见 intel/deploy/com.tsla.dashboard.plist（默认不 load）。
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import math
import sqlite3
from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from intel.store import DB_PATH

try:  # 价格上下文（yfinance 增量 + S2 读数）；导入失败降级为"取价失败"
    from intel.prices import get_price_context
except Exception:  # noqa: BLE001
    get_price_context = None  # type: ignore[assignment]

ET = ZoneInfo("America/New_York")

try:  # 与探测器冻结参数保持同源；导入失败退回写死值（容错，不炸仪表盘）
    from intel.detector import (
        CALIB_BDAYS, COST_LINE, LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT,
    )
except Exception:  # noqa: BLE001
    CALIB_BDAYS, COST_LINE = 20, -0.0006
    LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT = 20, 20, 10.0

OUT_PATH = DB_PATH.parent / "dashboard.html"

STALE_S = 30 * 60  # 最后轮询距今超过此秒数 → 顶栏标红

# ---- 走势与历史判断（标的视图）数据源 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BARS_CSV = PROJECT_ROOT / "data" / "TSLA_1h_alpaca.csv"
STATES_CSV = PROJECT_ROOT / "outputs" / "n3h_deduction" / "daily_states.csv"
PITS_CSV = PROJECT_ROOT / "outputs" / "n4_golden_pit" / "pits_catalog.csv"
FORM4_CSV = PROJECT_ROOT / "data" / "intel" / "edgar_form4.csv"

# 视图盒尺寸（viewBox 单位；渲染时宽度 100% 自适应）
_VB_W, _VB_H = 1040, 400
_ML, _MR, _MT, _MB = 50, 14, 14, 30
_EPOCH_ORD = date(1970, 1, 1).toordinal()  # JS 侧日期还原用

TIERS = ["T0", "T1", "T2", "T3"]

TIER_LABEL = {
    "T0": "T0 布局痕迹",
    "T1": "T1 法定披露",
    "T2": "T2 官方承诺",
    "T3": "T3 放风叙事",
}

# 探测器状态 → (语义色类, 中文短语, 一句话说明)。语义色，不用层级色。
# RISK_ON 措辞刻意去承诺化（P0-6）：探测器只覆盖"空头知情型下跌"一类，
# 不能说"正常持仓"——盲区声明常驻面板。
DET_STATE = {
    "CALIBRATING": ("warn", "标定中", "累积 nitter 口径基线，不出信号"),
    "RISK_ON": ("good", "未见目标风险",
                "未见空头知情型风险（仅覆盖此类，盲区见声明）"),
    "RISK_OFF": ("crit", "假想减仓", "空头 up-jump × Musk 密集命中"),
}

# 探测器证据等级（strategy-lab N3-H 结论，常驻面板，P0-6）
DET_EVIDENCE = (
    "证据等级：历史推演 2 段全对，但块 bootstrap p=0.14——仅方向性证据；"
    "窄谱过滤器，对空头回补型/宏观型下跌失明"
    "（如 2024-12→2025-04 的 −53.8% 整段无信号）。广谱回撤防线看「今日合议」S2 读数。"
)


# ---------------------------------------------------------------- utilities

def esc(s: object) -> str:
    return html_mod.escape(str(s), quote=True) if s is not None else ""


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fmt_local(dt: datetime | None, with_date: bool = True) -> str:
    """UTC 时刻转本机时区显示。"""
    if dt is None:
        return "—"
    loc = dt.astimezone()
    return loc.strftime("%m-%d %H:%M" if with_date else "%H:%M")


def fmt_ago(dt: datetime | None, now: datetime) -> str:
    if dt is None:
        return ""
    s = (now - dt).total_seconds()
    if s < 0:
        return "未来 " + fmt_dur(-s)
    return fmt_dur(s) + "前"


def fmt_dur(s: float) -> str:
    """秒数人类可读（中文单位）。"""
    neg = s < 0
    s = abs(s)
    if s < 90:
        out = f"{s:.0f} 秒"
    elif s < 5400:
        out = f"{s / 60:.0f} 分钟"
    elif s < 2 * 86400:
        out = f"{s / 3600:.1f} 小时"
    else:
        out = f"{s / 86400:.1f} 天"
    return ("-" + out) if neg else out


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return row is not None


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def tier_badge(tier: str | None) -> str:
    """层级徽章：色深 + ◆▲●○ 形状双编码（设计稿 §一）。"""
    t = tier if tier in TIERS else "T3"
    return (
        f'<span class="tier tier-{t[1]}"><svg aria-hidden="true">'
        f'<use href="#g-t{t[1]}"/></svg>{esc(tier or "T?")}</span>'
    )


def bdays_between(a: date, b: date) -> int:
    """[a, b) 间的工作日数（周一至周五，近似交易日）。"""
    if b <= a:
        return 0
    n, d = 0, a
    while d < b:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def add_bdays(d: date, n: int) -> date:
    """d 之后第 n 个工作日（n>=0；n=0 返回 d 本身或下个工作日）。"""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def calib_eta(state_date: str, baseline_days: int) -> date | None:
    """标定期满预计日：state_date 起再累积 (CALIB_BDAYS - baseline_days) 个交易日。"""
    try:
        sd = date.fromisoformat(state_date)
    except (TypeError, ValueError):
        return None
    return add_bdays(sd, max(0, CALIB_BDAYS - baseline_days))


def age_badge(d: date | None, now: datetime, warn_days: int, label: str = "数据龄") -> str:
    """数据龄徽章（P0-2 全站规范）：超过 warn_days 天转黄，双倍转红。"""
    if d is None:
        return f'<span class="age crit">{esc(label)}未知</span>'
    days = (now.astimezone(ET).date() - d).days
    cls = "crit" if days > 2 * warn_days else ("warn" if days > warn_days else "")
    txt = "今日" if days <= 0 else f"{days} 天前"
    return f'<span class="age {cls}">{esc(label)} {txt}</span>'


# ---------------------------------------------------------------- data pulls

def load_sources(conn: sqlite3.Connection) -> dict[str, dict]:
    if not has_table(conn, "sources"):
        return {}
    return {
        r["source_id"]: dict(r)
        for r in conn.execute("SELECT * FROM sources ORDER BY tier, source_id")
    }


def load_detector(conn: sqlite3.Connection) -> dict | None:
    """探测器面板数据；detector_state 缺失或无行 → None（整面板隐藏）。"""
    if not has_table(conn, "detector_state"):
        return None
    cur = conn.execute(
        "SELECT * FROM detector_state ORDER BY state_date DESC LIMIT 1"
    ).fetchone()
    if cur is None:
        return None
    switches: list[dict] = []
    if has_table(conn, "events"):
        for r in conn.execute(
            """SELECT event_time_utc, title, payload_json FROM events
               WHERE source_id = 'detector' AND type = 'detector_state'
               ORDER BY event_time_utc DESC LIMIT 5"""
        ):
            row = dict(r)
            try:
                row["state"] = json.loads(row.get("payload_json") or "{}").get("state")
            except ValueError:
                row["state"] = None
            switches.append(row)
    trades: list[dict] = []
    if has_table(conn, "detector_trades"):
        trades = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM detector_trades ORDER BY trade_time_utc, trade_id"
            )
        ]
    return {"cur": dict(cur), "switches": switches, "trades": trades}


def load_health(conn: sqlite3.Connection, sources: dict) -> list[dict]:
    """每渠道：最近轮询 + 连续失败数 + 累计事件数。"""
    if not has_table(conn, "poll_log"):
        return []
    ev_counts: dict[str, dict] = {}
    if has_table(conn, "events"):
        for r in conn.execute(
            """SELECT source_id, COUNT(*) n, MAX(observed_time_utc) last_obs
               FROM events GROUP BY source_id"""
        ):
            ev_counts[r["source_id"]] = dict(r)

    rows = []
    sids = list(sources) or [
        r["source_id"]
        for r in conn.execute("SELECT DISTINCT source_id FROM poll_log")
    ]
    for sid in sids:
        src = sources.get(sid, {})
        last = conn.execute(
            """SELECT poll_time_utc, ok, n_seen, n_new, duration_ms, error
               FROM poll_log WHERE source_id = ?
               ORDER BY poll_id DESC LIMIT 1""",
            (sid,),
        ).fetchone()
        streak = 0
        if last is not None:
            for r in conn.execute(
                "SELECT ok FROM poll_log WHERE source_id = ? "
                "ORDER BY poll_id DESC LIMIT 100",
                (sid,),
            ):
                if r["ok"] == 0:
                    streak += 1
                else:
                    break
        rows.append(
            {
                "source_id": sid,
                "name": src.get("name", sid),
                "tier": src.get("tier"),
                "method": src.get("method", ""),
                "weight": src.get("weight_source"),
                "poll_interval_s": src.get("poll_interval_s") or 0,
                "last": dict(last) if last is not None else None,
                "fail_streak": streak,
                "n_events": ev_counts.get(sid, {}).get("n", 0),
                "last_obs": ev_counts.get(sid, {}).get("last_obs"),
            }
        )
    # tier 升序 → 权重降序 → source_id（无 tier/权重的排最后）
    rows.sort(key=lambda h: (h["tier"] or "T9", -(h["weight"] or 0.0), h["source_id"]))
    return rows


def load_timeline(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    if not has_table(conn, "events"):
        return []
    join_tier = has_table(conn, "sources")
    sql = (
        """SELECT e.observed_time_utc, e.event_time_utc, e.source_id,
                  e.type, e.title, e.url{tier_col}
           FROM events e {join}
           ORDER BY e.observed_time_utc DESC, e.event_id DESC LIMIT ?"""
    ).format(
        tier_col=", s.tier" if join_tier else "",
        join="LEFT JOIN sources s ON s.source_id = e.source_id" if join_tier else "",
    )
    return [dict(r) for r in conn.execute(sql, (limit,))]


def load_latency(conn: sqlite3.Connection) -> list[dict]:
    """基于 v_latency 的渠道集合，p50/p90 在 Python 里补算。

    P2-1 稳态口径：p50/p90 只统计"首采日次日起 observed"的增量事件——
    首采回填（老事件今天才入库）的天文数字不再污染分位数；回填条数单列。
    """
    if not (has_table(conn, "v_latency") and has_table(conn, "events")):
        return []
    out = []
    for r in conn.execute("SELECT * FROM v_latency ORDER BY source_id"):
        rows = conn.execute(
            """SELECT observed_time_utc,
                      (julianday(observed_time_utc) - julianday(event_time_utc))
                      * 86400.0 AS lag_s
               FROM events WHERE source_id = ?""",
            (r["source_id"],),
        ).fetchall()
        first_day = min((x["observed_time_utc"] for x in rows), default="")[:10]
        steady = sorted(
            x["lag_s"] for x in rows if x["observed_time_utc"][:10] > first_day
        )
        row = dict(r)
        row["n_steady"] = len(steady)
        row["n_backfill"] = len(rows) - len(steady)
        row["p50"] = percentile(steady, 0.50)
        row["p90"] = percentile(steady, 0.90)
        out.append(row)
    return out


def load_tier_counts(conn: sqlite3.Connection, now: datetime) -> dict[str, dict]:
    """今日（ET 交易日口径，P2-2）与近 7 日按层级的事件计数（observed 口径）。"""
    if not (has_table(conn, "events") and has_table(conn, "sources")):
        return {}
    day0 = (now.astimezone(ET)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(timezone.utc))
    week0 = now - timedelta(days=7)
    counts: dict[str, dict] = {}
    for key, since in (("today", day0), ("week", week0)):
        for r in conn.execute(
            """SELECT s.tier, COUNT(*) n
               FROM events e JOIN sources s ON s.source_id = e.source_id
               WHERE e.observed_time_utc >= ?
               GROUP BY s.tier""",
            (since.isoformat(timespec="seconds"),),
        ):
            counts.setdefault(r["tier"], {"today": 0, "week": 0})[key] = r["n"]
    return counts


# ------------------------------------------------- symbol view · data pulls
# 约定：loader 返回 None = 数据源不可用（图层按钮置灰）；[] = 可用但无记录。

def _ffloat(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if f == f else None  # NaN → None
    except (TypeError, ValueError):
        return None


def load_daily_closes() -> tuple[list[date], list[float]]:
    """ET 交易日日线收盘（复用 src.common.data_io.load_bars）；失败返回空。"""
    try:
        import sys
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.common.data_io import load_bars
        df = load_bars(str(BARS_CSV))
        et = df.index.tz_convert("America/New_York")
        s = df["Close"].groupby(et.date).last()
        return list(s.index), [float(v) for v in s.values]
    except Exception:  # noqa: BLE001
        return [], []


def load_risk_segments() -> list[tuple[date, date]] | None:
    """daily_states.csv 中 RISK_OFF 连续区段（交易日口径的起止日期）。"""
    try:
        segs: list[tuple[date, date]] = []
        run_start = prev = None
        with STATES_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row["date"].strip())
                if (row.get("state") or "").strip().lower() == "risk_off":
                    if run_start is None:
                        run_start = d
                    prev = d
                elif run_start is not None:
                    segs.append((run_start, prev))
                    run_start = prev = None
        if run_start is not None and prev is not None:
            segs.append((run_start, prev))
        return segs
    except Exception:  # noqa: BLE001
        return None


def load_blind_segments() -> list[tuple[date, date]] | None:
    """daily_states.csv 中 tweet_data_blind=True 连续区段（Musk 数据失明期）。

    P0-3：历史推演在这些区段没有放风腿数据——图上"无避险段"≠"判定安全"。
    """
    try:
        segs: list[tuple[date, date]] = []
        run_start = prev = None
        with STATES_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row["date"].strip())
                if (row.get("tweet_data_blind") or "").strip().lower() in ("true", "1"):
                    if run_start is None:
                        run_start = d
                    prev = d
                elif run_start is not None:
                    segs.append((run_start, prev))
                    run_start = prev = None
        if run_start is not None and prev is not None:
            segs.append((run_start, prev))
        return segs
    except Exception:  # noqa: BLE001
        return None


SHADOW_JOURNAL = PROJECT_ROOT / "outputs" / "shadow_live" / "journal.sqlite"


def load_shadow(now: datetime) -> dict:
    """策略线（shadow 白跑）健康：journal.sqlite 最后运行/信号数；空则如实说。"""
    out: dict = {"exists": SHADOW_JOURNAL.exists(), "empty": True,
                 "last_run": None, "n_runs": 0, "n_orders": 0,
                 "n_orders_7d": 0, "last_equity_ts": None, "error": None}
    if not out["exists"] or SHADOW_JOURNAL.stat().st_size == 0:
        return out
    try:
        conn = sqlite3.connect(f"file:{SHADOW_JOURNAL}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "runs" in tables:
                r = conn.execute(
                    "SELECT COUNT(*) n, MAX(started_at) t FROM runs").fetchone()
                out["n_runs"], out["last_run"] = r["n"], r["t"]
            if "orders" in tables:
                out["n_orders"] = conn.execute(
                    "SELECT COUNT(*) FROM orders").fetchone()[0]
                week_ago = (now - timedelta(days=7)).isoformat(timespec="seconds")
                out["n_orders_7d"] = conn.execute(
                    "SELECT COUNT(*) FROM orders WHERE created_at >= ?",
                    (week_ago,)).fetchone()[0]
            if "equity_curve" in tables:
                out["last_equity_ts"] = conn.execute(
                    "SELECT MAX(ts) FROM equity_curve").fetchone()[0]
            out["empty"] = not (out["n_runs"] or out["n_orders"])
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


SHADOW_STATUS = PROJECT_ROOT / "outputs" / "shadow_status.json"


def load_shadow_status() -> dict | None:
    """shadow_status.json：各策略（e2/e8a）最近会话摘要，由 trading/run.py 会话结束时写。"""
    if not SHADOW_STATUS.exists():
        return None
    try:
        return json.loads(SHADOW_STATUS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_pits() -> list[dict] | None:
    """pits_catalog.csv：label=true → 真坑，label=false → 假坑（mid 不上图）。"""
    try:
        out = []
        with PITS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "").strip().lower()
                if label not in ("true", "false"):
                    continue  # mid 等口径含混的坑不上图
                close = _ffloat(row.get("pit_close"))
                if close is None:
                    continue
                out.append(
                    {
                        "date": date.fromisoformat(row["pit_date"].strip()),
                        "close": close,
                        "golden": label == "true",
                        "dd": _ffloat(row.get("dd_pct")),
                        "fwd": _ffloat(row.get("fwd60_max_pct")),
                        "si6": _ffloat(row.get("si_chg_6wk_pct")),
                        "mtr": _ffloat(row.get("musk_trend_ratio")),
                    }
                )
        return out
    except Exception:  # noqa: BLE001
        return None


def load_musk_buys() -> list[dict] | None:
    """edgar_form4.csv：insider_buy 且申报人含 Musk；日期取首个成交日。"""
    try:
        out = []
        with FORM4_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("type") or "").strip() != "insider_buy":
                    continue
                try:
                    pl = json.loads(row.get("payload") or "{}")
                except ValueError:
                    continue
                if "musk" not in (pl.get("owner") or "").lower():
                    continue
                tds = pl.get("trade_dates") or []
                if not tds:
                    continue
                out.append(
                    {
                        "date": date.fromisoformat(min(tds)),
                        "owner": pl.get("owner") or "Musk",
                        "usd": _ffloat(pl.get("value_usd")),
                        "vwap": _ffloat(pl.get("vwap")),
                        "shares": _ffloat(pl.get("shares")),
                    }
                )
        return out
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------- symbol view · rendering

def _fmt_pct_pt(v: float | None) -> str:
    """已是百分数值（-60.56 → −60.6%）。"""
    return f"{v:+.1f}%" if v is not None else "—"


def _fmt_usd_cn(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1e8:.1f} 亿美元" if v >= 1e8 else f"{v / 1e4:,.0f} 万美元"


def _log_ticks(lo: float, hi: float) -> list[float]:
    ladder = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70,
              100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
    ticks = [t for t in ladder if lo <= t <= hi]
    while len(ticks) > 7:
        ticks = ticks[::2]
    return ticks


def _x_ticks(d0: date, d1: date) -> list[tuple[date, str]]:
    span = (d1 - d0).days
    out: list[tuple[date, str]] = []
    if span > 4 * 365:  # 年初
        for y in range(d0.year, d1.year + 1):
            t = date(y, 1, 1)
            if d0 <= t <= d1:
                out.append((t, str(y)))
    elif span > 500:  # 半年
        for y in range(d0.year, d1.year + 1):
            for m in (1, 7):
                t = date(y, m, 1)
                if d0 <= t <= d1:
                    out.append((t, str(y) if m == 1 else f"{y}-07"))
    else:  # 双月
        y, m = d0.year, d0.month
        for _ in range(14):
            m += 2
            if m > 12:
                y, m = y + 1, m - 12
            t = date(y, m, 1)
            if t > d1:
                break
            out.append((t, f"{t.month} 月" if t.month != 1 else str(t.year)))
    return out


def _tip_attr(lines: list[str]) -> str:
    """自绘 tooltip 载荷（| 分隔）+ 原生 title 兜底共用的文本。"""
    return "|".join(lines)


# ---- 军事地图式旗标（设计稿 §02：halo → 杆 → 旗；颜色走 --m-* token）----

def _mk_halo(inner: str) -> str:
    return f'<g class="mk-halo">{inner}</g>'


def _mk_pit(x: float, y: float, solid: bool, color: str) -> str:
    """尖旗，杆朝下：真坑实心 / 假坑空心加斜杠。"""
    pole = f'M{x:.1f} {y + 4:.1f} V{y + 21:.1f}'
    flag = (
        f'M{x:.1f} {y + 21:.1f} L{x + 9:.1f} {y + 16.2:.1f} L{x:.1f} {y + 11.4:.1f} Z'
    )
    g = _mk_halo(f'<path d="{pole}"/><path d="{flag}"/>')
    g += f'<path d="{pole}" stroke="{color}" stroke-width="1.6" fill="none"/>'
    g += (
        f'<path d="{flag}" fill="{color}" stroke="none"/>'
        if solid
        else f'<path d="{flag}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    )
    if not solid:
        g += (
            f'<path d="M{x - 3.4:.1f} {y + 22.5:.1f} L{x + 7.4:.1f} {y + 9:.1f}" '
            f'stroke="{color}" stroke-width="1.5" fill="none"/>'
        )
    return g


def _mk_diamond(x: float, y: float, color: str) -> str:
    """菱旗，杆朝上：内部人买入。"""
    pole = f'M{x:.1f} {y - 4:.1f} V{y - 17:.1f}'
    dm = (
        f'M{x:.1f} {y - 27:.1f} L{x + 5:.1f} {y - 22:.1f} L{x:.1f} {y - 17:.1f} '
        f'L{x - 5:.1f} {y - 22:.1f} Z'
    )
    return (
        _mk_halo(f'<path d="{pole}"/><path d="{dm}"/>')
        + f'<path d="{pole}" stroke="{color}" stroke-width="1.6" fill="none"/>'
        + f'<path d="{dm}" fill="{color}" stroke="none"/>'
    )


def _mk_cross(x: float, y: float, tag: str, color: str) -> str:
    """十字准星 + 虚线杆 + 编号：假想单。"""
    cy = y - 38
    g = _mk_halo(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="7.5"/>')
    g += (
        f'<path d="M{x:.1f} {y - 4:.1f} V{cy + 9:.1f}" stroke="{color}" '
        'stroke-width="1.4" stroke-dasharray="3 2.4" fill="none"/>'
    )
    g += f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="7.5" fill="none" stroke="{color}" stroke-width="1.6"/>'
    g += f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="1.6" fill="{color}"/>'
    for dx1, dy1, dx2, dy2 in ((0, -11, 0, -6), (0, 6, 0, 11), (-11, 0, -6, 0), (6, 0, 11, 0)):
        g += (
            f'<path d="M{x + dx1:.1f} {cy + dy1:.1f} L{x + dx2:.1f} {cy + dy2:.1f}" '
            f'stroke="{color}" stroke-width="1.6" fill="none"/>'
        )
    g += (
        f'<text x="{x + 12:.1f}" y="{cy - 8:.1f}" class="mk-tag" '
        f'fill="{color}">{esc(tag)}</text>'
    )
    return g


def _render_view_svg(
    vid: str,
    dates: list[date],
    closes: list[float],
    i0: int,
    risk: list[tuple[date, date]],
    blind: list[tuple[date, date]],
    pits: list[dict],
    buys: list[dict],
    trades: list[dict],
    active: bool,
) -> tuple[str, dict]:
    """单个时间窗的完整 SVG；返回 (html, 供 JS 十字线用的标尺 meta)。"""
    sub_d, sub_c = dates[i0:], closes[i0:]
    d0, d1 = sub_d[0], sub_d[-1]
    d0o, d1o = d0.toordinal(), d1.toordinal()
    dspan = max(1, d1o - d0o)
    pw, ph = _VB_W - _ML - _MR, _VB_H - _MT - _MB
    llo, lhi = math.log10(min(sub_c)), math.log10(max(sub_c))
    pad = 0.05 * ((lhi - llo) or 1.0)
    llo, lhi = llo - pad, lhi + pad
    lspan = lhi - llo

    def x(o: int) -> float:
        return _ML + (o - d0o) / dspan * pw

    def y(c: float) -> float:
        return _MT + (1 - (math.log10(c) - llo) / lspan) * ph

    parts: list[str] = []

    # -- 失明期底纹（最底层，P0-3）：Musk 数据失明区段灰示——
    #    该段历史推演无放风腿数据，"无避险段"≠"判定安全"
    blind_rects = []
    for a, b in blind:
        ao, bo = max(a.toordinal(), d0o), min(b.toordinal() + 1, d1o)
        if bo <= d0o or ao >= d1o:
            continue
        bx, bw = x(ao), x(bo) - x(ao)
        tag = (
            f'<text class="blind-tag" x="{bx + bw / 2:.1f}" y="{_MT + ph - 8}" '
            f'text-anchor="middle">Musk 数据失明期（探测器该段无数据）</text>'
            if bw > 250
            else ""
        )
        blind_rects.append(
            f'<rect class="blind-wash" x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" '
            f'height="{ph}"><title>Musk 数据失明期 {a} → {b}：历史推演该段无'
            "放风腿数据，无避险段 ≠ 判定安全</title></rect>"
            f"{tag}"
        )
    if blind_rects:
        parts.append(f'<g class="ly ly-blind">{"".join(blind_rects)}</g>')

    # -- 避险底纹（状态色 wash + 45° 影线 + 虚线沿 + R 序号旗标）
    band_rects = []
    for bi, (a, b) in enumerate(risk, start=1):
        ao, bo = max(a.toordinal(), d0o), min(b.toordinal() + 1, d1o)
        if bo <= d0o or ao >= d1o:
            continue
        bx, bw = x(ao), x(bo) - x(ao)
        tag = (
            f'<text class="band-tag" x="{bx + bw / 2:.1f}" y="{_MT + 13}" '
            f'text-anchor="middle">避险 R{bi}</text>'
            if bw > 46
            else ""
        )
        band_rects.append(
            f'<rect class="band-wash" x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" '
            f'height="{ph}"><title>避险区段（RISK_OFF）{a} → {b}</title></rect>'
            f'<rect x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" height="{ph}" '
            'fill="url(#hatch45)" pointer-events="none"/>'
            f'<line class="band-edge" x1="{bx:.1f}" x2="{bx:.1f}" y1="{_MT}" y2="{_MT + ph}"/>'
            f'<line class="band-edge" x1="{bx + bw:.1f}" x2="{bx + bw:.1f}" '
            f'y1="{_MT}" y2="{_MT + ph}"/>{tag}'
        )
    if band_rects:
        parts.append(f'<g class="ly ly-risk">{"".join(band_rects)}</g>')

    # -- 网格 + 轴（隐性发丝线；文字用文本色 token）
    grid, labels = [], []
    for t in _log_ticks(10 ** llo, 10 ** lhi):
        ty = y(t)
        grid.append(f'<line x1="{_ML}" y1="{ty:.1f}" x2="{_VB_W - _MR}" y2="{ty:.1f}"/>')
        lab = f"{t:,.0f}" if t >= 1 else f"{t}"
        labels.append(
            f'<text x="{_ML - 8}" y="{ty:.1f}" class="tv-tick" text-anchor="end" '
            f'dominant-baseline="central">{lab}</text>'
        )
    for t, lab in _x_ticks(d0, d1):
        tx = x(t.toordinal())
        grid.append(f'<line x1="{tx:.1f}" y1="{_MT}" x2="{tx:.1f}" y2="{_MT + ph}"/>')
        labels.append(
            f'<text x="{tx:.1f}" y="{_VB_H - 10}" class="tv-tick" '
            f'text-anchor="middle">{esc(lab)}</text>'
        )
    parts.append(f'<g class="tv-grid">{"".join(grid)}</g>{"".join(labels)}')
    parts.append(
        f'<text x="{_ML - 8}" y="{_MT - 3}" class="tv-tick" text-anchor="end">log</text>'
    )

    # -- 价格折线（--ink-2 中性 2px，单系列不占图例色彩）+ 现价端标
    pts = " ".join(
        f"{x(d.toordinal()):.1f},{y(c):.1f}" for d, c in zip(sub_d, sub_c)
    )
    parts.append(f'<polyline class="tv-price" points="{pts}"/>')
    ex, ey = x(d1o), y(sub_c[-1])
    parts.append(
        f'<circle class="tv-end" cx="{ex:.1f}" cy="{ey:.1f}" r="3"/>'
        f'<text class="tv-endlab" x="{ex - 6:.1f}" y="{ey - 8:.1f}" '
        f'text-anchor="end">{sub_c[-1]:,.1f}</text>'
    )

    # -- 十字线 + 悬停捕捉面（标记层在其上，优先接管指针）
    parts.append(
        '<g class="xh" hidden><line/><circle r="4.5"/></g>'
        f'<rect class="tv-ov" data-view="{vid}" x="{_ML}" y="{_MT}" '
        f'width="{pw}" height="{ph}"/>'
    )

    # -- 标记层（形状=身份的第二编码；金/蓝已过配色校验，灰为刻意的中性槽位）
    def marker_group(cls: str, items: list[str]) -> None:
        if items:
            parts.append(f'<g class="ly {cls}">{"".join(items)}</g>')

    g_items, t_items = [], []
    for p in pits:
        if not (d0 <= p["date"] <= d1):
            continue
        mx, my = x(p["date"].toordinal()), y(p["close"])
        kind = "真坑" if p["golden"] else "假坑"
        tip = _tip_attr(
            [
                f"{kind} · {p['date']}",
                "判据：坑底后先收复+25%=真坑 / 先破位-10%=假坑（60日窗，事后）",
                f"坑底收盘 {p['close']:,.2f}",
                f"深度 {_fmt_pct_pt(p['dd'])}",
                f"后 60 日最高 {_fmt_pct_pt(p['fwd'])}",
                f"空头 6 周变化 {_fmt_pct_pt(p['si6'])}",
                f"Musk 趋势比 {p['mtr']:.2f}" if p["mtr"] is not None else "Musk 趋势比 —",
            ]
        )
        cvar = "var(--m-pit)" if p["golden"] else "var(--m-fake)"
        shape = f'<g class="{"mk-gold" if p["golden"] else "mk-trap"}">{_mk_pit(mx, my, p["golden"], cvar)}</g>'
        hit = (
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my + 14:.1f}" r="15" '
            f'data-tip="{esc(tip)}"><title>{esc(tip.replace("|", chr(10)))}</title>'
            "</circle>"
        )
        (g_items if p["golden"] else t_items).append(shape + hit)
    marker_group("ly-pitg", g_items)
    marker_group("ly-pitt", t_items)

    ords = [d.toordinal() for d in dates]
    m_items = []
    for b in buys:
        i = bisect_left(dates, b["date"])
        if i >= len(dates) or (dates[i] - b["date"]).days > 5:
            continue  # 早于价格数据起点等，无法对齐
        if not (d0 <= dates[i] <= d1):
            continue
        mx, my = x(ords[i]), y(closes[i])
        fwd20 = (
            f"{(closes[i + 20] / closes[i] - 1) * 100:+.1f}%"
            if i + 20 < len(closes)
            else "—（未满 20 交易日）"
        )
        vwap = f"{b['vwap']:,.2f}" if b["vwap"] is not None else "—"
        shares = f"{b['shares']:,.0f} 股" if b["shares"] is not None else ""
        tip = _tip_attr(
            [
                f"{b['owner']} 买入 · {b['date']}",
                f"金额 {_fmt_usd_cn(b['usd'])}",
                f"均价 {vwap}" + (f" · {shares}" if shares else ""),
                f"后 20 日 {fwd20}",
            ]
        )
        m_items.append(
            f'<g class="mk-musk">{_mk_diamond(mx, my, "var(--m-insider)")}</g>'
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my - 16:.1f}" r="15" '
            f'data-tip="{esc(tip)}"><title>{esc(tip.replace("|", chr(10)))}</title>'
            "</circle>"
        )
    marker_group("ly-musk", m_items)

    d_items = []
    n_reduce = 0
    for t in trades:
        reduce_ = t.get("action") == "REDUCE"
        if reduce_:
            n_reduce += 1
        try:
            td = date.fromisoformat(str(t.get("state_date")))
        except (TypeError, ValueError):
            continue
        px = t.get("price")
        if px is None or not (d0 <= td <= d1):
            continue
        mx, my = x(td.toordinal()), y(float(px))
        tag = (f"H{n_reduce}" if reduce_ else f"R{n_reduce}") or "H?"
        tip = _tip_attr(
            [
                f"假想单 {tag} · {t.get('action')} · {td}",
                f"TSLA 快照 {float(px):,.2f}",
                (t.get("note") or "")[:60] or "—",
            ]
        )
        d_items.append(
            f'<g class="mk-hypo">{_mk_cross(mx, my, tag, "var(--m-hypo)")}</g>'
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my - 38:.1f}" r="14" '
            f'data-tip="{esc(tip)}"><title>{esc(tip.replace("|", chr(10)))}</title>'
            "</circle>"
        )
    marker_group("ly-trade", d_items)

    svg = (
        f'<div class="tv-view{" act" if active else ""}" id="tv-view-{vid}">'
        f'<svg class="tv-svg" viewBox="0 0 {_VB_W} {_VB_H}" role="img" '
        f'aria-label="TSLA 日线收盘走势（对数刻度）及历史标注">{"".join(parts)}</svg></div>'
    )
    meta = {
        "i0": i0, "d0": d0o, "ds": dspan, "ml": _ML, "pw": pw,
        "mt": _MT, "ph": ph, "ly0": round(llo, 6), "lys": round(lspan, 6),
    }
    return svg, meta


# 图例芯片内嵌小样（设计稿 legendrow 的 20×14 视框微缩旗标）
_LG_SW = {
    "blind": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><rect x="1" y="1" width="18" '
        'height="12" fill="var(--muted)" opacity=".28"/></svg>'
    ),
    "risk": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><rect x="1" y="1" width="18" '
        'height="12" fill="var(--crit-wash)" stroke="var(--crit)" stroke-width="1" '
        'stroke-dasharray="2 2"/><path d="M4 12 10 2 M9 12 15 2" stroke="var(--crit)" '
        'stroke-width="1" opacity=".55" fill="none"/></svg>'
    ),
    "pitg": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><path d="M10 1 V8" '
        'stroke="var(--m-pit)" stroke-width="1.6" fill="none"/>'
        '<path d="M10 13 15 9.5 10 6 Z" fill="var(--m-pit)"/></svg>'
    ),
    "pitt": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><path d="M10 1 V8" '
        'stroke="var(--m-fake)" stroke-width="1.6" fill="none"/>'
        '<path d="M10 13 15 9.5 10 6 Z" fill="none" stroke="var(--m-fake)" '
        'stroke-width="1.4"/><path d="M6.5 13 L14 2" stroke="var(--m-fake)" '
        'stroke-width="1.4" fill="none"/></svg>'
    ),
    "musk": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><path d="M10 13 V7" '
        'stroke="var(--m-insider)" stroke-width="1.6" fill="none"/>'
        '<path d="M10 1 14 5 10 9 6 5 Z" fill="var(--m-insider)"/></svg>'
    ),
    "trade": (
        '<svg viewBox="0 0 20 14" aria-hidden="true"><circle cx="10" cy="7" r="4.4" '
        'fill="none" stroke="var(--m-hypo)" stroke-width="1.5"/>'
        '<path d="M10 .8 V3.4 M10 10.6 V13.2 M3.8 7 H6.4 M13.6 7 H16.2" '
        'stroke="var(--m-hypo)" stroke-width="1.5" fill="none"/></svg>'
    ),
}


def _layer_btn(
    layer: str, sw: str, label: str, count_txt: str,
    disabled_note: str | None = None, title: str = "",
) -> str:
    sw_svg = _LG_SW.get(layer, sw)
    if disabled_note:
        return (
            f'<button class="lg tv-lb" data-layer="{layer}" disabled '
            f'title="{esc(disabled_note)}">{sw_svg}{esc(label)} '
            f'<span class="ct">{esc(disabled_note)}</span></button>'
        )
    return (
        f'<button class="lg tv-lb" data-layer="{layer}" aria-pressed="true" '
        f'title="{esc(title)}">{sw_svg}{esc(label)} '
        f'<span class="ct">×{esc(count_txt)}</span></button>'
    )


def _pits_table(pits: list[dict], buys_aligned: list[tuple[dict, str]]) -> str:
    """标注明细表——tooltip 之外的无悬停可达路径（可折叠）。"""
    rows = []
    for p in sorted(pits, key=lambda q: q["date"]):
        rows.append(
            "<tr>"
            f"<td>{'真坑 ⚑' if p['golden'] else '假坑 ⚐'}</td>"
            f'<td class="num">{p["date"]}</td>'
            f'<td class="num">{p["close"]:,.2f}</td>'
            f'<td class="num">{_fmt_pct_pt(p["dd"])}</td>'
            f'<td class="num">{_fmt_pct_pt(p["fwd"])}</td>'
            f'<td class="num">{_fmt_pct_pt(p["si6"])}</td>'
            f'<td class="num">{f"{p['mtr']:.2f}" if p["mtr"] is not None else "—"}</td>'
            "</tr>"
        )
    buy_rows = [
        "<tr>"
        f"<td>Musk 买入 ◆</td>"
        f'<td class="num">{b["date"]}</td>'
        f'<td class="num">{f"{b['vwap']:,.2f}" if b["vwap"] is not None else "—"}</td>'
        f'<td class="num" colspan="2">{_fmt_usd_cn(b["usd"])}</td>'
        f'<td class="num" colspan="2">后 20 日 {fwd}</td>'
        "</tr>"
        for b, fwd in buys_aligned
    ]
    if not rows and not buy_rows:
        return ""
    return (
        '<details class="tv-table"><summary>标注明细（表格视图）</summary>'
        '<p class="footnote" style="margin:6px 0 4px">真假坑判据（N4 事后口径）：'
        "坑底后 60 日内<b>先收复 +25%</b> = 真坑；<b>先破位 −10%</b> 或反弹不足 = 假坑。"
        "「后 60 日最高」为前视指标，仅历史复盘。Musk 买入为 Form 4 申报事实（当时可知）。</p>"
        '<div class="scroll-x"><table>'
        "<thead><tr><th>类型</th><th>日期</th><th>价格</th><th>深度</th>"
        "<th>后 60 日最高</th><th>空头 6 周</th><th>Musk 趋势比</th></tr></thead>"
        f"<tbody>{''.join(rows)}{''.join(buy_rows)}</tbody></table></div></details>"
    )


def render_symbol_tabs(symbols: tuple[tuple[str, bool], ...] = (("TSLA", True),)) -> str:
    """标的标签栏——多标的预留位：将来每加一个标的多调一次 render_symbol_view。"""
    tabs = "".join(
        f'<span class="sym-tab{" act" if act else ""}"'
        + (' aria-current="true"' if act else "")
        + f">{esc(sym)}</span>"
        for sym, act in symbols
    )
    return f'<div class="sym-tabs" aria-label="标的">{tabs}<span class="sym-hint">多标的预留位</span></div>'


def render_symbol_view(
    symbol: str,
    dates: list[date],
    closes: list[float],
    risk: list[tuple[date, date]] | None,
    blind: list[tuple[date, date]] | None,
    pits: list[dict] | None,
    buys: list[dict] | None,
    trades: list[dict],
    has_trades_table: bool,
    fresh_badge: str = "",
) -> str:
    """③ 走势与历史判断——单标的完整区块（图 + 时间刷 + 图层开关 + 明细表）。"""
    if not dates:
        return f"""
<section>
  <h2><span class="sec-no">__NO__</span>战场走势<span class="h-sub">{esc(symbol)} 日线收盘 · 对数刻度 · 事后标注</span></h2>
  <div class="card"><p class="empty">价格数据不可读（{esc(str(BARS_CSV))}）——走势区块降级为空。</p></div>
</section>"""

    risk_l = risk or []
    blind_l = blind or []
    pits_l = pits or []
    buys_l = buys or []
    last = dates[-1]

    def idx_since(days: int) -> int:
        return bisect_left(dates, last - timedelta(days=days))

    views = [("1y", "1 年", idx_since(365)), ("3y", "3 年", idx_since(3 * 365)),
             ("8y", "8 年", 0)]
    svgs, metas = [], {}
    for vid, _, i0 in views:
        svg, meta = _render_view_svg(
            vid, dates, closes, i0, risk_l, blind_l, pits_l, buys_l, trades,
            active=(vid == "3y"),
        )
        svgs.append(svg)
        metas[vid] = meta

    # -- 图层按钮（含计数；数据源缺失→置灰说明）
    n_gold = sum(p["golden"] for p in pits_l)
    n_trap = len(pits_l) - n_gold
    buys_aligned: list[tuple[dict, str]] = []
    n_unaligned = 0
    for b in sorted(buys_l, key=lambda q: q["date"]):
        i = bisect_left(dates, b["date"])
        if i >= len(dates) or (dates[i] - b["date"]).days > 5:
            n_unaligned += 1
            continue
        fwd = (
            f"{(closes[i + 20] / closes[i] - 1) * 100:+.1f}%"
            if i + 20 < len(closes) else "—"
        )
        buys_aligned.append((b, fwd))
    risk_title = "；".join(f"{a} → {b}" for a, b in risk_l)
    musk_note = f"另 {n_unaligned} 笔早于价格数据起点，无法对齐" if n_unaligned else ""
    if not has_trades_table:
        trade_note: str | None = "detector_trades 表缺失"
    elif not trades:
        trade_note = "值班期尚无记录"
    else:
        trade_note = None
    blind_title = "；".join(f"{a} → {b}" for a, b in blind_l)
    # P0-3：图例分两区——研究回放（事后口径）不可与前向值班（当时可知）混读
    layer_btns = (
        '<span class="lg-zone">研究回放 · 事后口径</span>'
        + "".join(
            [
                _layer_btn(
                    "risk", "", "避险段",
                    f"{len(risk_l)}",
                    None if risk is not None else "数据缺失",
                    "N3-H 历史推演 RISK_OFF 区间（研究回放，非当时值班输出）"
                    + ("；" + risk_title if risk_title else ""),
                ),
                _layer_btn(
                    "pitg", "", "真坑",
                    f"{n_gold}", None if pits is not None else "数据缺失",
                    "事后判据：坑底后 60 日内先收复 +25%（含前视，仅历史复盘）",
                ),
                _layer_btn(
                    "pitt", "", "假坑",
                    f"{n_trap}", None if pits is not None else "数据缺失",
                    "事后判据：先破位 -10% 或 60 日反弹不足（含前视，仅历史复盘）",
                ),
                _layer_btn(
                    "blind", "", "失明期",
                    f"{len(blind_l)}",
                    None if blind is not None else "数据缺失",
                    "Musk 数据失明区段：探测器该段无放风腿数据，无避险段≠判定安全"
                    + ("；" + blind_title if blind_title else ""),
                ),
            ]
        )
        + '<span class="lg-zone">前向 / 事实 · 当时可知</span>'
        + "".join(
            [
                _layer_btn(
                    "musk", "", "Musk 买入",
                    f"{len(buys_aligned)}", None if buys is not None else "数据缺失",
                    "Form 4 申报事实（当时可知）" + ("；" + musk_note if musk_note else ""),
                ),
                _layer_btn(
                    "trade", "", "假想单",
                    f"{len(trades)}", trade_note, "探测器值班期的虚拟操作（真前向）",
                ),
            ]
        )
    )
    range_btns = "".join(
        f'<button class="tv-rb{" act" if vid == "3y" else ""}" '
        f'data-range="{vid}">{lab}</button>'
        for vid, lab, _ in views
    )

    data_json = json.dumps(
        {
            "D": [d.toordinal() for d in dates],
            "C": [round(c, 2) for c in closes],
            "epoch": _EPOCH_ORD,
            "views": metas,
        },
        separators=(",", ":"),
    )
    foot_extra = f"Musk 买入 {musk_note}。" if n_unaligned else ""
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>战场走势<span class="h-sub">{esc(symbol)} 日线收盘（ET 交易日聚合） · 对数刻度 · {esc(str(dates[0]))} → {esc(str(last))}</span>{fresh_badge}</h2>
  {render_symbol_tabs()}
  <div class="card chartcard" id="tv">
    <div class="tv-bar legendrow">
      <div class="tv-ranges" role="group" aria-label="时间范围">{range_btns}</div>
      <div class="tv-layers" role="group" aria-label="图层开关">{layer_btns}</div>
      <span class="spacer"></span>
      <span class="micro muted">点击图例开关图层</span>
    </div>
    <div class="tv-wrap" id="tv-wrap">
      {"".join(svgs)}
      <div id="tv-tip" hidden></div>
    </div>
    <div class="axis-note">
      <span>纵轴 USD（对数） · 横轴日线</span>
      <span>真坑 = 坑底后先收复 +25%（事后）</span>
      <span>假坑 = 先破位 −10% 或反弹不足（事后）</span>
      <span>避险段 = 历史推演 RISK_OFF（回放）</span>
      <span>灰底 = Musk 数据失明期</span>
    </div>
    {_pits_table(pits_l, buys_aligned)}
    <p class="footnote">「研究回放」区（避险段/真坑/假坑/失明期）全部为事后口径：
    坑由 N4 研究按"坑底后 60 日内先收复 +25% = 真坑、先破位 −10% = 假坑"贴标（含前视指标，
    仅作历史复盘）；避险段是 N3-H 对历史的回放，不是当时值班输出；灰色失明区段
    （Musk 归档止于 2025-05-08 之后至 nitter 前向覆盖前）内探测器无放风腿数据——
    该段没有避险标记是"看不见"，不是"判定安全"。「前向/事实」区：Musk 买入为 Form 4
    申报事实（菱旗落在首个成交日收盘价），假想单为值班期真前向记录。
    {esc(foot_extra)}价格轴为对数刻度；悬停标记看明细，悬停曲线看逐日收盘。</p>
  </div>
  <script type="application/json" id="tv-data">{data_json}</script>
</section>"""


# ------------------------------------------------------- 今日合议（决策卡）

def render_consensus(det: dict | None, px: dict | None, shadow: dict,
                     now: datetime) -> str:
    """① 今日合议：现价 · S2 开关读数 · 探测器 · 策略线 · 综合一句话.

    roadmap #1 / P0-1：把"我今天该怎么办"放在第一屏。规则合成，不引入新判断；
    诚实优先——无建议时明说无建议，并给出恢复时间预估。
    S2 = E11 压测过的全系统最强开关（距 252 交易日滚动高点回撤 >20% → 停用买入），
    从今天起由仪表盘每次生成时计算。
    """
    today_et = now.astimezone(ET).date()
    s2 = (px or {}).get("s2")

    # -- ① 现价
    if px and px.get("live_price") is not None:
        chg = px.get("chg_pct")
        chg_lab = ("今日" if px.get("chg_date") == today_et
                   else f"{px.get('chg_date')} 收盘" if px.get("chg_date") else "—")
        chg_cls = "crit-text" if (chg or 0) < 0 else "good-text"
        cache_tag = ('<span class="age warn">缓存价（本次取价失败）</span>'
                     if px.get("from_cache") else "")
        cell_px = (
            '<div class="cx-cell"><div class="cx-k">TSLA 现价</div>'
            f'<div class="cx-v num">{px["live_price"]:,.2f}'
            + (f' <span class="{chg_cls}">{chg:+.2f}%</span>' if chg is not None else "")
            + "</div>"
            f'<div class="cx-ref">{esc(chg_lab)}涨跌 · yfinance 快照 '
            f"{age_badge(px.get('price_asof'), now, 1, '价格龄')}{cache_tag}</div></div>"
        )
    else:
        err = (px or {}).get("error") or "价格模块不可用"
        cell_px = (
            '<div class="cx-cell crit"><div class="cx-k">TSLA 现价</div>'
            '<div class="cx-v crit-text">取价失败</div>'
            f'<div class="cx-ref" title="{esc(err)}">yfinance 与缓存均不可用——'
            "现价/S2 读数本次缺席，勿当作无风险</div></div>"
        )

    # -- ② S2 开关（每日必算）
    if s2:
        s2_cls = " crit" if s2["triggered"] else ""
        s2_pill = (
            '<span class="pill sm crit"><span class="dot"></span>S2 触发</span>'
            if s2["triggered"]
            else '<span class="pill sm good"><span class="dot"></span>未触发</span>'
        )
        s2_note = (
            f"超过 −20% 线 {abs(s2['margin_pp']):.1f} pp——E11 口径：买入策略停用区"
            if s2["triggered"]
            else f"距 −20% 线余量 {s2['margin_pp']:.1f} pp"
        )
        cell_s2 = (
            f'<div class="cx-cell{s2_cls}"><div class="cx-k">S2 开关 · 距 252 日高回撤'
            f"{s2_pill}</div>"
            f'<div class="cx-v num{" crit-text" if s2["triggered"] else ""}">'
            f'{s2["drawdown_pct"]:+.1f}%</div>'
            f'<div class="cx-ref">252 日高 {s2["high"]:,.2f}（{esc(str(s2["high_date"]))}）· '
            f'现值 {s2["ref_price"]:,.2f} · {esc(s2_note)}</div></div>'
        )
    else:
        cell_s2 = (
            '<div class="cx-cell crit"><div class="cx-k">S2 开关 · 距 252 日高回撤</div>'
            '<div class="cx-v crit-text">无法计算</div>'
            '<div class="cx-ref">价格序列不足或取价失败——今天没人替你算 S2，'
            "这是缺口不是安全</div></div>"
        )

    # -- ③ 探测器
    if det:
        cur = det["cur"]
        state = cur["state"]
        cls, phrase, why = DET_STATE.get(state, ("off", state, ""))
        eta = calib_eta(str(cur.get("state_date")), int(cur.get("baseline_days") or 0))
        if state == "CALIBRATING":
            d_txt = (f"标定中 {cur.get('baseline_days') or 0}/{CALIB_BDAYS} · "
                     + (f"预计 {eta.strftime('%m-%d')} 恢复出信号" if eta else "期满出信号"))
        elif state == "RISK_OFF":
            d_txt = f"假想减仓生效 · F{PERSIST_BDAYS} 至 {cur.get('risk_off_until') or '—'}"
        else:
            d_txt = why
        cell_det = (
            '<div class="cx-cell"><div class="cx-k">探测器（窄谱避险）</div>'
            f'<div class="cx-v"><span class="{cls}-text">{esc(phrase)}</span></div>'
            f'<div class="cx-ref">{esc(d_txt)} · 盲区：不覆盖空头回补型/宏观型下跌'
            "（历史证据 2 段，p=0.14）</div></div>"
        )
        det_state = state
        det_eta = eta
    else:
        cell_det = (
            '<div class="cx-cell crit"><div class="cx-k">探测器（窄谱避险）</div>'
            '<div class="cx-v crit-text">无状态</div>'
            '<div class="cx-ref">detector_state 缺失——避险侧无输出</div></div>'
        )
        det_state, det_eta = None, None

    # -- ④ 策略线（shadow 白跑）
    status = load_shadow_status()
    e8a = (status or {}).get("strategies", {}).get("e8a")
    if e8a:
        sess_end = parse_ts(e8a.get("session_end") or "")
        e8a_s2 = e8a.get("s2") or {}
        s2_off = e8a_s2.get("off")
        s_v = ("S2 停用中 · 不发买入" if s2_off
               else f"{e8a.get('signals', 0)} 信号/会话")
        s_cls = " warn" if s2_off else ""
        parts = [f"E8-A+S2 已值班（{e8a.get('out_dir', 'outputs/shadow_e8a')}）"]
        if sess_end:
            parts.append(f"最近会话 {fmt_local(sess_end)}（{fmt_ago(sess_end, now)}）")
        if e8a_s2.get("dd_pct") is not None:
            parts.append(f"策略侧 S2 读数 {e8a_s2['dd_pct'] * 100:+.1f}%")
        if e8a.get("error"):
            s_cls, s_v = " crit", "会话异常"
            parts.append(f"error: {e8a['error']}")
        elif sess_end and (now - sess_end).total_seconds() > 3 * 86400:
            s_cls = " crit"
            parts.append("超过 3 天未运行")
        s_ref = " · ".join(parts)
    elif not shadow["exists"]:
        s_v, s_ref, s_cls = "无输出", "journal.sqlite 不存在——shadow 可能从未成功运行", " crit"
    elif shadow["empty"]:
        s_v, s_ref, s_cls = (
            "尚无记录",
            "journal 存在但为空——shadow 尚无任何运行/信号记录（买入侧无输出）",
            " crit",
        )
    else:
        last_run = parse_ts(shadow["last_run"])
        s_v = f"{shadow['n_orders_7d']} 信号/7日"
        s_ref = (
            f"最后运行 {fmt_local(last_run)}（{fmt_ago(last_run, now)}） · "
            f"累计 {shadow['n_runs']} 次运行 · {shadow['n_orders']} 单"
        )
        s_cls = ""
        if last_run and (now - last_run).total_seconds() > 3 * 86400:
            s_cls = " crit"
            s_ref += " · 超过 3 天未运行"
    if not e8a:
        s_ref += " · E8-A+S2 冻结候选尚未接入实时层"
    cell_sh = (
        f'<div class="cx-cell{s_cls}"><div class="cx-k">策略线（shadow 白跑）</div>'
        f'<div class="cx-v{" crit-text" if s_cls == " crit" else ""}">{esc(s_v)}</div>'
        f'<div class="cx-ref">{esc(s_ref)}</div></div>'
    )

    # -- 综合一句话（规则拼接，不引入新判断）
    bits = []
    if s2:
        if s2["triggered"]:
            bits.append(
                f"<b class=\"crit-text\">S2 已触发</b>（回撤 {s2['drawdown_pct']:+.1f}%，"
                "超过 −20% 线）——广谱回撤防线亮红，E11 压测口径下属买入策略停用区"
            )
        else:
            bits.append(
                f"S2 未触发（回撤 {s2['drawdown_pct']:+.1f}%，"
                f"距 −20% 线 {s2['margin_pp']:.1f} pp）"
            )
    else:
        bits.append("S2 读数缺席（取价失败）")
    if det_state == "CALIBRATING":
        bits.append(
            "探测器标定中"
            + (f"（预计 {det_eta.strftime('%m-%d')} 起值班）" if det_eta else "")
            + "，避险侧无输出"
        )
    elif det_state == "RISK_OFF":
        bits.append("探测器 RISK_OFF，假想减仓生效（窄谱）")
    elif det_state == "RISK_ON":
        bits.append("探测器未见空头知情型风险（仅覆盖此类）")
    else:
        bits.append("探测器无状态")
    if e8a:
        if (e8a.get("s2") or {}).get("off"):
            bits.append("买入侧 E8-A+S2 值班中，S2 停用区内不发买入信号")
        else:
            bits.append(f"买入侧 E8-A+S2 值班中（本会话 {e8a.get('signals', 0)} 信号）")
    elif shadow["empty"] or not shadow["exists"]:
        bits.append("买入侧（shadow）无输出")
    verdict_tail = "——<b>系统今日无正式建议</b>；以上为每日读数，供人工判断"
    if s2 and s2["triggered"]:
        verdict_tail = (
            "——系统无正式买卖信号，但 <b class=\"crit-text\">S2 广谱开关处于触发区</b>"
            "（唯一已通过压测的实时可算防线）；其余两路无输出，供人工判断"
        )
    verdict = "；".join(bits) + verdict_tail

    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>今日合议<span class="h-sub">每日决策卡 · {esc(str(today_et))}（ET） · 规则合成 · 诚实优先：无建议时明说无建议</span></h2>
  <div class="card cx">
    <div class="cx-grid">{cell_px}{cell_s2}{cell_det}{cell_sh}</div>
    <p class="statement cx-verdict">综合：{verdict}。</p>
    <p class="footnote">S2 定义（E11 冻结口径，研究原文 research/e11_bear_switch.py）：
    收盘距 252 交易日滚动高点回撤超过 −20% → 停用买入策略；历史压测中是唯一显著改善
    崩盘段亏损的开关，滞后指标、只防大势不防急跌。本卡由仪表盘每次生成时用最新价格
    计算——S2 从今天起每天有人算。四格中任何一格标红即为当日需要人眼确认的缺口。</p>
  </div>
</section>"""


# ---------------------------------------------------------------- rendering

def _topbar_price(px: dict | None, now: datetime) -> str:
    """顶栏 TSLA 现价 + 最近涨跌 + 数据龄（P0-2：决策台第一锚点）。"""
    if not px or px.get("live_price") is None:
        err = (px or {}).get("error") or "价格模块不可用"
        return (
            f'<span class="px crit-text" title="{esc(err)}">TSLA 取价失败</span>'
        )
    chg = px.get("chg_pct")
    today_et = now.astimezone(ET).date()
    chg_lab = "今日" if px.get("chg_date") == today_et else esc(
        f"{px.get('chg_date') or '—'} 收盘"
    )
    chg_cls = "crit-text" if (chg or 0) < 0 else "good-text"
    chg_html = (
        f'<span class="{chg_cls}">{chg:+.2f}%</span>'
        f'<span class="px-lab">{chg_lab}</span>'
        if chg is not None else ""
    )
    cache_note = "（缓存价）" if px.get("from_cache") else ""
    return (
        f'<span class="px" title="yfinance 快照{esc(cache_note)} · '
        f'取于 {esc(px.get("live_time_utc") or "—")}">'
        f'TSLA <b class="num">{px["live_price"]:,.2f}</b>{chg_html}'
        + (f'<span class="px-lab warn-text">缓存</span>' if px.get("from_cache") else "")
        + "</span>"
    )


def render_topbar(conn: sqlite3.Connection, now: datetime,
                  px: dict | None = None) -> str:
    last_obs = last_poll = None
    if has_table(conn, "events"):
        last_obs = parse_ts(
            conn.execute("SELECT MAX(observed_time_utc) FROM events").fetchone()[0]
        )
    if has_table(conn, "poll_log"):
        last_poll = parse_ts(
            conn.execute("SELECT MAX(poll_time_utc) FROM poll_log").fetchone()[0]
        )
    poll_age = (now - last_poll).total_seconds() if last_poll else None
    stale = poll_age is None or poll_age > STALE_S
    pill_cls = "crit" if stale else "good"
    pill_txt = "轮询中断" if stale else "运行中"

    def stat(label: str, dt: datetime | None, extra_cls: str = "", iso_id: str = "") -> str:
        iso = dt.isoformat() if dt else ""
        return (
            f'<div class="stamp {extra_cls}">'
            f'<div class="k">{label}</div>'
            f'<div class="v num" data-iso="{esc(iso)}"'
            + (f' id="{iso_id}"' if iso_id else "")
            + f">{fmt_local(dt)}"
            f'<span class="rel">{esc(fmt_ago(dt, now))}</span></div></div>'
        )

    return f"""
<div class="topbar">
  <div class="topbar-in">
    <div class="brand">
      <svg class="ic" aria-hidden="true"><use href="#i-radar"/></svg>
      <div>
        <h1>因果探测器 · 哨兵</h1>
        <div class="sub">TSLA CAUSAL SENTINEL — INTEL COMMAND</div>
      </div>
      <span class="pill {pill_cls}" id="poll-pill"><span class="dot"></span>{pill_txt}</span>
      {_topbar_price(px, now)}
    </div>
    <div class="topmeta">
      {stat("页面生成于", now, iso_id="gen-ts")}
      {stat("最后事件入库", last_obs)}
      {stat("最后轮询", last_poll, "poll-stat" + (" crit-text" if stale else ""), "poll-ts")}
      <button class="themebtn" id="themebtn" type="button">切换主题</button>
    </div>
  </div>
  <div class="topbar-in">
    <div class="banner" id="stale-banner" hidden>
      此页面生成已久，数据可能过期——请重新运行 <code>python -m intel.dashboard</code>
      或加载 launchd 定时任务。
    </div>
  </div>
</div>"""


def _fmt_count(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"


def _det_trades_html(trades: list[dict], now: datetime) -> str:
    """假想单判分（REDUCE→RESTORE 配对；成本线同 detector_report）。"""
    if not trades:
        return ""
    trs = []
    open_reduce = None
    n_right = n_wrong = 0
    for t in trades:
        verdict = ""
        if t["action"] == "REDUCE":
            open_reduce = t
        elif t["action"] == "RESTORE" and open_reduce is not None:
            if open_reduce["price"] is not None and t["price"] is not None:
                ret = t["price"] / open_reduce["price"] - 1
                ok = ret < COST_LINE
                n_right, n_wrong = n_right + ok, n_wrong + (not ok)
                verdict = (
                    f'<span class="{"good" if ok else "crit"}-text">'
                    f"期间 B&amp;H {ret:+.2%} · "
                    f"{'对（避损成立）' if ok else '错（空仓错过收益）'}</span>"
                )
            else:
                verdict = "缺价格快照，无法判分"
            open_reduce = None
        act_cls = "crit" if t["action"] == "REDUCE" else "good"
        px = f"{t['price']:.2f}" if t["price"] is not None else "n/a"
        tt = parse_ts(t["trade_time_utc"])
        trs.append(
            "<tr>"
            f'<td class="num">{fmt_local(tt)}</td>'
            f'<td><span class="pill sm {act_cls}"><span class="dot"></span>'
            f'{esc(t["action"])}</span></td>'
            f'<td class="num">{px}</td>'
            f'<td class="num">{esc(t["state_date"])}</td>'
            f'<td class="detail" title="{esc(t.get("note") or "")}">'
            f'{verdict or esc((t.get("note") or "")[:80])}</td></tr>'
        )
    if open_reduce is not None:
        trs.append(
            '<tr><td class="num">—</td><td colspan="4" class="detail">'
            "REDUCE 未平仓——判分以 RESTORE 时点为准（静态页不取现价）</td></tr>"
        )
    score = (
        f'<span class="h-sub">战绩 {n_right} 对 / {n_wrong} 错（已平仓配对，'
        f"成本线 {COST_LINE:+.2%}）</span>"
        if (n_right + n_wrong)
        else f'<span class="h-sub">成本线 {COST_LINE:+.2%} · 判分以 RESTORE 配对</span>'
    )
    return (
        f'<div class="det-sub"><h3>假想单判分 {score}</h3>'
        '<div class="scroll-x"><table class="det-table">'
        "<thead><tr><th>时刻</th><th>动作</th><th>TSLA 快照</th>"
        "<th>状态日</th><th>判分</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody></table></div></div>"
    )


def _ring_svg(state: str, cls: str, phrase: str, cur: dict) -> tuple[str, str]:
    """标定环表盘（设计稿 §01）：外圈刻度环 + 进度环 + 中心状态词。

    进度语义（标定期为设计稿方案；正式期设计稿未预留，此处自定并在页脚注明）：
      CALIBRATING → baseline_days / CALIB_BDAYS（基线累积）
      RISK_ON     → musk_count / dense_thr（放风腿密度占比，阈值未出则 0）
      RISK_OFF    → F 窗已行进比例（state_date → risk_off_until，工作日近似）
    返回 (svg, 环下标注文本)。
    """
    if state == "CALIBRATING":
        days = int(cur.get("baseline_days") or 0)
        pct = min(100.0, days / CALIB_BDAYS * 100)
        eta = calib_eta(str(cur.get("state_date")), days)
        prog_txt = f"{days} / {CALIB_BDAYS} 交易日"
        sub_txt = (
            f"不出信号 · 预计 {eta.strftime('%m-%d')} 恢复" if eta
            else f"基线累积 {pct:.0f}% · 不出信号"
        )
        cap = (f"标定环 · 预计 {eta} 期满出信号" if eta else "标定环 · 期满出信号")
    elif state == "RISK_OFF":
        until = None
        try:
            until = date.fromisoformat(str(cur.get("risk_off_until")))
        except (TypeError, ValueError):
            pass
        sd = date.fromisoformat(str(cur["state_date"]))
        left = bdays_between(sd, until) if until else 0
        pct = min(100.0, max(0.0, (PERSIST_BDAYS - left) / PERSIST_BDAYS * 100))
        prog_txt = f"F{PERSIST_BDAYS} 剩 {left} 交易日" if until else f"F{PERSIST_BDAYS} 窗内"
        sub_txt = ("至 " + str(until) + " · 期满自动恢复") if until else "重叠触发顺延"
        cap = "避险环 · F 窗行进度"
    else:  # RISK_ON 及未知态
        mc, thr = cur.get("musk_count"), cur.get("dense_thr")
        pct = min(100.0, mc / thr * 100) if (mc and thr) else 0.0
        prog_txt = (
            f"密度 {_fmt_count(mc)} / 阈 {_fmt_count(thr)}" if thr else "阈值未生效"
        )
        sub_txt = "放风腿密度占比 · 两腿未同时命中"
        cap = "警戒环 · 放风腿占比"
    code = state if state in DET_STATE else (state or "?")
    svg = (
        f'<svg class="ring {cls}" viewBox="0 0 224 224" role="img" '
        f'aria-label="{esc(phrase)}，{esc(prog_txt)}">'
        '<circle class="wash" cx="112" cy="112" r="72"/>'
        '<circle class="dial" cx="112" cy="112" r="104" fill="none" stroke-width="5" '
        'pathLength="120" stroke-dasharray=".55 1.45" opacity=".7"/>'
        '<circle class="track" cx="112" cy="112" r="90" fill="none" stroke-width="8"/>'
        '<circle class="prog" cx="112" cy="112" r="90" fill="none" stroke-width="8" '
        f'pathLength="100" stroke-dasharray="{pct:.1f} {100 - pct:.1f}" '
        'stroke-dashoffset="0" transform="rotate(-90 112 112)"/>'
        f'<text class="t-state" x="112" y="106" text-anchor="middle"'
        + (' style="font-size:24px"' if len(phrase) > 4 else "")
        + f">{esc(phrase)}</text>"
        f'<text class="t-code" x="112" y="128" text-anchor="middle">{esc(code)}</text>'
        f'<text class="t-prog" x="112" y="150" text-anchor="middle">{esc(prog_txt)}</text>'
        f'<text class="t-sub" x="112" y="166" text-anchor="middle">{esc(sub_txt)}</text>'
        "</svg>"
    )
    return svg, cap


def _lamps_html(state: str) -> str:
    """三灯信号组：RISK_ON / CALIBRATING / RISK_OFF，当前态点亮。"""
    rows = []
    for st, dot_cls, zh in (
        ("RISK_ON", "g", "未见目标风险"),
        ("CALIBRATING", "w", "标定中"),
        ("RISK_OFF", "c", "假想减仓"),
    ):
        on = " on" if st == state else ""
        zh_txt = zh + (" · 当前" if on else "")
        rows.append(
            f'<div class="lamp {dot_cls}{on}"><span class="bulb"></span>'
            f'<span class="name">{st}</span><span class="zh">{esc(zh_txt)}</span></div>'
        )
    return '<div class="lamps" role="list" aria-label="状态灯组">' + "".join(rows) + "</div>"


def _sparks_html(cur: dict) -> str:
    """腿 B 火花条：musk_window_json 的日计数（最多 14 日），p95 线 = 密集阈值。"""
    try:
        win = json.loads(cur.get("musk_window_json") or "{}")
        vals = [float(win[k]) for k in sorted(win)][-14:]
    except (ValueError, TypeError):
        vals = []
    if not vals:
        return ""
    thr = cur.get("dense_thr")
    mx = max(vals + ([thr] if thr else [])) * 1.15 or 1.0
    bars = "".join(
        f'<i class="{"hi" if (thr and v >= thr) else ""}" '
        f'style="height:{v / mx * 100:.0f}%" title="{v:.0f} 帖"></i>'
        for v in vals
    )
    p95 = (
        f'<span class="p95" style="top:{100 - thr / mx * 100:.0f}%"></span>' if thr else ""
    )
    return (
        f'<div class="sparks" aria-label="近 {len(vals)} 日发帖密度">{p95}{bars}</div>'
    )


def _hypo_summary(trades: list[dict]) -> str:
    """假想单判分小结（kv 行，供 hero 右栏；明细表另在下方保留）。"""
    if not trades:
        return (
            '<div class="kv"><span class="k">值班期</span>'
            '<span class="v">尚无假想单记录</span></div>'
        )
    rows, open_reduce = [], None
    n_right = n_wrong = n_open = 0
    hn = 0
    for t in trades:
        if t["action"] == "REDUCE":
            hn += 1
            open_reduce = (hn, t)
        elif t["action"] == "RESTORE" and open_reduce is not None:
            k, r0 = open_reduce
            if r0["price"] is not None and t["price"] is not None:
                ret = t["price"] / r0["price"] - 1
                ok = ret < COST_LINE
                n_right, n_wrong = n_right + ok, n_wrong + (not ok)
                v = (
                    f'<span class="good-text">判对 · 避损 {ret:+.2%}</span>'
                    if ok
                    else f'<span class="crit-text">判错 · 错过 {ret:+.2%}</span>'
                )
            else:
                v = "缺价格快照"
            rows.append(
                f'<div class="kv"><span class="k">H{k} · {esc(r0["state_date"])} 减仓</span>'
                f'<span class="v">{v}</span></div>'
            )
            open_reduce = None
    if open_reduce is not None:
        k, r0 = open_reduce
        n_open += 1
        rows.append(
            f'<div class="kv"><span class="k">H{k} · {esc(r0["state_date"])} 减仓</span>'
            '<span class="v">待定 · 未平仓</span></div>'
        )
    rows.append(
        f'<div class="kv"><span class="k">累计</span>'
        f'<span class="v">{hn} 单 · {n_right} 对 · {n_wrong} 错 · {n_open} 待定</span></div>'
    )
    return "".join(rows[-4:])


def render_detector(data: dict | None, now: datetime) -> str:
    if not data:
        return ""
    cur = data["cur"]
    state = cur["state"]
    cls, phrase, why = DET_STATE.get(state, ("off", state, ""))
    upd = parse_ts(cur.get("updated_utc"))
    ring, ring_cap = _ring_svg(state, cls, phrase, cur)

    # -- 中栏：态势陈述 + 两腿读数
    eta = calib_eta(str(cur.get("state_date")), int(cur.get("baseline_days") or 0))
    if state == "CALIBRATING":
        stmt = (
            "重建 nitter 口径基线中，两腿读数仅观测、<b>不触发信号</b>"
            + (f"（预计 <b>{eta}</b> 期满恢复出信号）" if eta else "")
            + "；两腿需同时命中且各自持续，方转入假想减仓。"
        )
    elif state == "RISK_OFF":
        stmt = (
            "空头 up-jump × Musk 密集同窗命中，<b>假想减仓生效</b>；"
            f"F{PERSIST_BDAYS} 窗内维持，重叠触发顺延。"
        )
    else:
        stmt = (
            "两腿未同时命中——<b>未见空头知情型风险</b>"
            "（本探测器仅覆盖此类下跌，不构成持仓建议，盲区见下方证据行）；"
            "空头利益与 Musk 发帖密度按各自周期滚动监测。"
        )
    since_bits = [f"状态日 {esc(cur['state_date'])}"]
    if data["switches"]:
        s0 = data["switches"][0]
        since_bits.append(
            f"最近切换 {fmt_local(parse_ts(s0['event_time_utc']))} · "
            f"{esc((s0.get('title') or '')[:40])}"
        )
    since = " · ".join(since_bits) + (
        f'<span class="rel" data-iso="{esc(upd.isoformat() if upd else "")}">'
        f" · 更新于 {esc(fmt_ago(upd, now))}</span>"
    )

    chg = cur.get("short_chg_pct")
    upjump = bool(cur.get("short_upjump_recent"))
    a_pill = (
        '<span class="pill warn"><span class="dot"></span>命中</span>'
        if upjump
        else '<span class="pill"><span class="dot"></span>未命中</span>'
    )
    a_w = min(100.0, max(0.0, (chg or 0.0) / SHORT_JUMP_PCT * 100))
    try:
        settle_d = date.fromisoformat(str(cur.get("short_settlement")))
    except (TypeError, ValueError):
        settle_d = None
    leg_a = (
        '<div class="leg"><div class="leg-h">'
        '<svg class="ic" aria-hidden="true"><use href="#i-scales"/></svg>'
        f"腿 A · 空头利益跳变{a_pill}</div>"
        f'<div class="val num">{f"{chg:+.2f}%" if chg is not None else "—"}</div>'
        f'<div class="ref">阈值 +{SHORT_JUMP_PCT:.1f}% · FINRA 双周口径 · '
        f'结算 {esc(cur.get("short_settlement") or "—")} · 回看 {LOOKBACK_BDAYS}bd '
        f'{age_badge(settle_d, now, 18)}</div>'
        f'<div class="gauge{" hit" if upjump else ""}">'
        f'<i style="width:{a_w:.1f}%"></i><span class="th"></span></div></div>'
    )

    mc, thr = cur.get("musk_count"), cur.get("dense_thr")
    if state == "CALIBRATING":
        b_pill = '<span class="pill warn"><span class="dot"></span>标定中</span>'
        b_ref = f"阈值待标定（基线分位映射） · 口径日 {esc(cur.get('musk_count_day') or '—')}"
    elif thr and mc is not None and mc >= thr:
        b_pill = '<span class="pill warn"><span class="dot"></span>命中</span>'
        b_ref = f"密集阈值 {_fmt_count(thr)} 帖/日 · 口径日 {esc(cur.get('musk_count_day') or '—')}"
    else:
        b_pill = '<span class="pill"><span class="dot"></span>未命中</span>'
        b_ref = (
            f"密集阈值 {_fmt_count(thr)} 帖/日 · " if thr else "阈值未生效 · "
        ) + f"口径日 {esc(cur.get('musk_count_day') or '—')}"
    try:
        musk_d = date.fromisoformat(str(cur.get("musk_count_day")))
    except (TypeError, ValueError):
        musk_d = None
    leg_b = (
        '<div class="leg"><div class="leg-h">'
        '<svg class="ic" aria-hidden="true"><use href="#i-mega"/></svg>'
        f"腿 B · Musk 发帖密度{b_pill}</div>"
        f'<div class="val num">{_fmt_count(mc)}<span class="unit"> 帖/日</span></div>'
        f'<div class="ref">{b_ref} {age_badge(musk_d, now, 2)}</div>'
        f"{_sparks_html(cur)}</div>"
    )

    legnote = (
        '<div class="legnote"><svg class="ic" aria-hidden="true">'
        '<use href="#i-shield"/></svg><span>'
        f"判定冻结参数：两腿同窗命中（LOOKBACK {LOOKBACK_BDAYS}bd）且持续"
        f"（PERSIST {PERSIST_BDAYS}bd）→ RISK_OFF；假想单成本线 {COST_LINE:+.2%}。"
        "标定期读数灰示，不参与判定。</span></div>"
        # 证据等级行（P0-6）：常驻，不随状态变化
        '<div class="legnote evid"><svg class="ic" aria-hidden="true">'
        f'<use href="#i-pulse"/></svg><span>{DET_EVIDENCE}</span></div>'
    )

    # -- 右栏：三灯组 + 最近切换 + 假想单判分小结
    sw_rows = "".join(
        f'<div class="kv"><span class="k num">{fmt_local(parse_ts(s["event_time_utc"]))}</span>'
        f'<span class="v" title="{esc(s["title"])}">{esc(s.get("state") or "?")}</span></div>'
        for s in data["switches"][:5]
    ) or '<div class="kv"><span class="k">—</span><span class="v">暂无切换记录</span></div>'
    right = (
        _lamps_html(state)
        + '<div class="sideblock"><div class="bt">'
        '<svg class="ic" aria-hidden="true"><use href="#i-lamp"/></svg>最近状态切换</div>'
        + sw_rows
        + "</div>"
        + '<div class="sideblock"><div class="bt">'
        '<svg class="ic" aria-hidden="true"><use href="#i-pulse"/></svg>假想单判分</div>'
        + _hypo_summary(data["trades"])
        + "</div>"
    )

    footnote = (
        f"冻结规则 N3-H：Musk 密集发帖（act=次交易日）× 回看 {LOOKBACK_BDAYS} 交易日内"
        f"空头 change_pct ≥ +{SHORT_JUMP_PCT:.0f}% 发布 → RISK_OFF 持续 "
        f"F{PERSIST_BDAYS}（{PERSIST_BDAYS} 交易日，重叠触发顺延）。"
        f"标定期 {CALIB_BDAYS} 交易日只累积基线，不出信号。假想推演，不碰真钱。"
        "表盘进度语义：标定期 = 基线累积（设计稿方案）；RISK_ON = 放风腿密度占比、"
        "避险期 = F 窗行进度（设计稿未预留正式期方案，自定口径）。"
        "RISK_ON 读作「未见空头知情型风险」——本探测器仅覆盖此一类下跌，"
        "不是持仓建议；广谱回撤防线见「今日合议」S2 读数。"
    )
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>态势总览<span class="h-sub">N3-H 冻结规则 · 探测器状态机 · 两腿读数 · 前向虚拟推演</span></h2>
  <div class="card hero">
    <div class="ringwrap">
      {ring}
      <div class="ring-cap">{esc(ring_cap)}</div>
    </div>
    <div>
      <p class="statement st-{cls}">{stmt}</p>
      <div class="since num">{since}</div>
      <div class="legs">{leg_a}{leg_b}</div>
      {legnote}
    </div>
    <div>{right}</div>
  </div>
  <div class="card det-more">
    {_det_trades_html(data["trades"], now)}
    <p class="footnote">{footnote}</p>
  </div>
</section>"""


TIER_ICON = {  # 层级 → sprite symbol id（雷达/盾牌/天平/扩音器，设计稿 §四）
    "T0": "i-radar", "T1": "i-shield", "T2": "i-scales", "T3": "i-mega",
}


def tier_icon(tier: str | None, cls: str = "t-ic") -> str:
    sid = TIER_ICON.get(tier or "", "i-mega")
    return (
        f'<svg class="{cls}" aria-hidden="true" focusable="false">'
        f'<use href="#{sid}"/></svg>'
    )


def _src_card(h: dict, now: datetime, show_weight: bool = True) -> str:
    last = h["last"]
    if last is None:
        status_cls, status_txt, card_cls = "off", "未启用", " offline"
        last_t = None
        detail = ""
    else:
        last_t = parse_ts(last["poll_time_utc"])
        age = (now - last_t).total_seconds() if last_t else None
        interval = h["poll_interval_s"] or 300
        if last["ok"] == 0:
            status_cls, status_txt, card_cls = "crit", "失败", " crit"
            detail = esc((last.get("error") or "")[:90])
        elif age is not None and age > max(2 * interval, STALE_S):
            status_cls, status_txt, card_cls = "warn", "迟滞", ""
            detail = f"距上次轮询 {esc(fmt_dur(age))}"
        else:
            status_cls, status_txt, card_cls = "good", "正常", ""
            detail = f"抓 {last['n_seen']} / 新 {last['n_new']}"
    w = h.get("weight")
    if w is None or not show_weight:
        weight_html = ""
    else:
        t = h["tier"] if h["tier"] in TIERS else "T3"
        weight_html = (
            '<span class="wt-cell" title="人工先验权重（身位分初值，未经四维评分）">'
            f'<span class="wt-bar"><span class="wtf-{t[1]}" '
            f'style="width:{min(1.0, max(0.0, w)) * 100:.0f}%"></span></span>'
            f'<span class="num">{w:.1f}</span></span>'
        )
    streak = h["fail_streak"]
    streak_html = (
        f'<span class="crit-text">连败 <span class="num">{streak}</span></span>'
        if streak
        else ""
    )
    poll_html = (
        f'<span class="num">{fmt_local(last_t)}</span>'
        f'<span class="rel" data-iso="{esc(last_t.isoformat() if last_t else "")}">'
        f"{esc(fmt_ago(last_t, now))}</span>"
        if last_t
        else "—"
    )
    return (
        f'<div class="src-card{card_cls}">'
        f'<div class="sc-top"><strong>{esc(h["source_id"])}</strong>'
        f'<span class="pill sm {status_cls}"><span class="dot"></span>{status_txt}</span></div>'
        f'<div class="sc-name" title="{esc(h["name"])}">{esc(h["name"])}</div>'
        f'<div class="sc-met">{weight_html}'
        f'<span>事件 <span class="num">{h["n_events"]:,}</span></span>{streak_html}</div>'
        f'<div class="sc-sub">{poll_html}' + (f" · {detail}" if detail else "") + "</div>"
        "</div>"
    )


# 衍生信号（detector 等）：是"结论"不是"渠道"，移出层级矩阵单列（P0-4）
DERIVED_SOURCE_IDS = {"detector"}


def render_health(rows: list[dict], now: datetime) -> str:
    """④ 渠道矩阵：按 T0-T3 分组的卡片组 + 衍生信号单列小组。"""
    if not rows:
        return ""
    derived = [h for h in rows if h["source_id"] in DERIVED_SOURCE_IDS]
    channels = [h for h in rows if h["source_id"] not in DERIVED_SOURCE_IDS]
    groups: dict[str, list[dict]] = {}
    for h in channels:  # rows 已按 tier→权重排好序
        t = h["tier"] if h["tier"] in TIERS else "T?"
        groups.setdefault(t, []).append(h)
    blocks = []
    for t in [*TIERS, "T?"]:
        if t not in groups:
            continue
        cards = "".join(_src_card(h, now) for h in groups[t])
        label = TIER_LABEL.get(t, "未分层")
        blocks.append(
            f'<div class="tier-group"><div class="tier-head">{tier_icon(t)}'
            f"{tier_badge(t if t in TIERS else None)}"
            f"<h3>{esc(label)}</h3>"
            f'<span class="h-sub">{len(groups[t])} 渠道</span></div>'
            f'<div class="src-cards">{cards}</div></div>'
        )
    if derived:
        cards = "".join(_src_card(h, now, show_weight=False) for h in derived)
        blocks.append(
            f'<div class="tier-group"><div class="tier-head">{tier_icon("T0")}'
            "<h3>衍生信号</h3>"
            '<span class="h-sub">探测器是结论不是渠道——健康状态在此监控，'
            "不与情报源同池排序</span></div>"
            f'<div class="src-cards">{cards}</div></div>'
        )
    foot = (
        '<p class="footnote">权重口径（P0-5 如实标注）：列示权重为建渠道时'
        "人工拍定的<b>身位分初值</b>（人工先验），<b>不是</b>算出来的四维评分——"
        "四维评分（身位/事实/时效/意外）尚未实现，见 docs/intel-framework.md 第二节。"
        "分层依据 intel-framework 第一节 + strategy-lab N2：T0 布局痕迹（13D/G、Form 144、"
        "空头利益、暗池、期权快照；Polymarket 预测市场赔率亦归此层，属资金布局痕迹而非"
        "法定披露，口径最弱）。</p>"
    )
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>渠道健康<span class="h-sub">{len(channels)} 渠道 + {len(derived)} 衍生信号 · 按层级分组 · 组内按权重排序 · 权重 = 人工先验（未经四维评分）</span></h2>
  <div class="card">{"".join(blocks)}</div>
  {foot}
</section>"""


def render_timeline(rows: list[dict], now: datetime) -> str:
    if not rows:
        return ""
    items = []
    for e in rows:
        obs = parse_ts(e["observed_time_utc"])
        title = e.get("title") or e.get("type") or "(无标题)"
        url = e.get("url")
        title_html = (
            f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>'
            if url
            else esc(title)
        )
        badge = (
            '<span class="tier tier-d" title="衍生信号（探测器结论，非情报渠道）">信号</span>'
            if e["source_id"] in DERIVED_SOURCE_IDS
            else tier_badge(e.get("tier"))
        )
        items.append(
            '<li class="ev">'
            f'<span class="ev-time num" title="观察时刻（本机时区）">{fmt_local(obs)}</span>'
            f"{badge}"
            f'<span class="ev-src">{esc(e["source_id"])}</span>'
            f'<span class="ev-title">{title_html}</span>'
            f'<span class="ev-type micro">{esc(e.get("type") or "")}</span>'
            "</li>"
        )
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>最新情报流<span class="h-sub">最近 {len(rows)} 条 · 按观察时刻倒序</span></h2>
  <div class="card">
    <ol class="feed">{"".join(items)}</ol>
  </div>
</section>"""


# 时延对数轴上限：7 天（分钟计）
_LAT_MAX_MIN = 7 * 24 * 60.0


def _fmt_lat(s: float | None) -> str:
    """紧凑时延格式（等宽栏用）：26h / 3.9d 风格。"""
    if s is None:
        return "—"
    neg = "-" if s < 0 else ""
    s = abs(s)
    if s < 90:
        return f"{neg}{s:.0f}s"
    if s < 5400:
        return f"{neg}{s / 60:.0f}m"
    if s < 2 * 86400:
        return f"{neg}{s / 3600:.1f}h"
    return f"{neg}{s / 86400:.1f}d"


def _lat_pos(seconds: float | None) -> float | None:
    """秒 → 对数轴位置 0..1（1 分钟 → 7 天）。"""
    if seconds is None:
        return None
    m = max(1.0, seconds / 60.0)
    return min(1.0, math.log10(m) / math.log10(_LAT_MAX_MIN))


def _latency_card(rows: list[dict], sources: dict) -> str:
    """入库时延卡：每渠道 p50 → p90 对数轴标尺（设计稿 05 右卡）。"""
    ticks = "".join(
        f'<span class="tk" style="left:calc(150px + (100% - 150px - 120px)*{_lat_pos(m * 60):.4f})">{lab}</span>'
        for m, lab in ((1, "1m"), (60, "1h"), (1440, "1d"), (10080, "7d"))
    )
    lrows = []
    for r in rows:
        sid = r["source_id"]
        src = sources.get(sid, {})
        t = src.get("tier") if src.get("tier") in TIERS else "T3"
        notes = []
        if (r.get("lag_min_s") or 0) < 0:
            notes.append("日历预告（lag 为负 = 事件在未来，正常）")
        if r.get("n_backfill"):
            notes.append(f"另有回填 {r['n_backfill']} 条不计入分位数")
        a, b = _lat_pos(r["p50"]), _lat_pos(r["p90"])
        col = f"var(--tier{t[1]})"
        if a is not None and b is not None:
            rail = (
                '<span class="rail"><span class="base"></span>'
                f'<span class="range" style="left:{a * 100:.1f}%;width:{max(0.8, (b - a) * 100):.1f}%;'
                f'background:{col};opacity:.45"></span>'
                f'<span class="p50" style="left:{a * 100:.1f}%;background:{col}"></span>'
                f'<span class="p50 p90" style="left:{b * 100:.1f}%;background:{col}"></span></span>'
            )
        else:
            rail = '<span class="rail"><span class="base"></span></span>'
        if r.get("n_steady"):
            num_txt = f"{_fmt_lat(r['p50'])} / {_fmt_lat(r['p90'])}"
        else:
            num_txt = "仅回填"
        mint = fmt_dur(r["lag_min_s"]) if r.get("lag_min_s") is not None else "—"
        title = (
            f"稳态 n={r.get('n_steady', 0):,}（回填 {r.get('n_backfill', 0):,} 不计入）"
            f" · 最小 {mint}" + ("；" + "；".join(notes) if notes else "")
        )
        lrows.append(
            f'<div class="lrow" title="{esc(title)}">'
            f'<span class="lbl">{tier_badge(src.get("tier"))}'
            f'<span class="sid">{esc(sid)}</span></span>'
            f'{rail}<span class="n">{esc(num_txt)}</span></div>'
        )
    return (
        '<div class="card duo-card"><h3><svg class="ic" aria-hidden="true">'
        '<use href="#i-clock"/></svg>入库时延 p50 → p90（稳态口径）</h3>'
        '<div class="cap">observed − event · 对数轴 1m → 7d · 仅首采日之后的增量事件'
        '（回填不计入）· 悬停看 n / 最小 / 备注</div>'
        f'<div class="lat-scale">{ticks}</div>{"".join(lrows)}'
        "</div>"
    )


def _counts_card(counts: dict[str, dict]) -> str:
    """事件计数卡：层级双条（深 = 今日，浅 = 近 7 日；设计稿 05 左卡）。"""
    tiers = [t for t in TIERS if t in counts]
    maxc = max(
        [c["today"] for c in counts.values()] + [c["week"] for c in counts.values()]
    ) or 1
    brows = []
    for t in tiers:
        c = counts.get(t, {})
        col = f"var(--tier{t[1]})"
        brows.append(
            f'<div class="brow">{tier_badge(t)}'
            '<span class="bars">'
            f'<i class="b" style="width:{max(c.get("today", 0) / maxc * 100, 0.7):.1f}%;background:{col}"></i>'
            f'<i class="b dim" style="width:{max(c.get("week", 0) / maxc * 100, 0.7):.1f}%;background:{col}"></i>'
            "</span>"
            f'<span class="n">{c.get("today", 0):,} / {c.get("week", 0):,}</span></div>'
        )
    return (
        '<div class="card duo-card"><h3><svg class="ic" aria-hidden="true">'
        '<use href="#i-pulse"/></svg>事件计数</h3>'
        '<div class="cap">深色 = 今日（ET 交易日） · 浅色 = 近 7 日 · observed 口径 · 同标尺</div>'
        + "".join(brows)
        + "</div>"
    )


def render_counts_latency(
    counts: dict[str, dict], lat_rows: list[dict], sources: dict
) -> str:
    """⑤ 计数与时延（设计稿 05 双栏；任一侧数据缺失则单栏降级）。"""
    left = _counts_card(counts) if counts else ""
    right = _latency_card(lat_rows, sources) if lat_rows else ""
    if not (left or right):
        return ""
    foot = (
        '<p class="footnote">时延口径注：p50/p90 为稳态口径——只统计首采日之后 '
        "observed 的增量事件；首采回填（老事件当天集中入库）不参与分位数，"
        "条数见悬停备注。显示「仅回填」的渠道尚无稳态样本，等增量事件积累。</p>"
    )
    duo_cls = "duo" if (left and right) else "duo one"
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>计数与时延<span class="h-sub">按层级计数 · 按渠道时延</span></h2>
  <div class="{duo_cls}">{left}{right}</div>
  {foot if right else ""}
</section>"""


# ---------------------------------------------------------------- page shell

# 手绘图标 sprite（设计稿 §四：24 视框 / stroke 1.6 / 圆帽圆角）
# 雷达=探测器 盾牌=避险 天平=两腿判定 扩音器=舆情腿 信号灯=状态机
# 脉冲=情报流 时钟=时延；g-t0..3 = 层级形状（色深+形状双编码）；
# hatch45 = 避险段 45° 影线图案（全页共用）。
_ICON_SPRITE = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="i-radar" viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="8.6"/>
  <circle cx="12" cy="12" r="4.6" opacity=".45"/>
  <path d="M12 12 L18.1 5.9"/>
  <circle cx="18.1" cy="5.9" r="1.3" fill="currentColor" stroke="none"/>
</symbol>
<symbol id="i-shield" viewBox="0 0 24 24">
  <path d="M12 3.4 L19 6.1 V11.6 C19 16.4 16.2 19.4 12 20.9 C7.8 19.4 5 16.4 5 11.6 V6.1 Z"/>
  <path d="M9 11.8 L11.2 14.2 L15.2 9.6"/>
</symbol>
<symbol id="i-scales" viewBox="0 0 24 24">
  <path d="M12 4.5 V19.5 M8.6 19.5 H15.4 M4.5 7 H19.5"/>
  <path d="M4.5 7 L2.4 12.2 A2.35 2.35 0 0 0 6.6 12.2 Z"/>
  <path d="M19.5 7 L17.4 12.2 A2.35 2.35 0 0 0 21.6 12.2 Z"/>
</symbol>
<symbol id="i-mega" viewBox="0 0 24 24">
  <path d="M4 10.2 H7.2 L16.4 5.2 V18.8 L7.2 13.8 H4 Z"/>
  <path d="M8.4 14 V17.6"/>
  <path d="M19.2 9.4 A3.7 3.7 0 0 1 19.2 14.6"/>
</symbol>
<symbol id="i-lamp" viewBox="0 0 24 24">
  <rect x="8.6" y="3.2" width="6.8" height="17.6" rx="3.4"/>
  <circle cx="12" cy="7.4" r="1.5"/>
  <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/>
  <circle cx="12" cy="16.6" r="1.5"/>
</symbol>
<symbol id="i-pulse" viewBox="0 0 24 24">
  <path d="M3 12 H7.4 L9.8 6.4 L13.8 17.6 L16.2 12 H21"/>
</symbol>
<symbol id="i-clock" viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="8.6"/>
  <path d="M12 7.4 V12 L15.2 14.2"/>
</symbol>
<symbol id="g-t0" viewBox="0 0 10 10"><path d="M5 .6 9.4 5 5 9.4 .6 5Z" fill="currentColor"/></symbol>
<symbol id="g-t1" viewBox="0 0 10 10"><path d="M5 .9 9.4 9.1 H.6Z" fill="currentColor"/></symbol>
<symbol id="g-t2" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4.2" fill="currentColor"/></symbol>
<symbol id="g-t3" viewBox="0 0 10 10"><circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.6"/></symbol>
<pattern id="hatch45" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
  <line x1="0" y1="0" x2="0" y2="7" stroke="var(--crit)" stroke-width="1" opacity=".22"/>
</pattern>
</defs></svg>"""

# 亮色 token 组（设计稿 §一）：写两处——[data-theme=light] 属性覆盖（页内按钮）
# 与 prefers-color-scheme 媒体回退（无 JS / 未显式选择时跟随系统）。
_LIGHT_TOKENS = """
  color-scheme: light;
  --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ec;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --border:rgba(11,11,11,.10); --grid:#e1e0d9; --baseline:#c3c2b7;
  --good:#0ca30c; --good-text:#006300; --good-wash:rgba(12,163,12,.08);
  --warn:#fab219; --warn-text:#7a5200; --warn-wash:rgba(250,178,25,.14);
  --crit:#d03b3b; --crit-text:#b02a2a; --crit-wash:rgba(208,59,59,.07);
  --tier0:#1c5cab; --tier0-ink:#ffffff;
  --tier1:#2a78d6; --tier1-ink:#ffffff;
  --tier2:#5598e7; --tier2-ink:#0b0b0b;
  --tier3:#86b6ef; --tier3-ink:#0b0b0b;
  --m-insider:#2a78d6; --m-pit:#008300; --m-fake:#c98500; --m-hypo:#e87ba4;
  --link:#1c5cab;
"""

_CSS = """
/* ===== tokens（设计稿色板：暗色默认，亮色经 data-theme 或系统偏好） ===== */
:root {
  color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#232322;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --border:rgba(255,255,255,.10); --grid:#2c2c2a; --baseline:#383835;
  --good:#0ca30c; --good-text:#0ca30c; --good-wash:rgba(12,163,12,.16);
  --warn:#fab219; --warn-text:#fab219; --warn-wash:rgba(250,178,25,.13);
  --crit:#d03b3b; --crit-text:#e66767; --crit-wash:rgba(208,59,59,.14);
  --off:#898781;
  --tier0:#9ec5f4; --tier0-ink:#0b0b0b;
  --tier1:#5598e7; --tier1-ink:#0b0b0b;
  --tier2:#256abf; --tier2-ink:#ffffff;
  --tier3:#184f95; --tier3-ink:#ffffff;
  --m-insider:#3987e5; --m-pit:#008300; --m-fake:#c98500; --m-hypo:#d55181;
  --link:#86b6ef;
  --font-serif:"Songti SC","STSong","Noto Serif CJK SC","Source Han Serif SC",serif;
  --font-sans:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
:root[data-theme="light"] { __LIGHT__ }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) { __LIGHT__ }
}
/* ===== base（字阶：rem 基 14px；数据一律等宽 + tabular-nums） ===== */
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font: 14px/1.6 var(--font-sans);
  -webkit-font-smoothing: antialiased;
}
main, .topbar-in { max-width: 1200px; margin: 0 auto; padding: 0 24px; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: var(--font-mono); font-size: 0.92em; }
.num, td.num, th.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.small { font-size: 12px; }
.micro { font-family: var(--font-mono); font-size: 11px; letter-spacing: .08em; }
.muted { color: var(--muted); }
.good-text { color: var(--good-text); }
.warn-text { color: var(--warn-text); }
.crit-text { color: var(--crit-text); }
svg.ic { width: 18px; height: 18px; flex: none; stroke: currentColor; fill: none;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }

/* ===== top bar ===== */
.topbar { border-bottom: 1px solid var(--border); background: var(--surface); }
.topbar-in { display: flex; align-items: center; gap: 16px;
  padding-top: 14px; padding-bottom: 14px; flex-wrap: wrap; }
.topbar-in:empty, .topbar-in:has(> [hidden]:only-child) { display: none; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand > svg.ic { width: 26px; height: 26px; color: var(--warn); }
h1 { font: 600 21px/1.3 var(--font-serif); margin: 0; letter-spacing: .04em; }
.brand .sub { font-family: var(--font-mono); font-size: 11px; letter-spacing: .22em;
  color: var(--muted); text-transform: uppercase; margin-top: 1px; }
.topmeta { margin-left: auto; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
.stamp { text-align: right; }
.stamp .k { font-size: 11px; color: var(--muted); letter-spacing: .06em; }
.stamp .v { font-family: var(--font-mono); font-size: 12.5px;
  font-variant-numeric: tabular-nums; color: var(--ink-2); }
.stamp .rel::before { content: " · "; }
.stamp.crit-text .k, .stamp.crit-text .v { color: var(--crit-text); }
.themebtn { font: 12px/1 var(--font-mono); letter-spacing: .1em; color: var(--ink-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px;
  padding: 7px 12px; cursor: pointer; }
.themebtn:hover { color: var(--ink); border-color: var(--muted); }
.banner { flex: 1; padding: 8px 12px; border: 1px solid var(--crit);
  border-radius: 4px; background: var(--crit-wash); color: var(--crit-text);
  font-size: 13px; }

/* ===== sections（宋体衬线板块题 + 等宽序号 + 延展线） ===== */
section { margin-top: 48px; }
h2 { display: flex; align-items: baseline; gap: 10px; margin: 0 0 14px;
  font: 600 17px/1.4 var(--font-serif); letter-spacing: .05em; flex-wrap: wrap; }
h2 .sec-no { font: 400 11px var(--font-mono); letter-spacing: .18em; color: var(--muted); }
h2 .h-sub { font: 400 12px var(--font-sans); color: var(--muted); letter-spacing: 0; }
h2::after { content: ""; flex: 1; border-top: 1px solid var(--border);
  align-self: center; margin-left: 6px; min-width: 40px; }
h3 { font: 600 13.5px var(--font-sans); margin: 0 0 6px; color: var(--ink-2); }
.h-sub { font-size: 12px; font-weight: 400; color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }

/* ===== pills ===== */
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
  font-weight: 600; font-family: var(--font-mono); letter-spacing: .06em;
  padding: 2px 9px; border: 1px solid var(--border); border-radius: 3px;
  color: var(--ink-2); white-space: nowrap; }
.pill.sm { padding: 1px 7px; }
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--off); flex: none; }
.pill.good { color: var(--good-text); } .pill.good .dot { background: var(--good); }
.pill.warn { color: var(--warn-text); } .pill.warn .dot { background: var(--warn); }
.pill.crit { color: var(--crit-text); } .pill.crit .dot { background: var(--crit); }

/* ===== 层级徽章（色深 + 形状双编码） ===== */
.tier { display: inline-flex; align-items: center; gap: 5px; font-family: var(--font-mono);
  font-size: 11px; font-weight: 700; letter-spacing: .04em; padding: 1px 7px 1px 5px;
  border-radius: 3px; border: 1px solid transparent; font-variant-numeric: tabular-nums; }
.tier svg { width: 9px; height: 9px; flex: none; }
.tier-0 { background: var(--tier0); color: var(--tier0-ink); }
.tier-1 { background: var(--tier1); color: var(--tier1-ink); }
.tier-2 { background: var(--tier2); color: var(--tier2-ink); }
.tier-3 { background: var(--tier3); color: var(--tier3-ink); }
.tier-d { background: var(--surface-2); color: var(--warn-text);
  border-color: var(--warn); }

/* ===== 顶栏现价 + 数据龄徽章（P0-2 全站规范） ===== */
.px { display: inline-flex; align-items: baseline; gap: 7px;
  font-family: var(--font-mono); font-size: 13px; color: var(--ink-2);
  padding-left: 14px; border-left: 1px solid var(--border); }
.px b { font-size: 16px; color: var(--ink); font-weight: 600; }
.px-lab { font-size: 10.5px; color: var(--muted); letter-spacing: .04em; }
.age { display: inline-block; font-family: var(--font-mono); font-size: 10px;
  letter-spacing: .05em; padding: 0 6px; border-radius: 3px;
  border: 1px solid var(--border); color: var(--muted); white-space: nowrap; }
.age.warn { color: var(--warn-text); border-color: var(--warn);
  background: var(--warn-wash); }
.age.crit { color: var(--crit-text); border-color: var(--crit);
  background: var(--crit-wash); }

/* ===== 01 今日合议（决策卡） ===== */
.cx { padding: 16px 18px 8px; }
.cx-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1080px){ .cx-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px){ .cx-grid { grid-template-columns: 1fr; } }
.cx-cell { background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 14px; min-width: 0; }
.cx-cell.crit { background: var(--crit-wash); border-color: var(--crit); }
.cx-k { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  font-size: 11.5px; color: var(--muted); letter-spacing: .04em; }
.cx-k .pill { margin-left: auto; }
.cx-v { font: 600 21px/1.3 var(--font-mono); font-variant-numeric: tabular-nums;
  margin: 6px 0 3px; color: var(--ink); }
.cx-v .good-text, .cx-v .crit-text { font-size: 14px; }
.cx-ref { font-size: 11.5px; color: var(--muted); line-height: 1.55; }
.cx-verdict { margin: 14px 2px 6px; }

/* ===== 01 态势总览 hero ===== */
.hero { display: grid; grid-template-columns: 264px minmax(0,1fr) 300px; gap: 0;
  overflow: hidden; }
.hero > div { padding: 20px; min-width: 0; }
.hero > div + div { border-left: 1px solid var(--border); }
@media (max-width: 980px){ .hero { grid-template-columns: 1fr; }
  .hero > div + div { border-left: none; border-top: 1px solid var(--border); } }

.ringwrap { display: flex; flex-direction: column; align-items: center;
  justify-content: center; }
.ring { width: 224px; height: 224px; }
.ring .dial { stroke: var(--baseline); }
.ring .track { stroke: var(--grid); }
.ring text { font-family: var(--font-mono); }
.ring .t-state { font: 600 34px var(--font-serif); fill: var(--ink); letter-spacing: .08em; }
.ring .t-code { font-size: 11px; letter-spacing: .3em; }
.ring .t-prog { font-size: 13px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
.ring .t-sub  { font-size: 10px; fill: var(--muted); letter-spacing: .05em; }
.ring .wash { animation: breathe 4.5s ease-in-out infinite; transform-origin: 112px 112px; }
@keyframes breathe { 0%,100%{opacity:.55;} 50%{opacity:1;} }
@media (prefers-reduced-motion: reduce){ .ring .wash { animation: none; } }
.ring.warn .prog { stroke: var(--warn); } .ring.warn .wash { fill: var(--warn-wash); }
.ring.warn .t-code { fill: var(--warn-text); }
.ring.good .prog { stroke: var(--good); } .ring.good .wash { fill: var(--good-wash); }
.ring.good .t-code { fill: var(--good-text); }
.ring.crit .prog { stroke: var(--crit); } .ring.crit .wash { fill: var(--crit-wash); }
.ring.crit .t-code { fill: var(--crit-text); }
.ring.off .prog { stroke: var(--muted); } .ring.off .wash { fill: var(--surface-2); }
.ring.off .t-code { fill: var(--muted); }
.ring-cap { margin-top: 10px; font-size: 11px; color: var(--muted);
  letter-spacing: .06em; font-family: var(--font-mono); }

.statement { font: 400 15px/1.7 var(--font-serif); color: var(--ink-2); margin: 0 0 4px; }
.statement b { font-weight: 600; }
.st-warn b { color: var(--warn-text); } .st-good b { color: var(--good-text); }
.st-crit b { color: var(--crit-text); } .st-off b { color: var(--muted); }
.since { font-family: var(--font-mono); font-size: 11.5px; color: var(--muted); }
.legs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
@media (max-width: 620px){ .legs { grid-template-columns: 1fr; } }
.leg { background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 16px; min-width: 0; }
.leg-h { display: flex; align-items: center; gap: 8px; font-size: 12px;
  color: var(--ink-2); flex-wrap: wrap; }
.leg-h svg.ic { width: 16px; height: 16px; color: var(--muted); }
.leg-h .pill { margin-left: auto; }
.leg .val { font: 600 22px/1.2 var(--font-mono); font-variant-numeric: tabular-nums;
  margin: 8px 0 2px; }
.leg .val .unit { font-size: 12px; font-weight: 400; color: var(--muted); }
.leg .ref { font-size: 11.5px; color: var(--muted); font-family: var(--font-mono); }
.gauge { height: 5px; background: var(--grid); border-radius: 2px; margin-top: 10px;
  position: relative; overflow: visible; }
.gauge > i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--muted);
  border-radius: 2px; }
.gauge.hit > i { background: var(--warn); }
.gauge .th { position: absolute; right: 0; top: -3px; bottom: -3px; width: 2px;
  background: var(--crit); }
.sparks { display: flex; align-items: flex-end; gap: 2px; height: 34px; margin-top: 10px;
  border-bottom: 1px solid var(--baseline); position: relative; }
.sparks > i { flex: 1; max-width: 18px; background: var(--grid); min-height: 2px; }
.sparks > i.hi { background: var(--warn); }
.sparks .p95 { position: absolute; left: 0; right: 0; border-top: 1px dashed var(--muted); }
.legnote { margin-top: 14px; font-size: 12px; color: var(--muted); display: flex;
  gap: 8px; align-items: flex-start; }
.legnote svg.ic { width: 15px; height: 15px; margin-top: 2px; }
.legnote.evid { margin-top: 8px; padding: 7px 10px; border: 1px dashed var(--border);
  border-radius: 5px; background: var(--surface-2); }
.legnote.evid svg.ic { color: var(--warn); }

.lamps { display: flex; flex-direction: column; gap: 2px; }
.lamp { display: flex; align-items: center; gap: 12px; padding: 9px 6px; border-radius: 6px; }
.lamp .bulb { width: 16px; height: 16px; border-radius: 50%; flex: none;
  border: 1.6px solid var(--muted); background: transparent; }
.lamp .name { font-family: var(--font-mono); font-size: 11px; letter-spacing: .14em;
  color: var(--muted); width: 88px; }
.lamp .zh { font-size: 12.5px; color: var(--muted); }
.lamp.g .bulb { border-color: var(--good); }
.lamp.w .bulb { border-color: var(--warn); }
.lamp.c .bulb { border-color: var(--crit); }
.lamp.on .zh { color: var(--ink); font-weight: 600; }
.lamp.on.w { background: var(--warn-wash); }
.lamp.on.w .bulb { background: var(--warn); border-color: var(--warn);
  box-shadow: 0 0 0 3px var(--warn-wash), 0 0 14px 1px var(--warn-wash); }
.lamp.on.w .name { color: var(--warn-text); }
.lamp.on.g { background: var(--good-wash); }
.lamp.on.g .bulb { background: var(--good); border-color: var(--good);
  box-shadow: 0 0 0 3px var(--good-wash), 0 0 14px 1px var(--good-wash); }
.lamp.on.g .name { color: var(--good-text); }
.lamp.on.c { background: var(--crit-wash); }
.lamp.on.c .bulb { background: var(--crit); border-color: var(--crit);
  box-shadow: 0 0 0 3px var(--crit-wash), 0 0 14px 1px var(--crit-wash); }
.lamp.on.c .name { color: var(--crit-text); }
.sideblock { margin-top: 16px; border-top: 1px solid var(--border); padding-top: 12px; }
.sideblock .bt { display: flex; gap: 8px; align-items: center; font-size: 12px;
  color: var(--ink-2); margin-bottom: 8px; }
.sideblock .bt svg.ic { width: 15px; height: 15px; color: var(--muted); }
.kv { display: flex; justify-content: space-between; gap: 12px; font-size: 12px;
  padding: 3px 0; }
.kv .k { color: var(--muted); }
.kv .v { font-family: var(--font-mono); font-variant-numeric: tabular-nums;
  color: var(--ink-2); text-align: right; }
.det-more { margin-top: 12px; }
.det-sub h3 { margin: 12px 16px 4px; }
.det-table { min-width: 640px; }

/* ===== 表格 ===== */
.scroll-x { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 720px; }
th, td { text-align: left; padding: 8px 14px; white-space: nowrap; }
thead th { font-size: 11px; font-weight: 600; color: var(--muted); letter-spacing: .08em;
  font-family: var(--font-mono); border-bottom: 1px solid var(--grid); }
tbody tr { border-bottom: 1px solid var(--grid); }
tbody tr:last-child { border-bottom: none; }
tr.row-crit { background: var(--crit-wash); }
td .sub { display: block; font-size: 11px; color: var(--muted);
  font-family: var(--font-mono); white-space: nowrap; }
td.detail { font-size: 12px; color: var(--ink-2); max-width: 340px;
  overflow: hidden; text-overflow: ellipsis; }

/* ===== 02 战场走势 ===== */
.sym-tabs { display: flex; align-items: center; gap: 6px; margin: 0 0 10px; }
.sym-tab { font-family: var(--font-mono); font-size: 12px; font-weight: 700;
  letter-spacing: .06em; padding: 4px 16px; border: 1px solid var(--border);
  border-radius: 4px; color: var(--muted); background: transparent; }
.sym-tab.act { background: var(--surface-2); color: var(--ink);
  border-color: var(--muted); }
.sym-hint { font-family: var(--font-mono); font-size: 11px; color: var(--muted);
  margin-left: 6px; letter-spacing: .06em; }
.chartcard { padding: 16px 18px 10px; }
.legendrow { display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  margin-bottom: 12px; padding: 0; }
.legendrow .spacer { flex: 1; }
.tv-ranges { display: inline-flex; border: 1px solid var(--border);
  border-radius: 4px; overflow: hidden; flex: none; }
.tv-rb { appearance: none; border: none; background: transparent; cursor: pointer;
  color: var(--ink-2); font: 600 11px var(--font-mono); letter-spacing: .08em;
  padding: 6px 14px; border-right: 1px solid var(--border); }
.tv-rb:last-child { border-right: none; }
.tv-rb.act { background: var(--surface-2); color: var(--ink); }
.tv-layers { display: flex; gap: 8px; flex-wrap: wrap; }
.lg { display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
  user-select: none; appearance: none; font: inherit; border: 1px solid var(--border);
  border-radius: 4px; padding: 4px 10px; font-size: 12px; color: var(--ink-2);
  background: var(--surface-2); }
.lg svg { width: 20px; height: 14px; flex: none; }
.lg .ct { font-family: var(--font-mono); font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums; }
.lg.off { opacity: .38; border-style: dashed; }
.lg:hover:not([disabled]) { border-color: var(--muted); }
.lg[disabled] { opacity: .38; cursor: not-allowed; border-style: dashed; }
.tv-wrap { position: relative; padding: 2px 0 0; }
.tv-view { display: none; }
.tv-view.act { display: block; }
.tv-svg { width: 100%; height: auto; display: block; }
.tv-grid line { stroke: var(--grid); stroke-width: 1; }
.tv-tick { font-family: var(--font-mono); font-size: 10.5px; fill: var(--muted);
  font-variant-numeric: tabular-nums; }
.tv-price { fill: none; stroke: var(--ink-2); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round; }
.tv-end { fill: var(--ink); }
.tv-endlab { font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  fill: var(--ink); }
.blind-wash { fill: var(--muted); opacity: .13; }
.blind-tag { font-family: var(--font-mono); font-size: 10px; letter-spacing: 1px;
  fill: var(--muted); }
.lg-zone { font-family: var(--font-mono); font-size: 10px; color: var(--muted);
  letter-spacing: .08em; padding: 2px 0 2px 10px; border-left: 2px solid var(--border);
  white-space: nowrap; align-self: center; }
.band-wash { fill: var(--crit-wash); }
.band-edge { stroke: var(--crit); stroke-width: 1; stroke-dasharray: 3 3; opacity: .6; }
.band-tag { font-family: var(--font-mono); font-size: 10px; letter-spacing: 1px;
  fill: var(--crit-text); }
.mk-halo { stroke: var(--surface); stroke-width: 4.5; fill: none; }
.mk-tag { font-family: var(--font-mono); font-size: 10.5px; font-weight: 700; }
.tv-hit { fill: transparent; cursor: pointer; }
.tv-ov { fill: transparent; }
.xh line { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
.xh circle { fill: none; stroke: var(--ink); stroke-width: 1.6; }
#tv.hide-risk .ly-risk, #tv.hide-pitg .ly-pitg, #tv.hide-pitt .ly-pitt,
#tv.hide-musk .ly-musk, #tv.hide-trade .ly-trade,
#tv.hide-blind .ly-blind { display: none; }
#tv-tip { position: absolute; z-index: 5; pointer-events: none;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px;
  padding: 6px 10px; font-family: var(--font-mono); font-size: 11.5px;
  box-shadow: 0 4px 14px rgba(0,0,0,.25); max-width: 300px; }
#tv-tip div { color: var(--ink-2); white-space: nowrap;
  font-variant-numeric: tabular-nums; }
#tv-tip .tt-head { font-weight: 600; color: var(--ink); }
.axis-note { display: flex; gap: 18px; padding: 8px 2px 6px; font-size: 11px;
  color: var(--muted); font-family: var(--font-mono); flex-wrap: wrap; }
.tv-table { margin: 4px 0 8px; font-size: 12.5px; }
.tv-table summary { cursor: pointer; color: var(--muted); font-size: 12px;
  padding: 4px 0; }
.tv-table table { min-width: 640px; }

/* ===== 03 渠道健康（层级分组卡片） ===== */
.t-ic { width: 18px; height: 18px; flex: none; fill: none; stroke: currentColor;
  stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round;
  color: var(--ink-2); }
.tier-group { border-bottom: 1px solid var(--grid); padding: 12px 16px 16px; }
.tier-group:last-child { border-bottom: none; }
.tier-head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.tier-head h3 { margin: 0; }
.tier-head .h-sub { font-family: var(--font-mono); font-size: 11px; }
.src-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(236px, 1fr));
  gap: 12px; }
.src-card { border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface-2); padding: 10px 12px; display: flex;
  flex-direction: column; gap: 4px; min-width: 0; }
.src-card.crit { background: var(--crit-wash); border-color: var(--crit); }
.src-card.offline { color: var(--muted); opacity: .75; }
.sc-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.sc-top strong { font-family: var(--font-mono); font-size: 12.5px; letter-spacing: .02em; }
.sc-name { font-size: 11.5px; color: var(--muted); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.sc-met { display: flex; gap: 12px; align-items: center; font-size: 12px;
  color: var(--ink-2); flex-wrap: wrap; }
.sc-sub { font-family: var(--font-mono); font-size: 11px; color: var(--muted); }
.sc-sub .rel { margin-left: 6px; }
.wt-cell { display: inline-flex; align-items: center; gap: 8px; }
.wt-bar { width: 52px; height: 4px; border-radius: 2px; background: var(--grid);
  overflow: hidden; flex: none; }
.wt-bar span { display: block; height: 100%; background: var(--ink-2); }
.wtf-0, .wtf-1, .wtf-2, .wtf-3 { background: var(--ink-2); }

/* ===== 04 最新情报流 ===== */
.feed { list-style: none; margin: 0; padding: 6px 0; }
.ev { display: flex; align-items: baseline; gap: 12px; padding: 8px 16px;
  border-bottom: 1px solid var(--grid); }
.ev:last-child { border-bottom: none; }
.ev-time { flex: none; width: 86px; color: var(--muted); font-size: 11.5px; }
.ev .tier { flex: none; }
.ev-src { flex: none; width: 96px; font-size: 12px; color: var(--ink-2);
  overflow: hidden; text-overflow: ellipsis; }
.ev-title { min-width: 0; flex: 1; font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.ev-title a { color: inherit; }
.ev-title a:hover { color: var(--link); }
.ev-type { flex: none; color: var(--muted); }

/* ===== 05 计数与时延 ===== */
.duo { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.duo.one { grid-template-columns: 1fr; }
@media (max-width: 860px){ .duo { grid-template-columns: 1fr; } }
.duo-card { padding: 16px 18px; }
.duo-card h3 { display: flex; align-items: center; gap: 8px; margin: 0 0 4px; }
.duo-card h3 svg.ic { width: 15px; height: 15px; color: var(--muted); }
.duo-card .cap { font-size: 11px; color: var(--muted); margin-bottom: 12px;
  font-family: var(--font-mono); }
.brow { display: grid; grid-template-columns: 64px 1fr 96px; gap: 10px;
  align-items: center; padding: 7px 0; }
.brow .bars { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.brow .b { height: 8px; border-radius: 0 2px 2px 0; min-width: 2px; display: block; }
.brow .b.dim { opacity: .42; }
.brow .n { font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-2);
  font-variant-numeric: tabular-nums; text-align: right; line-height: 1.5; }
.lat-scale { position: relative; height: 14px; margin: 2px 0 10px; }
.lat-scale .tk { position: absolute; top: 0; font-family: var(--font-mono);
  font-size: 10px; color: var(--muted); transform: translateX(-50%); }
.lrow { display: grid; grid-template-columns: 140px 1fr 122px; gap: 10px;
  align-items: center; padding: 8px 0; }
@media (max-width: 1100px){ .lrow { grid-template-columns: 120px 1fr 110px; } }
.lrow .lbl { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
.lrow .sid { font-family: var(--font-mono); font-size: 11px; color: var(--ink-2);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lrow .rail { position: relative; height: 14px; min-width: 0; }
.lrow .rail .base { position: absolute; left: 0; right: 0; top: 6.5px; height: 1px;
  background: var(--grid); }
.lrow .range { position: absolute; top: 5px; height: 4px; border-radius: 2px; }
.lrow .p50 { position: absolute; top: 2px; width: 10px; height: 10px;
  border-radius: 50%; transform: translateX(-50%); border: 2px solid var(--surface); }
.lrow .p50.p90 { width: 7px; height: 7px; top: 3.5px; border-radius: 1px; }
.lrow .n { font-family: var(--font-mono); font-size: 11px; color: var(--ink-2);
  font-variant-numeric: tabular-nums; text-align: right; }

/* ===== footer ===== */
.footnote { font-size: 12px; color: var(--muted); margin: 8px 16px 12px; }
section > .footnote { margin-left: 2px; margin-right: 2px; }
.empty { color: var(--muted); padding: 30px 16px; }
footer { max-width: 1200px; margin: 56px auto 40px; padding: 16px 24px 0;
  border-top: 1px solid var(--border); display: flex; gap: 10px; flex-wrap: wrap;
  align-items: center; font-size: 12px; color: var(--muted); }
.chip { font-family: var(--font-mono); font-size: 11px; color: var(--muted);
  border: 1px solid var(--border); border-radius: 3px; padding: 3px 9px;
  font-variant-numeric: tabular-nums; letter-spacing: .03em; }
footer .end { margin-left: auto; font-family: var(--font-mono); font-size: 11px;
  color: var(--muted); letter-spacing: .06em; }
@media (max-width: 640px) {
  .ev-title { white-space: normal; }
  .ev { flex-wrap: wrap; }
}
""".replace("__LIGHT__", _LIGHT_TOKENS)

_JS = """
/* 主题：默认跟随系统（prefers-color-scheme），URL ?theme= 或页内按钮显式覆盖 */
(function () {
  var root = document.documentElement;
  var q = null;
  try { q = new URLSearchParams(location.search).get("theme"); } catch (e) {}
  if (q === "light" || q === "dark") {
    root.dataset.theme = q;
  } else if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    root.dataset.theme = mq.matches ? "light" : "dark";
    if (mq.addEventListener)
      mq.addEventListener("change", function (e) {
        if (!root.dataset.userTheme)
          root.dataset.theme = e.matches ? "light" : "dark";
      });
  }
  var btn = document.getElementById("themebtn");
  if (!btn) return;
  function label() {
    btn.textContent = root.dataset.theme === "dark" ? "切换亮色" : "切换暗色";
  }
  btn.addEventListener("click", function () {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.userTheme = "1";
    label();
  });
  label();
})();
(function () {
  var STALE_MS = %(stale_ms)d;
  function ago(iso) {
    var s = (Date.now() - Date.parse(iso)) / 1000;
    if (isNaN(s)) return "";
    var a = Math.abs(s), t;
    if (a < 90) t = Math.round(a) + " 秒";
    else if (a < 5400) t = Math.round(a / 60) + " 分钟";
    else if (a < 172800) t = (a / 3600).toFixed(1) + " 小时";
    else t = (a / 86400).toFixed(1) + " 天";
    return s >= 0 ? t + "前" : "未来 " + t;
  }
  function refresh() {
    document.querySelectorAll("[data-iso]").forEach(function (el) {
      var iso = el.getAttribute("data-iso");
      if (!iso) return;
      var rel = el.querySelector(".rel") ||
                (el.nextElementSibling && el.nextElementSibling.classList &&
                 el.nextElementSibling.classList.contains("rel")
                 ? el.nextElementSibling : null);
      if (rel) rel.textContent = ago(iso);
    });
    var pt = document.getElementById("poll-ts");
    var pill = document.getElementById("poll-pill");
    if (pt && pt.getAttribute("data-iso")) {
      var stale = Date.now() - Date.parse(pt.getAttribute("data-iso")) > STALE_MS;
      var stat = pt.closest(".stamp");
      if (stat) stat.classList.toggle("crit-text", stale);
      if (pill && stale) {
        pill.className = "pill crit";
        pill.innerHTML = '<span class="dot"></span>轮询中断';
      }
    }
    var gen = document.getElementById("gen-ts");
    var banner = document.getElementById("stale-banner");
    if (gen && banner)
      banner.hidden = Date.now() - Date.parse(gen.getAttribute("data-iso")) <= STALE_MS;
  }
  refresh();
  setInterval(refresh, 30000);
})();
"""

# 走势图交互（时间刷 / 图层开关 / 标记 tooltip / 十字线）。
# 不经 %-格式化，直接拼接；当前单实例（#tv），多标的时改为按容器迭代即可。
_TV_JS = """
(function () {
  var tv = document.getElementById("tv");
  if (!tv) return;
  var dataEl = document.getElementById("tv-data");
  var DATA = null;
  try { DATA = JSON.parse(dataEl.textContent); } catch (e) {}

  tv.querySelectorAll(".tv-rb").forEach(function (b) {
    b.addEventListener("click", function () {
      tv.querySelectorAll(".tv-rb").forEach(function (x) {
        x.classList.toggle("act", x === b);
      });
      var v = b.getAttribute("data-range");
      tv.querySelectorAll(".tv-view").forEach(function (w) {
        w.classList.toggle("act", w.id === "tv-view-" + v);
      });
    });
  });

  tv.querySelectorAll(".tv-lb:not([disabled])").forEach(function (b) {
    b.addEventListener("click", function () {
      b.classList.toggle("off");
      var off = b.classList.contains("off");
      b.setAttribute("aria-pressed", off ? "false" : "true");
      tv.classList.toggle("hide-" + b.getAttribute("data-layer"), off);
    });
  });

  var tip = document.getElementById("tv-tip");
  var wrap = document.getElementById("tv-wrap");
  function showTip(lines, cx, cy) {
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    lines.forEach(function (t, i) {
      var d = document.createElement("div");
      if (!i) d.className = "tt-head";
      d.textContent = t;
      tip.appendChild(d);
    });
    tip.hidden = false;
    var wr = wrap.getBoundingClientRect();
    var x = cx - wr.left + 14, y = cy - wr.top + 14;
    if (x + tip.offsetWidth > wr.width - 6) x = cx - wr.left - tip.offsetWidth - 14;
    if (y + tip.offsetHeight > wr.height - 4) y = cy - wr.top - tip.offsetHeight - 12;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }

  tv.querySelectorAll(".tv-hit").forEach(function (h) {
    function on(e) { showTip(h.getAttribute("data-tip").split("|"), e.clientX, e.clientY); }
    h.addEventListener("pointerenter", on);
    h.addEventListener("pointermove", on);
    h.addEventListener("pointerleave", hideTip);
  });

  if (!DATA) return;
  tv.querySelectorAll(".tv-ov").forEach(function (ov) {
    var svg = ov.closest("svg");
    var m = DATA.views[ov.getAttribute("data-view")];
    var xh = svg.querySelector(".xh");
    var xl = xh.querySelector("line"), xc = xh.querySelector("circle");
    ov.addEventListener("pointermove", function (e) {
      var r = svg.getBoundingClientRect();
      var vx = (e.clientX - r.left) * svg.viewBox.baseVal.width / r.width;
      var day = m.d0 + (vx - m.ml) / m.pw * m.ds;
      var lo = m.i0, hi = DATA.D.length - 1, mid;
      while (lo < hi) {
        mid = (lo + hi) >> 1;
        if (DATA.D[mid] < day) lo = mid + 1; else hi = mid;
      }
      if (lo > m.i0 && Math.abs(DATA.D[lo - 1] - day) < Math.abs(DATA.D[lo] - day)) lo--;
      var px = m.ml + (DATA.D[lo] - m.d0) / m.ds * m.pw;
      var py = m.mt + (1 - (Math.log(DATA.C[lo]) / Math.LN10 - m.ly0) / m.lys) * m.ph;
      xh.removeAttribute("hidden");
      xl.setAttribute("x1", px); xl.setAttribute("x2", px);
      xl.setAttribute("y1", m.mt); xl.setAttribute("y2", m.mt + m.ph);
      xc.setAttribute("cx", px); xc.setAttribute("cy", py);
      var iso = new Date((DATA.D[lo] - DATA.epoch) * 86400000)
        .toISOString().slice(0, 10);
      showTip([iso + "  收盘 " + DATA.C[lo].toFixed(2)], e.clientX, e.clientY);
    });
    ov.addEventListener("pointerleave", function () {
      xh.setAttribute("hidden", "");
      hideTip();
    });
  });
})();
"""


def render(db_path: Path = DB_PATH) -> str:
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sources = load_sources(conn)
        det = load_detector(conn)
        has_trades_table = has_table(conn, "detector_trades")

        # -- 价格上下文：CSV 日线 + yfinance 增量 + 现价 + S2（失败降级，P0-2）
        csv_dates, csv_closes = load_daily_closes()
        px: dict | None = None
        if get_price_context is not None:
            try:
                px = get_price_context(csv_dates, csv_closes, now=now)
            except Exception:  # noqa: BLE001
                px = None
        dates, closes = (px["dates"], px["closes"]) if px else (csv_dates, csv_closes)

        # -- 走势图价格新鲜度徽章（>1 交易日转红，P0-2）
        fresh_badge = ""
        if dates:
            today_et = now.astimezone(ET).date()
            stale_bd = bdays_between(dates[-1], today_et)
            if px is None or (px.get("error") and not px.get("from_cache")):
                fresh_badge = (
                    '<span class="pill sm crit"><span class="dot"></span>'
                    f"STALE · 增量更新失败 · 价格截至 {esc(str(dates[-1]))}</span>"
                )
            elif stale_bd > 1:
                fresh_badge = (
                    '<span class="pill sm crit"><span class="dot"></span>'
                    f"STALE · 价格截至 {esc(str(dates[-1]))} · 已滞后 {stale_bd} 交易日</span>"
                )
            elif stale_bd == 1:
                fresh_badge = (
                    f'<span class="pill sm"><span class="dot"></span>'
                    f"价格截至 {esc(str(dates[-1]))}（最近收盘）</span>"
                )
            else:
                fresh_badge = (
                    '<span class="pill sm good"><span class="dot"></span>'
                    "价格已更新至今日</span>"
                )

        shadow = load_shadow(now)
        body = [_ICON_SPRITE, render_topbar(conn, now, px)]
        symbol_block = render_symbol_view(
            "TSLA", dates, closes,
            load_risk_segments(), load_blind_segments(), load_pits(),
            load_musk_buys(),
            det["trades"] if det else [], has_trades_table, fresh_badge,
        )
        sections = [
            render_consensus(det, px, shadow, now),
            render_detector(det, now),
            symbol_block,
            render_health(load_health(conn, sources), now),
            render_timeline(load_timeline(conn), now),
            render_counts_latency(
                load_tier_counts(conn, now), load_latency(conn), sources
            ),
        ]
        rendered = [s for s in sections if s]
        # 板块等宽序号（01 态势…）：按实际渲染顺序编号，缺板块自动顺延
        rendered = [
            s.replace("__NO__", f"{i:02d}", 1) for i, s in enumerate(rendered, start=1)
        ]
        if not rendered:
            rendered = ['<p class="empty">数据库暂无可展示的表——待哨兵首采后刷新。</p>']
        body.append("<main>" + "".join(rendered) + "</main>")
    finally:
        conn.close()
    body.append(
        "<footer>"
        f'<span class="chip">CALIB {CALIB_BDAYS}bd</span>'
        f'<span class="chip">SHORT_JUMP +{SHORT_JUMP_PCT:.1f}%</span>'
        f'<span class="chip">LOOKBACK {LOOKBACK_BDAYS}bd</span>'
        f'<span class="chip">PERSIST {PERSIST_BDAYS}bd</span>'
        f'<span class="chip">COST {COST_LINE * 1e4:+.2f}bp</span>'
        f"<span>数据源 {esc(str(db_path))} · 只读渲染 · "
        "重生成 <code>python -m intel.dashboard</code></span>"
        '<span class="end">静态快照 · 非实时 · 假想推演不碰真钱</span>'
        "</footer>"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta http-equiv="refresh" content="300">\n'
        "<title>因果探测器 · 哨兵</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n<script>"
        + _JS % {"stale_ms": STALE_S * 1000}
        + _TV_JS
        + "</script>\n</body>\n</html>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="生成哨兵静态 HTML 仪表盘")
    ap.add_argument("--db", type=Path, default=DB_PATH, help="sentinel.sqlite 路径")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="输出 HTML 路径")
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"数据库不存在：{args.db}（先跑哨兵采集）")
    html = render(args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"written: {args.out}  ({len(html):,} bytes)")
    print(f"open: file://{args.out.resolve()}")


if __name__ == "__main__":
    main()
