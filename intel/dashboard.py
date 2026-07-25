"""因果探测器 · 哨兵 — 静态 HTML 仪表盘生成器.

读 data/intel/sentinel.sqlite（只读，mode=ro），渲染完全自包含的
data/intel/dashboard.html：内联全部 CSS/JS，无外部依赖，浏览器直接打开。
视觉语言合并自设计稿 data/intel/dashboard_design.html（v1.0，存档保留）：
暗色默认 + 亮色（尊重 prefers-color-scheme，页内可切换）、宋体衬线板块题
+ 等宽序号、标定环表盘 + 三灯信号组、军事地图旗标、T0-T3 色深+形状双编码。

视图（顶部标的标签栏之下切换，hash 记忆）：
  实时值班 = 下列全部板块（默认）；
  模拟探测（历史推演）= 双模式（模式切换 hash 记忆：#replay 探测器 / #fullsys 全系统）：
   · 探测器推演（防守层）= N3-H 考场（2023-07→2026-07）逐日回放播放器——
     价格曲线光标推进 + 两腿指标面板同步 + 触发日自动暂停弹日记卡
     （trigger_diary.md：看到什么→决定→之后 20 日实际→判对/错）+
     「使用指标」固定卡（两腿定义/冻结参数/证据等级/walk-forward 口径）+
     失明期灰显；播放数据预计算内嵌 JSON（精简字段控体积）。
   · 全系统推演（进攻层）= E8-A+S2 留出段（2025-10→2026-07）逐笔回放——
     outputs/e8a_replay（research/e8a_trades_export.py 由 models/e8a/holdout_ref.csv
     重放、与 frontier_shift/e11 存档核对一致）54 笔成交 + 8 笔 S2 拦截对照；
     同款播放器骨架（交易密，速度预设 20x），总结卡年化收益率置顶。

板块（等宽序号按实际渲染顺序编排）：
  顶栏：TSLA 现价与最近涨跌 / 生成时刻 / 最后事件入库时刻 / 最后轮询时刻
     （>30 分钟标红）+ 主题切换；<meta refresh> 每 5 分钟自动重载
  01 今日合议（每日决策卡：现价 · S2 开关读数（距 252 日高回撤，E11 冻结口径）·
     探测器状态与标定倒计时 · 策略线 shadow 健康 · 规则合成的综合一句话）
  02 晨间简报（自上次开盘以来：新 T0/T1 事件 · 探测器状态变化 · S2 读数变化 ·
     shadow 各策略昨晚会话（shadow_status.json）· 事件日历条（财报/FOMC/
     FINRA 空头发布/标定期满，未来 7 天高亮、30 天窗口）；无变化如实说无）
  03 态势总览（detector_state：标定环表盘（标定倒计时）+ 态势陈述 +
     两腿读数卡（含数据龄徽章）+ 证据等级行 + 三灯信号组 + 假想单判分 +
     等待板（已有信息 / 在等信息（下期 FINRA 发布推算、标定期满日）/
     条件行动手册 IF→THEN——由当前状态动态生成，注明 N3-H/N6/E11 依据）；
     表缺失整面板隐藏）
  04 战场走势（标的视图：TSLA 日线收盘（yfinance 增量补到最新，失败降级
     + STALE 徽章）对数坐标折线 + 1/3/8 年时间刷 + 可开关图层——图例分
     「研究回放（事后）」区（避险影线带 / 真坑 / 假坑 / Musk 数据失明期灰底）
     与「前向/事实（当时可知）」区（Musk 菱旗 / 假想单十字准星）；
     坑判据写进 tooltip 与明细表。页面预留多标的标签栏。）
  05 渠道健康（按 T0-T3 分组的渠道卡片 + 衍生信号单列；权重标注人工先验；
     质量旗标（options oi_quality）与下游依赖（x_nitter→腿 B）上卡）
  06 最新情报流（治理版：高信号置顶区（T0/T1/衍生信号）+ T2/T3 限额可展开 +
     同题跨源去重折叠「另 N 源」+ 层级筛选按钮 + 类型/来源人话化）
  07 计数与时延（层级双条计数（ET 日口径）+ 每渠道 p50→p90 稳态口径标尺）

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
import re
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from intel.store import DB_PATH

try:  # 价格上下文（yfinance 增量 + S2 读数）；导入失败降级为"取价失败"
    from intel.prices import get_price_context, s2_reading
except Exception:  # noqa: BLE001
    get_price_context = None  # type: ignore[assignment]
    s2_reading = None  # type: ignore[assignment]

ET = ZoneInfo("America/New_York")

try:  # 与探测器冻结参数保持同源；导入失败退回写死值（容错，不炸仪表盘）
    from intel.detector import (
        CALIB_BDAYS, COST_LINE, LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT,
    )
except Exception:  # noqa: BLE001
    CALIB_BDAYS, COST_LINE = 20, -0.0006
    LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT = 20, 20, 10.0

try:  # 标定常量与拆股防护阈值同源引用；失败退回写死值
    from intel.detector import DENSE_QUANTILE, DENSE_REF_COUNT as DENSE_REF
except Exception:  # noqa: BLE001
    DENSE_REF, DENSE_QUANTILE = 65, 0.8789
try:
    from intel.splits import SPLIT_GUARD_PCT as SPLIT_GUARD_PCT_V
except Exception:  # noqa: BLE001
    SPLIT_GUARD_PCT_V = 50.0
DENSE_QUANT_TXT = f"{DENSE_QUANTILE:.4f}"

OUT_PATH = DB_PATH.parent / "dashboard.html"

STALE_S = 30 * 60  # 最后轮询距今超过此秒数 → 顶栏标红

# ---- 走势与历史判断（标的视图）数据源 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BARS_CSV = PROJECT_ROOT / "data" / "TSLA_1h_alpaca.csv"
STATES_CSV = PROJECT_ROOT / "outputs" / "n3h_deduction" / "daily_states.csv"
PITS_CSV = PROJECT_ROOT / "outputs" / "n4_golden_pit" / "pits_catalog.csv"
FORM4_CSV = PROJECT_ROOT / "data" / "intel" / "edgar_form4.csv"
DIARY_MD = PROJECT_ROOT / "outputs" / "n3h_deduction" / "trigger_diary.md"
APP_CSV = PROJECT_ROOT / "outputs" / "n3h_deduction" / "application_results.csv"
FINRA_CSV = PROJECT_ROOT / "data" / "intel" / "finra_short.csv"
E8A_REPLAY_DIR = PROJECT_ROOT / "outputs" / "e8a_replay"  # 全系统推演数据

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
    "RISK_OFF": ("crit", "假想减仓",
                 "空头 up-jump × Musk 密集命中 · 口径：假想全仓→现金（应用 A）"),
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


NITTER_ALERT_STREAK = 3  # x_nitter 连续失败达此数 → 探测器面板依赖告警（P1-4）


def load_health_extras(conn: sqlite3.Connection, health: list[dict],
                       now: datetime) -> tuple[dict[str, dict], str | None]:
    """渠道卡附加信息（质量旗标/下游依赖）+ 探测器面板的放风腿依赖告警。

    - options_snapshot：最新快照 payload.oi_quality=suspect_zero → 旗标上卡；
    - x_nitter：常驻「下游依赖 = 探测器腿 B」标注；连续失败 ≥ NITTER_ALERT_STREAK
      时升级 crit 并返回告警文案（render_detector 顶部显示）。
    """
    extras: dict[str, dict] = {}
    # -- options_snapshot 质量旗标（采集器已产出，透传即可）
    if has_table(conn, "events"):
        r = conn.execute(
            """SELECT payload_json FROM events WHERE type = 'options_snapshot'
               ORDER BY observed_time_utc DESC LIMIT 1"""
        ).fetchone()
        if r is not None:
            try:
                pl = json.loads(r["payload_json"] or "{}")
            except ValueError:
                pl = {}
            q = pl.get("oi_quality")
            if q == "suspect_zero":
                extras["options_snapshot"] = {"flag": (
                    "warn",
                    f"oi_quality=suspect_zero（{pl.get('snapshot_date', '—')} 快照 "
                    "Yahoo OI 整链近零）——OI 类读数当日不可用，volume 类不受影响",
                )}
            elif q == "ok":
                extras["options_snapshot"] = {"flag": (
                    "", f"oi_quality=ok（{pl.get('snapshot_date', '—')} 快照）")}
    # -- x_nitter 下游依赖 + 连败告警
    nit = next((h for h in health if h["source_id"] == "x_nitter"), None)
    nitter_alert: str | None = None
    if nit is not None:
        streak = nit.get("fail_streak") or 0
        dep_txt = "探测器腿 B（Musk 放风腿）唯一前向数据源——本渠道死亡即腿 B 失明"
        if streak >= NITTER_ALERT_STREAK:
            last_t = parse_ts((nit.get("last") or {}).get("poll_time_utc"))
            nitter_alert = (
                f"放风腿数据源危险：x_nitter 已连续失败 {streak} 次"
                + (f"（最后尝试 {fmt_ago(last_t, now)}）" if last_t else "")
                + "——腿 B（Musk 发帖密度）正在断供，基线与触发判定不可信；"
                "nitter 实例死亡时启用付费备选 x_api_paid（sources 已登记）。"
            )
            extras["x_nitter"] = {"dep": dep_txt + f"；当前连败 {streak} 次",
                                  "dep_lvl": "crit"}
        else:
            extras["x_nitter"] = {"dep": dep_txt,
                                  "dep_lvl": "warn" if streak else ""}
    return extras, nitter_alert


def load_timeline(conn: sqlite3.Connection, limit: int = 250) -> list[dict]:
    """情报流原始行（含 payload_json，供来源名/去重用）；抓宽再治理（P1-1）。"""
    if not has_table(conn, "events"):
        return []
    join_tier = has_table(conn, "sources")
    sql = (
        """SELECT e.event_id, e.observed_time_utc, e.event_time_utc, e.source_id,
                  e.type, e.title, e.url, e.payload_json{tier_col}
           FROM events e {join}
           ORDER BY e.observed_time_utc DESC, e.event_id DESC LIMIT ?"""
    ).format(
        tier_col=", s.tier" if join_tier else "",
        join="LEFT JOIN sources s ON s.source_id = e.source_id" if join_tier else "",
    )
    return [dict(r) for r in conn.execute(sql, (limit,))]


def load_hi_timeline(conn: sqlite3.Connection, limit: int = 20,
                     per_source: int = 5) -> list[dict]:
    """高信号置顶区专用查询：T0/T1/衍生信号按自身时间线取，不受 T3 流量挤出窗口；
    单一渠道限 per_source 条（窗口函数），防 polymarket 快照类批量刷屏。"""
    if not (has_table(conn, "events") and has_table(conn, "sources")):
        return []
    try:
        return [dict(r) for r in conn.execute(
            """SELECT * FROM (
                 SELECT e.event_id, e.observed_time_utc, e.event_time_utc,
                        e.source_id, e.type, e.title, e.url, e.payload_json, s.tier,
                        ROW_NUMBER() OVER (PARTITION BY e.source_id
                          ORDER BY e.event_time_utc DESC, e.event_id DESC) AS rn
                 FROM events e LEFT JOIN sources s ON s.source_id = e.source_id
                 WHERE s.tier IN ('T0', 'T1') OR e.source_id = 'detector')
               WHERE rn <= ?
               ORDER BY observed_time_utc DESC, event_time_utc DESC, event_id DESC
               LIMIT ?""",
            (per_source, limit),
        )]
    except sqlite3.OperationalError:  # 老 SQLite 无窗口函数 → 简单退化
        return [dict(r) for r in conn.execute(
            """SELECT e.event_id, e.observed_time_utc, e.event_time_utc, e.source_id,
                      e.type, e.title, e.url, e.payload_json, s.tier
               FROM events e LEFT JOIN sources s ON s.source_id = e.source_id
               WHERE s.tier IN ('T0', 'T1') OR e.source_id = 'detector'
               ORDER BY e.observed_time_utc DESC, e.event_id DESC LIMIT ?""",
            (limit * 10,),
        )]


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


def load_daily_opens() -> dict[date, float]:
    """ET 交易日日线开盘价 {日期: 开盘}（假想交易执行价用）；失败返回空 dict。"""
    try:
        import sys
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.common.data_io import load_bars
        df = load_bars(str(BARS_CSV))
        et = df.index.tz_convert("America/New_York")
        s = df["Open"].groupby(et.date).first()
        return {d: float(v) for d, v in zip(s.index, s.values)}
    except Exception:  # noqa: BLE001
        return {}


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


# ------------------------------------------- replay & waitboard · data pulls

def load_replay_days() -> list[dict] | None:
    """daily_states.csv 全行（N3-H 考场逐日）；缺失/坏行 → None（回放页降级）。"""
    try:
        out = []
        with STATES_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.append(
                    {
                        "date": date.fromisoformat(row["date"].strip()),
                        "off": (row.get("state") or "").strip().lower() == "risk_off",
                        "tweets": _ffloat(row.get("prev_day_tweets")),
                        "thr": _ffloat(row.get("dense_thr")),
                        "trig": (row.get("trigger") or "").strip() == "True",
                        "blind": (row.get("tweet_data_blind") or "").strip() == "True",
                        "reason": (row.get("reason") or "").strip(),
                    }
                )
        return out or None
    except Exception:  # noqa: BLE001
        return None


def load_e8a_replay() -> dict | None:
    """outputs/e8a_replay/{trades,daily}.csv + summary.json → 全系统推演数据。

    由 research/e8a_trades_export.py 从 models/e8a/holdout_ref.csv 逐笔重放导出，
    并已与 frontier_shift.csv / e11 switch_table.csv 存档核对；
    任一文件缺失/坏行 → None（全系统推演页整体降级为空态卡）。
    """
    try:
        trades = []
        with (E8A_REPLAY_DIR / "trades.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trades.append({
                    "seq": int(row["seq"]),
                    "et_day": date.fromisoformat(row["et_day"].strip()),
                    "entry_hm": row["entry_et"].strip()[-5:],
                    "exit_hm": row["exit_et"].strip()[-5:],
                    "prob": float(row["prob"]),
                    "entry_fill": float(row["entry_fill"]),
                    "exit_px": float(row["exit_px"]),
                    "exit_type": row["exit_type"].strip(),
                    "ret": float(row["ret"]),
                    "blocked": row["blocked"].strip() == "True",
                    "cum_eq": _ffloat(row.get("cum_eq_s2")),
                })
        daily = []
        with (E8A_REPLAY_DIR / "daily.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                daily.append({"date": date.fromisoformat(row["et_day"].strip()),
                              "close": float(row["close"]),
                              "off": row["s2_off"].strip() == "True"})
        summary = json.loads(
            (E8A_REPLAY_DIR / "summary.json").read_text(encoding="utf-8"))
        if not trades or not daily or "s2" not in summary:
            return None
        return {"trades": trades, "daily": daily, "summary": summary}
    except Exception:  # noqa: BLE001
        return None


def load_app_results() -> dict | None:
    """application_results.csv → 应用 A 收益对比与 p 值；缺失/坏行 → None（总结卡降级）。"""
    try:
        out: dict = {}
        with APP_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = ((row.get("application") or "").strip(),
                       (row.get("variant") or "").strip())
                if key == ("A_bh", "buy_and_hold"):
                    out["bh"] = _ffloat(row.get("total_ret"))
                elif key[0] == "A_overlay":
                    out["ov"] = _ffloat(row.get("total_ret"))
                    out["p_boot"] = _ffloat(row.get("p_boot"))
                    out["off_days"] = _ffloat(row.get("off_days"))
                    out["switches"] = _ffloat(row.get("switches"))
        if out.get("bh") is None or out.get("ov") is None:
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


def load_finra_releases() -> list[dict] | None:
    """finra_short.csv（拆股修正后主文件）：按发布时刻升序的空头利益发布。"""
    try:
        out = []
        with FINRA_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("type") or "").strip() != "short_interest":
                    continue
                try:
                    pl = json.loads(row.get("payload") or "{}")
                    pub = datetime.fromisoformat(row["event_time_utc"])
                except (ValueError, KeyError):
                    continue
                chg = _ffloat(pl.get("change_pct"))
                if chg is None:
                    continue
                out.append(
                    {
                        "pub_utc": pub,
                        "pub_date": pub.astimezone(ET).date(),
                        "settle": (pl.get("settlement_date") or "").strip(),
                        "chg": chg,
                        "si": pl.get("short_interest"),
                        "dtc": _ffloat(pl.get("days_to_cover")),
                    }
                )
        out.sort(key=lambda r: r["pub_utc"])
        return out or None
    except Exception:  # noqa: BLE001
        return None


def load_trigger_cards() -> dict[str, dict] | None:
    """trigger_diary.md → {触发日: 日记卡}（看到什么/决定/之后实际/判定）。

    解析按行进行、对格式漂移容错：Episode 头行开新组，触发行归组，
    组尾的 决定/之后实际/判定 行回填给组内全部触发日。
    """
    try:
        text = DIARY_MD.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    cards: dict[str, dict] = {}
    ep_title, ep_dates = "", []
    ep_no = 0

    def _strip_md(s: str) -> str:
        return s.replace("**", "").strip()

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            ep_no += 1
            ep_title, ep_dates = _strip_md(line[3:]), []
            continue
        m = line.strip()
        if m.startswith("- **触发 "):
            head, _, body = m[2:].partition("：")
            d = _strip_md(head).replace("触发", "").strip()
            cards[d] = {"ep": ep_no, "ep_title": ep_title,
                        "seen": _strip_md(body), "decide": "", "after": "",
                        "ok": None, "verdict": ""}
            ep_dates.append(d)
        elif m.startswith("- 决定："):
            for d in ep_dates:
                cards[d]["decide"] = _strip_md(m[len("- 决定："):])
        elif m.startswith("- 之后实际："):
            for d in ep_dates:
                cards[d]["after"] = _strip_md(m[len("- 之后实际："):])
        elif m.startswith("- 判定："):
            v = _strip_md(m[len("- 判定："):])
            ok = v.startswith("对")
            for d in ep_dates:
                cards[d]["ok"] = ok
                cards[d]["verdict"] = v
    return cards or None


def _roll_back_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def next_short_period(latest_settle: date) -> tuple[date, date]:
    """按 FINRA 结算日规则（每月 15 日与月末，遇周末前移）推算下一期。

    返回 (下一结算日, 预计发布日)；发布 ≈ 结算 + 9 交易日 16:00 ET
    （与 finra_short.csv publication_rule 同口径，busday 近似不剔假日）。
    """
    y, m = latest_settle.year, latest_settle.month
    cands: list[date] = []
    for k in range(3):
        yy, mm = y + (m - 1 + k) // 12, (m - 1 + k) % 12 + 1
        last_day = (date(yy + (mm == 12), mm % 12 + 1, 1) - timedelta(days=1))
        cands.append(_roll_back_weekday(date(yy, mm, 15)))
        cands.append(_roll_back_weekday(last_day))
    nxt = min(c for c in cands if c > latest_settle)
    return nxt, add_bdays(nxt, 9)


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
                (f"Musk 趋势比 {p['mtr']:.2f}（坑期发帖/前 60 日常态，>1 = 更活跃）"
                 if p["mtr"] is not None else "Musk 趋势比 —"),
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
        act_zh = ("减仓（口径：全仓→现金，应用 A）" if reduce_
                  else "恢复（全仓买回）")
        tip = _tip_attr(
            [
                f"假想单 {tag} · {t.get('action')} {act_zh} · {td}",
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
        "<th>后 60 日最高</th><th>空头 6 周</th>"
        '<th title="坑期发帖密度 / 前 60 日常态，&gt;1 = 更活跃">Musk 趋势比</th></tr></thead>'
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


def render_view_tabs() -> str:
    """视图切换（标的标签栏之下）：实时值班（默认，全部现有板块）｜模拟探测（历史推演）。

    切换纯前端显隐（JS），location.hash 记忆视图，meta refresh 后不丢。
    """
    return (
        '<div class="viewbar">'
        '<div class="viewtabs" role="tablist" aria-label="视图">'
        '<button class="vt act" type="button" data-view="live" role="tab" '
        'aria-selected="true">实时值班</button>'
        '<button class="vt" type="button" data-view="replay" role="tab" '
        'aria-selected="false">模拟探测（历史推演）</button>'
        '<a class="vt vt-link" href="playbook.html">棋谱预案 ↗</a>'
        "</div>"
        '<span class="sym-hint">模拟探测 = 历史推演双模式：防守层（N3-H 探测器'
        "考场回放）／进攻层（E8-A+S2 留出段逐笔回放）</span></div>"
    )


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


# ------------------------------------------------------- 等待板（值班台账）

def _wb_cell(k: str, v: str, ref: str, cls: str = "") -> str:
    return (
        f'<div class="cx-cell{cls}"><div class="cx-k">{k}</div>'
        f'<div class="cx-v num">{v}</div>'
        f'<div class="cx-ref">{ref}</div></div>'
    )


def _pb_row(cond: str, act: str, src: str) -> str:
    return (
        '<div class="pb-row"><span class="pb-chip if">IF</span>'
        f'<span class="pb-cond">{cond}</span>'
        '<span class="pb-chip then">THEN</span>'
        f'<span class="pb-act">{act}</span>'
        f'<span class="pb-src">{esc(src)}</span></div>'
    )


def render_waitboard(det: dict | None, px: dict | None,
                     finra: list[dict] | None, now: datetime) -> str:
    """等待板：已有信息 / 在等信息 / 条件行动手册（全部由当前状态动态生成）。

    数据源：detector_state 最新行（两腿读数、标定进度）+ finra_short.csv
    修正后主文件（最新期数值与发布日、下一期推算）+ 价格上下文 S2。
    任一源缺失对应格降级显示，不炸。
    """
    if det is None:
        return ""
    cur = det["cur"]
    state = cur.get("state")
    calibrating = state == "CALIBRATING"
    baseline_days = int(cur.get("baseline_days") or 0)
    s2 = (px or {}).get("s2")

    # ---- 已有信息 ----
    latest = finra[-1] if finra else None
    if latest:
        si_txt = (f"{latest['si'] / 1e6:.1f}M 股" if latest.get("si") else "—")
        mismatch = ""
        if cur.get("short_settlement") and cur["short_settlement"] != latest["settle"]:
            mismatch = (f" · <b class=\"warn-text\">与 detector_state 记录"
                        f"（{esc(cur['short_settlement'])}）不一致</b>")
        cell_short = _wb_cell(
            "空头最新期（FINRA 双周）",
            f"{latest['chg']:+.2f}%",
            f"{si_txt} · 结算 {esc(latest['settle'])} · 发布 {esc(str(latest['pub_date']))} "
            + age_badge(latest["pub_date"], now, 14, "发布龄") + mismatch,
        )
    elif cur.get("short_chg_pct") is not None:
        cell_short = _wb_cell(
            "空头最新期（FINRA 双周）",
            f"{cur['short_chg_pct']:+.2f}%",
            f"结算 {esc(cur.get('short_settlement') or '—')} · "
            "finra_short.csv 不可读，退回 detector_state 记录", " crit",
        )
    else:
        cell_short = _wb_cell("空头最新期（FINRA 双周）", "无数据",
                              "finra_short.csv 与 detector_state 均无记录", " crit")

    mc = cur.get("musk_count")
    try:
        musk_d = date.fromisoformat(str(cur.get("musk_count_day")))
    except (TypeError, ValueError):
        musk_d = None
    cell_musk = _wb_cell(
        "Musk 昨日计数（nitter 口径）",
        f"{_fmt_count(mc)} 帖" if mc is not None else "无数据",
        f"口径日 {esc(cur.get('musk_count_day') or '—')} · post+RT，无 reply "
        + age_badge(musk_d, now, 2),
    )

    if s2:
        cell_s2 = _wb_cell(
            "S2 今日读数（距 252 日高回撤）",
            f"{s2['drawdown_pct']:+.1f}%",
            (f"已触发，超线 {abs(s2['margin_pp']):.1f} pp" if s2["triggered"]
             else f"未触发，距 −20% 线余量 {s2['margin_pp']:.1f} pp")
            + f" · 252 日高 {s2['high']:,.2f}（{esc(str(s2['high_date']))}）",
            " crit" if s2["triggered"] else "",
        )
    else:
        cell_s2 = _wb_cell("S2 今日读数（距 252 日高回撤）", "无法计算",
                           "取价失败或序列不足——缺口不是安全", " crit")

    thr = cur.get("dense_thr")
    cell_calib = _wb_cell(
        "标定基线进度",
        f"{baseline_days} / {CALIB_BDAYS} 交易日",
        (f"标定期内不出信号 · 阈值未生效（历史参考 {DENSE_REF}帖/日，Sprinklr 口径）"
         if calibrating
         else f"标定完成 · 当日密集阈值 {_fmt_count(thr)} 帖/日（nitter 基线分位映射）"),
    )

    # ---- 在等信息 ----
    today_et = now.astimezone(ET).date()
    if latest or cur.get("short_settlement"):
        try:
            base_settle = date.fromisoformat(
                str(latest["settle"] if latest else cur["short_settlement"]))
            nxt_settle, nxt_pub = next_short_period(base_settle)
            n_bd = bdays_between(today_et, nxt_pub)
            cell_next = _wb_cell(
                "下期 FINRA 空头发布",
                nxt_pub.strftime("%m-%d"),
                f"结算 {nxt_settle} · 预计发布 {nxt_pub}（结算 + 9 交易日 16:00 ET 口径，"
                + (f"还差约 {n_bd} 个交易日）" if n_bd > 0 else "已到期，等采集）"),
            )
        except (TypeError, ValueError):
            cell_next = _wb_cell("下期 FINRA 空头发布", "无法推算", "最新结算日不可解析", " crit")
    else:
        cell_next = _wb_cell("下期 FINRA 空头发布", "无法推算", "无最新期结算日", " crit")

    eta = calib_eta(str(cur.get("state_date")), baseline_days)
    remain = max(0, CALIB_BDAYS - baseline_days)
    cell_eta = _wb_cell(
        "标定期满日",
        str(eta) if eta else "—",
        ("期满起出正式信号（CALIBRATING → RISK_ON/OFF）" if calibrating
         else "已期满，正常值班"),
    )
    cell_remain = _wb_cell(
        "nitter 基线缺口",
        f"还差 {remain} 交易日" if remain else "已集齐",
        f"基线 = 扩张窗日计数，阈值取其 {DENSE_QUANT_TXT} 分位（65 帖/日的历史分位映射）",
    )

    # ---- 条件行动手册（动态分支）----
    thr_txt = (f"密集阈值 {_fmt_count(thr)} 帖/日" if thr is not None
               else "密集阈值（标定期满后由基线分位映射生效）")
    latest_chg = latest["chg"] if latest else cur.get("short_chg_pct")
    calib_note = (
        f"；<b class=\"warn-text\">标定期内（至 {eta}）仅记录不触发</b>" if calibrating and eta
        else "；<b class=\"warn-text\">标定期内仅记录不触发</b>" if calibrating else ""
    )
    rows = [
        _pb_row(
            f"下期空头 change ≥ +{SHORT_JUMP_PCT:.0f}%（信息A）<b>且</b>"
            f"当日 Musk 发帖 &gt; {esc(thr_txt) if thr is None else thr_txt}（信息B）",
            f"RISK_OFF {PERSIST_BDAYS} 交易日（重叠触发顺延）+ 假想减仓单"
            f"（口径：全仓→现金，应用 A；记 detector_trades，价格快照判分）{calib_note}",
            "N3-H 冻结规则",
        ),
        _pb_row(
            "只有 A 无 B（空头跳升但 Musk 不密集）/ 只有 B 无 A（密集但无 up-jump 发布）",
            "不触发，detector_state 逐日记录在案——两腿必须同窗命中且发布时刻早于密集日结束",
            "N3-H 冻结规则",
        ),
        _pb_row(
            f"下期空头 change ≥ +{SPLIT_GUARD_PCT_V:.0f}%",
            "拆股防护：<b>不自动触发</b>，发 detector_split_guard 人工复核事件"
            "（确认真实跳变后人工处置；历史上 +345.8%/+202.2% 均为拆股伪影）",
            "N6 SPLIT_GUARD",
        ),
    ]
    if s2:
        dd = s2["drawdown_pct"]
        if s2["triggered"]:
            rows.append(_pb_row(
                f"S2 当前 {dd:+.1f}%（已触发），若回撤修复至 −20% 以内",
                "E8-A 买入侧恢复发信号资格（shadow 白跑恢复出单）",
                "E11 S2",
            ))
            rows.append(_pb_row(
                f"S2 当前 {dd:+.1f}%，若进一步恶化",
                "维持停用——E8-A 崩盘段系统性反选，S2 是唯一压测通过的截断",
                "E11 S2",
            ))
        else:
            rows.append(_pb_row(
                f"S2 当前 {dd:+.1f}%（未触发），若回撤跌破 −20% 线"
                f"（现余量 {s2['margin_pp']:.1f} pp）",
                "E8-A 买入侧停用发信号资格（S2 开关关闸，只记录不出买入）",
                "E11 S2",
            ))
            rows.append(_pb_row(
                f"S2 当前 {dd:+.1f}%，若维持在 −20% 以内",
                "E8-A 买入侧保持发信号资格（gate+几何冻结口径照常出单）",
                "E11 S2",
            ))
    else:
        rows.append(_pb_row(
            "S2 读数缺席（取价失败）",
            "今天没人替你算 S2——需人工核对回撤后再谈 E8-A 发信号资格",
            "E11 S2",
        ))

    return f"""
  <div class="card waitboard">
    <div class="wb-head"><h3><svg class="ic" aria-hidden="true"><use href="#i-clock"/></svg>等待板 · 值班台账</h3>
    <span class="h-sub">已有信息 / 在等信息 / 条件行动手册 —— 由当前 detector_state · finra_short · S2 实时合成</span></div>
    <div class="wb-sec">已有信息</div>
    <div class="wb-grid">{cell_short}{cell_musk}{cell_s2}{cell_calib}</div>
    <div class="wb-sec">在等信息</div>
    <div class="wb-grid three">{cell_next}{cell_eta}{cell_remain}</div>
    <div class="wb-sec">条件行动手册（IF → THEN，每条注明依据）</div>
    <div class="playbook">{"".join(rows)}</div>
    <p class="footnote">推算口径：下期发布日按「结算 + 9 交易日」busday 近似（不剔美股假日，可能偏移 ~1 日）；
    最新空头 change_pct {f"{latest_chg:+.2f}%" if latest_chg is not None else "—"} 未达 +{SHORT_JUMP_PCT:.0f}% 时腿 A 未命中，手册首条分支等的是<b>下一期</b>发布。
    行动手册全部为假想推演口径（不碰真钱）；分支依据：N3-H 冻结规则（intel/detector.py）·
    N6 SPLIT_GUARD（intel/splits.py，阈值 +{SPLIT_GUARD_PCT_V:.0f}%）· E11 S2（models/e8a/meta.json s2_switch）。</p>
  </div>"""


# ------------------------------------------- 模拟探测（历史推演）· rendering

_RP_VB_H = 360  # 回放图视框高


def _replay_segs(days: list[dict], key: str) -> list[tuple[date, date]]:
    segs: list[tuple[date, date]] = []
    run = prev = None
    for r in days:
        if r[key]:
            if run is None:
                run = r["date"]
            prev = r["date"]
        elif run is not None:
            segs.append((run, prev))
            run = prev = None
    if run is not None:
        segs.append((run, prev))
    return segs


def render_replay_view(days: list[dict] | None, price_dates: list[date],
                       price_closes: list[float],
                       finra: list[dict] | None,
                       cards: dict[str, dict] | None,
                       opens: dict[date, float] | None = None,
                       app: dict | None = None) -> str:
    """模拟探测页：N3-H 考场逐日回放播放器 + 推演日志 + 假想交易 + 总结卡。

    数据全部预计算内嵌 JSON；daily_states/价格任一缺失 → 降级空态卡；
    开盘价/application_results 缺失 → 交易表与总结卡对应项降级为缺口说明。
    """
    price_map = dict(zip(price_dates, price_closes))
    rows: list[dict] = []
    if days:
        last_c = None
        for r in days:
            c = price_map.get(r["date"], last_c)
            if c is None:
                continue  # 价格起点之前
            last_c = c
            rows.append({**r, "close": c})
    if not rows:
        return """
<section>
  <h2><span class="sec-no">R</span>模拟探测<span class="h-sub">N3-H 考场逐日回放（2023-07 → 2026-07）</span></h2>
  <div class="card"><p class="empty">daily_states.csv 或价格数据不可读——回放页降级为空。</p></div>
</section>"""

    d0, d1 = rows[0]["date"], rows[-1]["date"]
    n = len(rows)
    n_trig = sum(r["trig"] for r in rows)
    n_blind = sum(r["blind"] for r in rows)

    # ---- 空头发布对齐：每日「当日可见的最新一期」索引（walk-forward，只看发布日 <= 当日）
    rels = [r for r in (finra or []) if r["pub_date"] <= d1]
    si_idx: list[int] = []
    j = -1
    for r in rows:
        while j + 1 < len(rels) and rels[j + 1]["pub_date"] <= r["date"]:
            j += 1
        si_idx.append(j)

    # ---- 假想交易：避险开始=生效日开盘清仓卖出；解除=期满次日开盘买回 ----
    cards = cards or {}
    opens = opens or {}
    open_dates = sorted(opens)
    row_pos = {r["date"]: k for k, r in enumerate(rows)}
    trig_no: dict[date, int] = {}
    tn = 0
    for r in rows:
        if r["trig"]:
            tn += 1
            trig_no[r["date"]] = tn
    ep_bh: dict[int, float] = {}  # 日记口径：risk-off 期间 B&H %
    for c in cards.values():
        mm = re.search(r"risk-off 期间 B&H ([+-]?\d+(?:\.\d+)?)%", c.get("after") or "")
        if mm and c.get("ep"):
            ep_bh[c["ep"]] = float(mm.group(1))
    off_segs = _replay_segs(rows, "off")
    trades: list[dict] = []
    for ei, (a, b) in enumerate(off_segs, start=1):
        k = bisect_right(open_dates, b)
        nxt = open_dates[k] if k < len(open_dates) else None
        sp = opens.get(a)
        bp = opens.get(nxt) if nxt else None
        res = (sp - bp) / sp * 100 if sp is not None and bp is not None else None
        ra = rows[row_pos[a]]
        si = si_idx[row_pos[a]]
        rd = rels[si] if si >= 0 else None
        t_txt = "—" if ra["tweets"] is None else f"{ra['tweets']:.0f}"
        h_txt = "—" if ra["thr"] is None else f"{ra['thr']:.1f}"
        chg_txt = "—" if rd is None else f"{rd['chg']:+.1f}%"
        trades.append({
            "ep": ei, "sd": str(a), "bd": str(nxt) if nxt else None,
            "sp": None if sp is None else round(sp, 2),
            "bp": None if bp is None else round(bp, 2),
            "res": None if res is None else round(res, 2),
            "bh": ep_bh.get(ei),
            "sb": (f"第 {trig_no.get(a, '?')} 次触发：前日发帖 {t_txt} 条 > 阈值 {h_txt}"
                   f"（365 日 90 分位）；回看 {LOOKBACK_BDAYS}bd 内空头 up-jump {chg_txt}"
                   + (f"（结算 {rd['settle']}）" if rd else "")
                   + " → risk-off 生效日开盘清仓转现金"),
            "sbs": f"第 {trig_no.get(a, '?')} 次触发 · 帖 {t_txt}>{h_txt} · 空头 {chg_txt}",
            "bb": (f"E{ei} risk-off {a} → {b}（F{PERSIST_BDAYS} 重叠触发顺延后期满），"
                   + (f"{nxt} 开盘买回" if nxt else "买回日超出价格数据")),
            "bbs": f"F{PERSIST_BDAYS} 期满解除",
        })

    # ---- SVG（几何同战场走势：对数 y，日期 x）----
    d0o, d1o = d0.toordinal(), d1.toordinal()
    dspan = max(1, d1o - d0o)
    pw, ph = _VB_W - _ML - _MR, _RP_VB_H - _MT - _MB
    closes = [r["close"] for r in rows]
    llo, lhi = math.log10(min(closes)), math.log10(max(closes))
    pad = 0.06 * ((lhi - llo) or 1.0)
    llo, lhi = llo - pad, lhi + pad
    lspan = lhi - llo

    def x(o: int) -> float:
        return _ML + (o - d0o) / dspan * pw

    def y(c: float) -> float:
        return _MT + (1 - (math.log10(c) - llo) / lspan) * ph

    parts: list[str] = []
    # 失明期灰底（时间轴灰显，P0-3 同口径）
    for a, b in _replay_segs(rows, "blind"):
        bx, bw = x(a.toordinal()), x(b.toordinal() + 1) - x(a.toordinal())
        tag = (f'<text class="blind-tag" x="{bx + bw / 2:.1f}" y="{_MT + ph - 8}" '
               'text-anchor="middle">失明期 · 放风腿无数据（Musk 归档缺口）</text>'
               if bw > 260 else "")
        parts.append(
            f'<rect class="blind-wash" x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" '
            f'height="{ph}"><title>Musk 归档缺口 {a} → {b}：该段探测器无放风腿数据，'
            "无触发 ≠ 判定安全</title></rect>" + tag
        )
    # risk-off 底纹
    for i, (a, b) in enumerate(_replay_segs(rows, "off"), start=1):
        bx, bw = x(a.toordinal()), x(b.toordinal() + 1) - x(a.toordinal())
        parts.append(
            f'<rect class="band-wash" x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" height="{ph}">'
            f"<title>risk-off E{i}：{a} → {b}</title></rect>"
            f'<rect x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" height="{ph}" '
            'fill="url(#hatch45)" pointer-events="none"/>'
            f'<text class="band-tag" x="{bx + bw / 2:.1f}" y="{_MT + 13}" '
            f'text-anchor="middle">E{i}</text>'
        )
    # 网格 + 轴
    grid, labels = [], []
    for t in _log_ticks(10 ** llo, 10 ** lhi):
        ty = y(t)
        grid.append(f'<line x1="{_ML}" y1="{ty:.1f}" x2="{_VB_W - _MR}" y2="{ty:.1f}"/>')
        labels.append(f'<text x="{_ML - 8}" y="{ty:.1f}" class="tv-tick" '
                      f'text-anchor="end" dominant-baseline="central">{t:,.0f}</text>')
    for t, lab in _x_ticks(d0, d1):
        tx = x(t.toordinal())
        grid.append(f'<line x1="{tx:.1f}" y1="{_MT}" x2="{tx:.1f}" y2="{_MT + ph}"/>')
        labels.append(f'<text x="{tx:.1f}" y="{_RP_VB_H - 10}" class="tv-tick" '
                      f'text-anchor="middle">{esc(lab)}</text>')
    parts.append(f'<g class="tv-grid">{"".join(grid)}</g>{"".join(labels)}')
    # 触发日标记（顶部小旗刻度）
    trig_marks = "".join(
        f'<line x1="{x(r["date"].toordinal()):.1f}" x2="{x(r["date"].toordinal()):.1f}" '
        f'y1="{_MT}" y2="{_MT + 10}"/>'
        for r in rows if r["trig"]
    )
    parts.append(f'<g class="rp-trigmarks">{trig_marks}</g>')
    # 价格折线 + 播放光标（竖线 + 圆点，JS 驱动）
    pts = " ".join(f"{x(r['date'].toordinal()):.1f},{y(r['close']):.1f}" for r in rows)
    parts.append(f'<polyline class="tv-price" points="{pts}"/>')
    parts.append(
        f'<g class="rp-cursor"><line id="rp-cur-line" x1="{_ML}" x2="{_ML}" '
        f'y1="{_MT}" y2="{_MT + ph}"/>'
        f'<circle id="rp-cur-dot" cx="{_ML}" cy="{y(rows[0]["close"]):.1f}" r="4"/>'
        f'<text id="rp-cur-px" x="{_ML + 8}" y="{_MT + 16}" class="rp-curlab"></text></g>'
    )
    svg = (f'<svg class="tv-svg" viewBox="0 0 {_VB_W} {_RP_VB_H}" role="img" '
           f'aria-label="N3-H 考场回放：TSLA 日线收盘（对数）与探测器状态">{"".join(parts)}</svg>')

    # ---- 内嵌 JSON（精简字段控体积）----
    card_out = {}
    for r in rows:
        k = str(r["date"])
        if r["trig"] and k in cards:
            c = cards[k]
            card_out[k] = {"ep": c["ep"], "t": c["ep_title"], "s": c["seen"],
                           "d": c["decide"], "a": c["after"],
                           "ok": bool(c["ok"]), "v": c["verdict"]}
        elif r["trig"]:
            card_out[k] = {"ep": 0, "t": "", "s": r.get("reason") or "触发（日记缺失）",
                           "d": "", "a": "", "ok": None, "v": "日记卡缺失"}
    data_json = json.dumps(
        {
            "D": [r["date"].toordinal() for r in rows],
            "C": [round(r["close"], 2) for r in rows],
            "T": [None if r["tweets"] is None else round(r["tweets"]) for r in rows],
            "H": [None if r["thr"] is None else round(r["thr"], 1) for r in rows],
            "S": [int(r["off"]) for r in rows],
            "G": [int(r["trig"]) for r in rows],
            "B": [int(r["blind"]) for r in rows],
            "SI": si_idx,
            "R": [[str(r["pub_date"]), r["settle"], round(r["chg"], 2)] for r in rels],
            "cards": card_out,
            "trades": trades,
            "epoch": _EPOCH_ORD,
            "meta": {"d0": d0o, "ds": dspan, "ml": _ML, "pw": pw,
                     "mt": _MT, "ph": ph, "ly0": round(llo, 6),
                     "lys": round(lspan, 6)},
        },
        separators=(",", ":"), ensure_ascii=False,
    )

    # ---- 页头「使用指标」固定卡 ----
    spec_card = f"""
  <div class="card rp-spec">
    <div class="rp-spec-grid">
      <div><div class="bt">两腿指标定义</div>
        <p>腿 A · 空头利益跳变：FINRA 双周空头利益 change_pct ≥ +{SHORT_JUMP_PCT:.0f}%
        的发布（拆股修正后口径），按<b>发布日</b>生效——act = 发布次交易日；
        腿 B · Musk 密集发帖：日发帖数 &gt; 密集阈值（本推演口径：过去 365 日 90 分位，
        参考值 {DENSE_REF} 帖/日，Sprinklr 归档）。两腿同窗命中（回看 {LOOKBACK_BDAYS} 交易日）才触发。</p></div>
      <div><div class="bt">冻结参数</div>
        <p><span class="chip">up-jump +{SHORT_JUMP_PCT:.0f}%</span>
        <span class="chip">密集阈值 {DENSE_REF}</span>
        <span class="chip">F{PERSIST_BDAYS}</span>
        <span class="chip">LOOKBACK {LOOKBACK_BDAYS}bd</span>
        —— 触发起 {PERSIST_BDAYS} 交易日 risk-off，重叠触发顺延；参数 2018→2023-06
        发现段冻结，考场内不许改。</p></div>
      <div><div class="bt">证据等级</div>
        <p>考场 2 段全对（2023-12 段 B&amp;H −1.7%、2024-02→04 段 −11.0%），但块 bootstrap
        <b>p=0.14</b>——仅方向性证据；<b>窄谱</b>过滤器（只防空头知情型下跌，对 2024-12→2025-04
        的 −53.8% 失明）；<b>N6 拆股修正后幸存</b>（考场段无拆股，2/2 判对原样）。</p></div>
      <div><div class="bt">口径</div>
        <p><b>Walk-forward</b>：考场段（2023-07 → 2026-07）未参与规则发现，探测器逐日只见
        当日开盘前信息（空头看发布日、发帖看前一日）；回放即重演当时视野，无事后修饰。
        失明期（Musk 归档止于 2025-05-08 后）灰显，无触发 ≠ 判定安全。</p></div>
    </div>
  </div>"""

    # ---- 假想交易记录小表（2 段 = 4 笔；开盘价缺口降级）----
    if trades:
        t_rows = []
        for t in trades:
            res = t["res"]
            if res is None:
                res_td = '<td class="num" rowspan="2">—</td>'
            else:
                cls = "good-text" if res >= 0 else "crit-text"
                res_td = (f'<td class="num rp-res {cls}" rowspan="2">'
                          f'{"躲掉" if res >= 0 else "错过"} {res:+.1f}%</td>')
            sp_txt = "—" if t["sp"] is None else f"{t['sp']:,.2f}"
            bp_txt = "—" if t["bp"] is None else f"{t['bp']:,.2f}"
            t_rows.append(
                f'<tr class="tr-sell"><td class="num">{esc(t["sd"])}</td>'
                f'<td>卖出 → 现金</td><td class="num">{sp_txt}</td>'
                f'<td class="rp-basis">{esc(t["sbs"])}</td>{res_td}</tr>'
                f'<tr class="tr-buy"><td class="num">{esc(t["bd"] or "—")}</td>'
                f'<td>买回</td><td class="num">{bp_txt}</td>'
                f'<td class="rp-basis">{esc(t["bbs"])}</td></tr>'
            )
        trades_tbl = ('<table class="rp-trades"><thead><tr><th>日期</th><th>动作</th>'
                      '<th>开盘价</th><th>依据</th><th>该段结果</th></tr></thead>'
                      f'<tbody>{"".join(t_rows)}</tbody></table>')
    else:
        trades_tbl = '<p class="lg-empty">开盘价数据或 risk-off 区段缺失——无法推导假想交易。</p>'

    # ---- 推演总结卡（数字全部现算/读取；缺失降级为缺口说明）----
    n_ep = len(off_segs)
    eps_ok: dict[int, bool | None] = {}
    for c in cards.values():
        if c.get("ep"):
            eps_ok[c["ep"]] = c.get("ok")
    ok_n = sum(1 for v in eps_ok.values() if v is True)
    judged_txt = f"{ok_n} / {n_ep} 对" if eps_ok and n_ep else "—"
    wins = sum(1 for t in trades if t["res"] is not None and t["res"] > 0)
    tot_tr = sum(1 for t in trades if t["res"] is not None)
    wr_txt = f"{wins} / {tot_tr}（{wins / tot_tr * 100:.0f}%）" if tot_tr else "—"
    if app:
        ov_p, bh_p = app["ov"] * 100, app["bh"] * 100
        ret_v = f'{ov_p:+.1f}% <span class="sub">vs 裸持 {bh_p:+.1f}%</span>'
        ret_line = (f"覆盖策略（risk-off 转现金，开盘执行含成本）考场总收益 <b>{ov_p:+.1f}%</b> "
                    f"vs 裸持有 <b>{bh_p:+.1f}%</b>，少亏/多赚 <b>{ov_p - bh_p:+.1f}pp</b>"
                    + (f"（切换 {app['switches']:.0f} 次 · risk-off {app['off_days']:.0f} 交易日）。"
                       if app.get("switches") is not None and app.get("off_days") is not None
                       else "。"))
        p_txt = f"{app['p_boot']:.2f}" if app.get("p_boot") is not None else "—"
    else:
        ret_v = "—"
        ret_line = "application_results.csv 不可读——收益对比本次缺席（数据缺口，非零改善）。"
        p_txt = "—"
    ep_lines = []
    for t in trades:
        res, bh = t["res"], t["bh"]
        seg = f"E{t['ep']} {t['sd']} → {t['bd'] or '—'}"
        mid = "" if bh is None else f"期间 B&amp;H {bh:+.1f}%，"
        if res is None:
            tail = "开盘价缺失，无法结算"
        else:
            cls = "good-text" if res >= 0 else "crit-text"
            tail = (f"假想 {t['sp']:,.2f} 卖出 → {t['bp']:,.2f} 买回，"
                    f'<b class="{cls}">{"躲掉" if res >= 0 else "错过"} {res:+.1f}%</b>')
        ep_lines.append(f'<p class="rs-ep"><b class="num">{esc(seg)}</b>：{mid}{tail}</p>')
    sum_card = f"""
  <div class="rp-sum" id="rp-sum" hidden role="dialog" aria-label="推演总结">
    <div class="rs-h"><span>推演总结 · N3-H 考场 {esc(str(d0))} → {esc(str(d1))}</span>
      <button class="rp-btn" id="rp-sum-close" type="button">关闭</button></div>
    <p class="rs-model">模型：FINRA 空头 change ≥ +{SHORT_JUMP_PCT:.0f}% 发布（腿 A）×
      Musk 日发帖 &gt; 365 日 90 分位（腿 B），回看 {LOOKBACK_BDAYS}bd 同窗命中才触发 →
      risk-off {PERSIST_BDAYS} 交易日（重叠触发顺延）；参数 2018→2023-06 发现段冻结，考场内未改。</p>
    <div class="rs-grid">
      <div class="rs-cell"><div class="rs-k">触发 → 避险段</div>
        <div class="rs-v num">{n_trig} 次 → {n_ep} 段</div></div>
      <div class="rs-cell"><div class="rs-k">段判定</div>
        <div class="rs-v num">{judged_txt}</div></div>
      <div class="rs-cell"><div class="rs-k">假想交易胜率</div>
        <div class="rs-v num">{wr_txt}</div></div>
      <div class="rs-cell"><div class="rs-k">覆盖 vs 裸持</div>
        <div class="rs-v num">{ret_v}</div></div>
    </div>
    {"".join(ep_lines)}
    <p class="rs-ep">{ret_line}</p>
    <div class="rs-note">诚实注脚：仅 {n_ep} 段 risk-off 样本、块 bootstrap p = {p_txt}——以上属
      <b>方向性证据</b>，不是统计结论，不构成任何上钱依据；另有 {n_blind} 个交易日
      （Musk 归档尽头之后）放风腿失明，该段「无触发」是覆盖缺口，不是安全判定。</div>
  </div>"""

    return f"""
<section id="rp">
  <h2><span class="sec-no">R1</span>推演播放器<span class="h-sub">N3-H 考场 {esc(str(d0))} → {esc(str(d1))} · {n} 交易日 · 触发 {n_trig} 次 · 失明 {n_blind} 日 · 事件实时写入推演日志 · 播完出总结</span></h2>
  {spec_card}
  <div class="card chartcard">
    <div class="rp-controls">
      <button id="rp-play" class="rp-btn primary" type="button">开始推演</button>
      <button id="rp-step" class="rp-btn" type="button">单步</button>
      <button id="rp-sum-btn" class="rp-btn" type="button">总结</button>
      <div class="tv-ranges" role="group" aria-label="速度">
        <button class="tv-rb rp-speed act" data-s="1">1x</button>
        <button class="tv-rb rp-speed" data-s="5">5x</button>
        <button class="tv-rb rp-speed" data-s="20">20x</button>
      </div>
      <input type="range" id="rp-range" min="0" max="{n - 1}" value="0" step="1"
        aria-label="推演进度">
      <span class="rp-date num" id="rp-date">{esc(str(d0))}</span>
      <span class="rp-count num" id="rp-count">1 / {n}</span>
    </div>
    <div class="tv-wrap rp-wrap" id="rp-wrap">
      {svg}
    </div>
    <div class="rp-panel">
      <div class="leg"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-scales"/></svg>
        腿 A · 空头最新期<span class="pill sm" id="rp-a-pill"><span class="dot"></span><span id="rp-a-pilltxt">—</span></span></div>
        <div class="val num" id="rp-a-val">—</div>
        <div class="ref" id="rp-a-ref">当日可见的最新一期（发布日 ≤ 当日）</div></div>
      <div class="leg" id="rp-legb"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-mega"/></svg>
        腿 B · Musk 前日发帖<span class="pill sm" id="rp-b-pill"><span class="dot"></span><span id="rp-b-pilltxt">—</span></span></div>
        <div class="val num" id="rp-b-val">—</div>
        <div class="ref" id="rp-b-ref">阈值 = 过去 365 日 90 分位（参考 {DENSE_REF}）</div></div>
      <div class="leg rp-statecell"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-lamp"/></svg>
        当日状态</div>
        <div class="val"><span class="pill" id="rp-st-pill"><span class="dot"></span><span id="rp-st-txt">RISK_ON</span></span></div>
        <div class="ref" id="rp-st-ref">—</div></div>
    </div>
    <div class="rp-lower">
      <div class="rp-logcard">
        <div class="lg-h">推演日志<span class="lg-cnt num" id="rp-log-cnt">0 条</span></div>
        <div class="lg-empty" id="rp-log-empty">按「开始推演」逐日回放：触发 / 连环确认 /
          避险开始 / 避险解除 / 失明期与假想买卖会实时追加到这里；点击行可展开当次完整日记。</div>
        <div class="lg-scroll" id="rp-log" role="log" aria-live="polite"></div>
      </div>
      <div class="rp-tradecard">
        <div class="lg-h">假想交易记录<span class="lg-cnt">避险=生效日开盘卖出 · 解除=次日开盘买回</span></div>
        {trades_tbl}
      </div>
    </div>
    <p class="footnote">回放为 N3-H 历史推演的逐日重演（outputs/n3h_deduction/daily_states.csv +
    trigger_diary.md），播放不中断：光标经过事件日自动追加推演日志，点击行展开完整日记；
    假想交易由 risk-off 区段推导——避险生效日开盘清仓、F{PERSIST_BDAYS} 期满次日开盘买回
    （开盘价取自 data/TSLA_1h_alpaca.csv 日线）。空头读数来自拆股修正后 finra_short.csv，
    按发布日对齐（当日只见已发布的最新一期）。灰底 = Musk 归档缺口（2025-05-08 后 {n_blind}
    个交易日放风腿无数据，探测器无法触发——该段"无触发"是失明，不是安全）。播放到达终点自动
    弹出推演总结，也可随时点「总结」查看。假想推演，不碰真钱。</p>
  </div>
  {sum_card}
  <script type="application/json" id="rp-data">{data_json}</script>
</section>"""


# ----------------------------------- 全系统推演（进攻层 E8-A+S2）· rendering


def render_replay_modebar() -> str:
    """模拟探测页内的推演模式切换 + 防守/进攻两层一行对比条（hash 记忆）。"""
    return (
        '<div class="rmodebar">'
        '<div class="viewtabs" role="tablist" aria-label="推演模式">'
        '<button class="rm act" type="button" data-mode="det" role="tab" '
        'aria-selected="true">探测器推演（防守层）</button>'
        '<button class="rm" type="button" data-mode="full" role="tab" '
        'aria-selected="false">全系统推演（进攻层 E8-A+S2）</button>'
        "</div>"
        '<span class="rm-cmp"><b>探测器推演 = 防守层</b>：稀疏出手，只为躲开特定'
        "类型的大跌（3 年考场仅 2 段避险、4 次假想买卖）；"
        "<b>全系统推演 = 进攻层</b>：E8-A 高频小额抄底 + 双开关"
        "（留出 10 个月 54 笔）。——「系统 3 年只交易 2 次」是只看防守层的误读，"
        "两层并行、各管一件事。</span></div>"
    )


def render_fullsys_view(data: dict | None,
                        det_days: list[dict] | None = None) -> str:
    """全系统推演页：E8-A+S2 留出段逐笔回放播放器 + 交易日志 + 总结卡。

    数据全部来自 outputs/e8a_replay（重放导出 + 存档核对）；缺失 → 降级空态卡。
    det_days（N3-H daily_states）用于窗口内防守层状态行，缺失只减行不炸。
    """
    if not data:
        return """
<section>
  <h2><span class="sec-no">R2</span>全系统推演<span class="h-sub">E8-A+S2 留出段逐笔回放（2025-10 → 2026-07）</span></h2>
  <div class="card"><p class="empty">outputs/e8a_replay/ 不可读——先跑
  <code>.venv/bin/python research/e8a_trades_export.py</code> 重建逐笔交易，本页降级为空。</p></div>
</section>"""

    trades, daily, s = data["trades"], data["daily"], data["summary"]
    s2, alls, blk = s["s2"], s["all"], s["blocked"]
    dist = s.get("exit_dist_s2", {})
    d0, d1 = daily[0]["date"], daily[-1]["date"]
    n_days = len(daily)
    n_kept = int(s2["n"])
    n_blk = int(blk["n"])
    wins = sum(1 for t in trades if not t["blocked"] and t["ret"] > 0)
    ann10 = s["ann_10mo"] * 100
    ann_act = s["ann_actual"] * 100

    # ---- 每日推进数组：S2 流累计收益 / 已成交笔数 / 已拦截笔数 ----------------
    q_day, n_day, k_day = [], [], []
    eq, n_done, n_b, ti = 1.0, 0, 0, 0
    kept_no: dict[int, int] = {}  # seq -> 第几笔（S2 流内编号）
    for t in trades:
        if not t["blocked"]:
            kept_no[t["seq"]] = len(kept_no) + 1
    for r in daily:
        while ti < len(trades) and trades[ti]["et_day"] <= r["date"]:
            t = trades[ti]
            if t["blocked"]:
                n_b += 1
            else:
                eq = t["cum_eq"] if t["cum_eq"] is not None else eq * (1 + t["ret"])
                n_done += 1
            ti += 1
        q_day.append(round((eq - 1) * 100, 2))
        n_day.append(n_done)
        k_day.append(n_b)

    # ---- 防守层状态行（窗口内如有触发/避险/失明如实入日志）--------------------
    det_rows: list[list] = []
    dw = [r for r in (det_days or []) if d0 <= r["date"] <= d1]
    if dw and dw[0]["blind"]:
        det_rows.append([d0.toordinal(), "防守层 · 失明",
                         "探测器（N3-H）窗口内全程失明：Musk 归档止于 2025-05-08，"
                         "放风腿无数据、不可能触发——缺席是覆盖缺口，不是安全判定",
                         ""])
    for j, r in enumerate(dw):
        prev = dw[j - 1] if j else None
        if r["blind"] and prev is not None and not prev["blind"]:
            det_rows.append([r["date"].toordinal(), "防守层 · 失明进入",
                             "Musk 归档尽头——放风腿无数据，无触发 ≠ 安全", ""])
        if r["trig"]:
            det_rows.append([r["date"].toordinal(), "防守层 · 触发",
                             r.get("reason") or "两腿同窗命中", ""])
        if r["off"] and (prev is None or not prev["off"]):
            det_rows.append([r["date"].toordinal(), "防守层 · 避险开始",
                             f"risk_off 生效 · F{PERSIST_BDAYS}", ""])
        if not r["off"] and prev is not None and prev["off"]:
            det_rows.append([r["date"].toordinal(), "防守层 · 避险解除",
                             f"F{PERSIST_BDAYS} 期满 · 回到 risk_on", ""])

    # ---- SVG（几何同探测器推演：对数 y，日期 x）------------------------------
    d0o, d1o = d0.toordinal(), d1.toordinal()
    dspan = max(1, d1o - d0o)
    pw, ph = _VB_W - _ML - _MR, _RP_VB_H - _MT - _MB
    closes = [r["close"] for r in daily]
    llo, lhi = math.log10(min(closes)), math.log10(max(closes))
    pad = 0.06 * ((lhi - llo) or 1.0)
    llo, lhi = llo - pad, lhi + pad
    lspan = lhi - llo

    def x(o: float) -> float:
        return _ML + (o - d0o) / dspan * pw

    def y(c: float) -> float:
        return _MT + (1 - (math.log10(c) - llo) / lspan) * ph

    parts: list[str] = []
    for a, b in _replay_segs(daily, "off"):  # S2 关闸底纹（窄段不标字免重叠）
        bx, bw = x(a.toordinal()), x(b.toordinal() + 1) - x(a.toordinal())
        tag = (f'<text class="band-tag" x="{bx + bw / 2:.1f}" y="{_MT + 13}" '
               'text-anchor="middle">S2 关闸</text>' if bw > 52 else "")
        parts.append(
            f'<rect class="band-wash" x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" height="{ph}">'
            f"<title>S2 关闸 {a} → {b}：距 252 日高点回撤 &gt; 20%，停止新开仓</title></rect>"
            f'<rect x="{bx:.1f}" y="{_MT}" width="{bw:.1f}" height="{ph}" '
            'fill="url(#hatch45)" pointer-events="none"/>' + tag
        )
    grid, labels = [], []
    for t in _log_ticks(10 ** llo, 10 ** lhi):
        ty = y(t)
        grid.append(f'<line x1="{_ML}" y1="{ty:.1f}" x2="{_VB_W - _MR}" y2="{ty:.1f}"/>')
        labels.append(f'<text x="{_ML - 8}" y="{ty:.1f}" class="tv-tick" '
                      f'text-anchor="end" dominant-baseline="central">{t:,.0f}</text>')
    for t, lab in _x_ticks(d0, d1):
        tx = x(t.toordinal())
        grid.append(f'<line x1="{tx:.1f}" y1="{_MT}" x2="{tx:.1f}" y2="{_MT + ph}"/>')
        labels.append(f'<text x="{tx:.1f}" y="{_RP_VB_H - 10}" class="tv-tick" '
                      f'text-anchor="middle">{esc(lab)}</text>')
    parts.append(f'<g class="tv-grid">{"".join(grid)}</g>{"".join(labels)}')
    marks = []
    for t in trades:  # 逐笔小旗刻度：绿盈红亏灰拦截；日内时间微移避免重叠
        hh, mm = t["entry_hm"].split(":")
        frac = min(1.0, max(0.0, (int(hh) * 60 + int(mm) - 570) / 390))
        mx = x(t["et_day"].toordinal() + frac)
        cls = "b" if t["blocked"] else ("w" if t["ret"] > 0 else "l")
        marks.append(f'<line class="{cls}" x1="{mx:.1f}" x2="{mx:.1f}" '
                     f'y1="{_MT}" y2="{_MT + 10}"/>')
    parts.append(f'<g class="fs-marks">{"".join(marks)}</g>')
    pts = " ".join(f"{x(r['date'].toordinal()):.1f},{y(r['close']):.1f}" for r in daily)
    parts.append(f'<polyline class="tv-price" points="{pts}"/>')
    parts.append(
        f'<g class="rp-cursor"><line id="fs-cur-line" x1="{_ML}" x2="{_ML}" '
        f'y1="{_MT}" y2="{_MT + ph}"/>'
        f'<circle id="fs-cur-dot" cx="{_ML}" cy="{y(daily[0]["close"]):.1f}" r="4"/>'
        f'<text id="fs-cur-px" x="{_ML + 8}" y="{_MT + 16}" class="rp-curlab"></text></g>'
    )
    svg = (f'<svg class="tv-svg" viewBox="0 0 {_VB_W} {_RP_VB_H}" role="img" '
           f'aria-label="全系统推演：TSLA 日线收盘（对数）与 E8-A+S2 逐笔交易">{"".join(parts)}</svg>')

    # ---- 内嵌 JSON（精简字段控体积）----
    t_out = [{
        "o": t["et_day"].toordinal(), "sq": t["seq"],
        "k": kept_no.get(t["seq"], 0),
        "e": t["entry_hm"], "x": t["exit_hm"],
        "p": round(t["prob"], 3),
        "ep": round(t["entry_fill"], 2), "xp": round(t["exit_px"], 2),
        "ty": t["exit_type"], "r": round(t["ret"] * 100, 2),
        "b": int(t["blocked"]),
        "q": None if t["cum_eq"] is None else round((t["cum_eq"] - 1) * 100, 2),
    } for t in trades]
    data_json = json.dumps(
        {
            "D": [r["date"].toordinal() for r in daily],
            "C": [round(r["close"], 2) for r in daily],
            "S": [int(r["off"]) for r in daily],
            "Q": q_day, "N": n_day, "K": k_day,
            "T": t_out, "DET": det_rows,
            "epoch": _EPOCH_ORD,
            "meta": {"d0": d0o, "ds": dspan, "ml": _ML, "pw": pw,
                     "mt": _MT, "ph": ph, "ly0": round(llo, 6),
                     "lys": round(lspan, 6), "nk": n_kept, "nb": n_blk},
        },
        separators=(",", ":"), ensure_ascii=False,
    )

    # ---- 页头「使用策略」固定卡 ----
    spec_card = f"""
  <div class="card rp-spec">
    <div class="rp-spec-grid">
      <div><div class="bt">进攻层定义（E8-A）</div>
        <p>5m V 形反转事件（预登记网格并集去重）→ 池化 GBDT 打分（25 标的留一训练，
        无 TSLA，训练截止 2025-09-30）→ 门控 prob ≥ 0.4325（gate0.70 保留率匹配，
        Top≈10%）→ 下一根 5m bar 开盘做多；绑定 <b>S2 开关</b>：距 252 日滚动高点回撤
        &gt;20%（昨收判定）停止新开仓（E11 冻结）。</p></div>
      <div><div class="bt">冻结参数</div>
        <p><span class="chip">gate 0.4325</span>
        <span class="chip">tp +0.5%</span>
        <span class="chip">sl −2%</span>
        <span class="chip">timeout 48×5m</span>
        <span class="chip">日内平</span>
        <span class="chip">S2 −20%</span>
        —— 2026-07-24 冻结（models/e8a/meta.json），参数与开关规则不得再调。</p></div>
      <div><div class="bt">证据等级</div>
        <p>留出 62 笔 WR 80.6% / +16.0bp，但 tp/sl/timeout×门控组合出身<b>同一留出段的
        事后网格</b>（n=62），Bonferroni 校正后不显著（bootstrap p 0.04–0.28）；
        崩盘段（2021-10→2023-01）门反选 −31.4%，S2 只截后续损失（削减 78%）、救不了
        顶部 27 个交易日——仅方向性证据。</p></div>
      <div><div class="bt">口径</div>
        <p>交易流 = models/e8a/holdout_ref.csv 62 笔入场 × E9 <b>悲观结算</b>重放
        （费 1bp/边 + 滑点 2bp；TP 限价需严格穿越、SL 停损市价含跳空、同 bar 双触发算
        SL；日内强平），已与 frontier_shift / e11 存档核对一致
        （research/e8a_trades_export.py）。S2 拦截 {n_blk} 笔照常结算、灰色对照显示，
        不入权益。</p></div>
    </div>
  </div>"""

    # ---- S2 拦截对照小表（8 笔灰条）----
    blk_rows = "".join(
        f'<tr class="fs-blkrow"><td class="num">{esc(str(t["et_day"]))} {esc(t["entry_hm"])}</td>'
        f'<td class="num">{t["entry_fill"]:,.2f}</td>'
        f'<td class="num">{t["prob"]:.3f}</td>'
        f'<td class="num">{t["ret"] * 100:+.2f}%</td></tr>'
        for t in trades if t["blocked"]
    )
    blk_tbl = (
        '<table class="rp-trades fs-blktbl"><thead><tr><th>入场时刻（ET）</th><th>假想入场价</th>'
        "<th>GBDT 概率</th><th>若成交结果</th></tr></thead>"
        f"<tbody>{blk_rows}</tbody></table>"
        f'<p class="fs-blknote">被拦 {n_blk} 笔若成交合计 {blk["total"] * 100:+.2f}%'
        f"（净亏）——方向与开关假设一致，但属同一留出段事后观察，不作加分证据。</p>"
    ) if blk_rows else '<p class="lg-empty">窗口内无 S2 拦截记录。</p>'

    # ---- 总结卡（年化收益率置顶）----
    dist_txt = (f"止盈 {dist.get('tp', 0)} · 止损 {dist.get('sl', 0)} · "
                f"超时 {dist.get('timeout', 0)} · 日终 {dist.get('dayend', 0)}")
    sum_card = f"""
  <div class="rp-sum" id="fs-sum" hidden role="dialog" aria-label="全系统推演总结">
    <div class="rs-h"><span>全系统推演总结 · E8-A+S2 留出段 {esc(str(d0))} → {esc(str(d1))}</span>
      <button class="rp-btn" id="fs-sum-close" type="button">关闭</button></div>
    <div class="fs-ann"><span class="fs-ann-k">年化收益率</span>
      <span class="fs-ann-v num">≈ {ann10:+.1f}%</span>
      <span class="fs-ann-sub">口径：留出 10 个月总收益 {s2["total"] * 100:+.2f}% 复利年化
      (1{s2["total"] * 100:+.2f}%)^(12/10) − 1；按实际 {s["window"]["cal_days"]} 天算
      ≈ {ann_act:+.1f}%。假想推演口径，含悲观成本，不含税与资金占用。</span></div>
    <div class="rs-grid">
      <div class="rs-cell"><div class="rs-k">交易笔数</div>
        <div class="rs-v num">{n_kept} 笔 <span class="sub">≈{n_kept / 10:.1f} 笔/月（10 个月口径）</span></div></div>
      <div class="rs-cell"><div class="rs-k">胜率</div>
        <div class="rs-v num">{s2["wr"] * 100:.1f}% <span class="sub">{wins}/{n_kept}；S2 前 62 笔 {alls["wr"] * 100:.1f}%</span></div></div>
      <div class="rs-cell"><div class="rs-k">出场分布</div>
        <div class="rs-v num" style="font-size:13px">{dist_txt}</div></div>
      <div class="rs-cell"><div class="rs-k">最大回撤</div>
        <div class="rs-v num">{s2["mtm_mdd"] * 100:.2f}% <span class="sub">bar 级盯市；逐笔口径 {s2["mdd_closed"] * 100:.2f}%</span></div></div>
    </div>
    <p class="rs-ep"><b>S2 开关贡献</b>：拦截 {n_blk} 笔（若成交合计 {blk["total"] * 100:+.2f}%），
      总收益 {alls["total"] * 100:+.2f}% → <b>{s2["total"] * 100:+.2f}%</b>，盯市 MDD
      {alls["mtm_mdd"] * 100:.2f}% → {s2["mtm_mdd"] * 100:.2f}%——被删笔净为亏损、方向与
      假设一致，但这是同一留出段的又一次事后观察（n 62 → {n_kept}），不作为加分证据。</p>
    <p class="rs-ep"><b>单笔期望</b>：+{s2["avg_bp"]:.1f}bp/笔（tp +0.5% 高频小额），
      平均 {n_days / max(1, n_kept):.1f} 个交易日一笔——进攻层的「高频小额」与防守层的
      「稀疏避险」互补，合成整套系统的出手节奏。</p>
    <div class="rs-note">诚实注脚：① 留出期为 2025-10 → 2026-07 <b>单一时段</b>（TSLA
      本段整体下跌，抄底事件密集——换一段行情未必复现）；② 配方出身 n=62 的<b>事后网格</b>，
      Bonferroni 校正后不显著（bootstrap p 0.04–0.28），S2 过滤后的 +12.04% 亦含选择偏差；
      ③ 策略 2026-07-24 冻结、shadow 前向验证（≥8 周）进行中，效力由前向裁决——本页全部为
      假想推演，<b>不构成任何上钱依据</b>。</div>
  </div>"""

    return f"""
<section id="fs">
  <h2><span class="sec-no">R2</span>全系统推演<span class="h-sub">E8-A+S2 留出段 {esc(str(d0))} → {esc(str(d1))} · {n_days} 交易日 · 成交 {n_kept} 笔 · S2 拦截 {n_blk} 笔 · 逐笔写入交易日志 · 播完出总结</span></h2>
  {spec_card}
  <div class="card chartcard">
    <div class="rp-controls">
      <button id="fs-play" class="rp-btn primary" type="button">开始推演</button>
      <button id="fs-step" class="rp-btn" type="button">单步</button>
      <button id="fs-sum-btn" class="rp-btn" type="button">总结</button>
      <div class="tv-ranges" role="group" aria-label="速度">
        <button class="tv-rb fs-speed" data-s="1">1x</button>
        <button class="tv-rb fs-speed" data-s="5">5x</button>
        <button class="tv-rb fs-speed act" data-s="20">20x</button>
      </div>
      <input type="range" id="fs-range" min="0" max="{n_days - 1}" value="0" step="1"
        aria-label="推演进度">
      <span class="rp-date num" id="fs-date">{esc(str(d0))}</span>
      <span class="rp-count num" id="fs-count">1 / {n_days}</span>
    </div>
    <div class="tv-wrap rp-wrap">
      {svg}
    </div>
    <div class="rp-panel">
      <div class="leg"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-shield"/></svg>
        S2 开关（回撤 &gt; 20% 关闸）<span class="pill sm good" id="fs-s2-pill"><span class="dot"></span><span id="fs-s2-txt">开闸</span></span></div>
        <div class="val num" id="fs-s2-val">—</div>
        <div class="ref">距 252 日滚动高点回撤，昨收判定（E11 冻结口径）</div></div>
      <div class="leg"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-pulse"/></svg>
        累计收益（S2 流）</div>
        <div class="val num" id="fs-eq-val">+0.00%</div>
        <div class="ref">已平仓复利 · 悲观成本口径 · 拦截笔不入权益</div></div>
      <div class="leg"><div class="leg-h"><svg class="ic" aria-hidden="true"><use href="#i-clock"/></svg>
        成交进度</div>
        <div class="val num" id="fs-n-val">0 / {n_kept} 笔</div>
        <div class="ref" id="fs-n-ref">S2 拦截 0 / {n_blk} 笔</div></div>
    </div>
    <div class="rp-lower">
      <div class="rp-logcard">
        <div class="lg-h">交易日志<span class="lg-cnt num" id="fs-log-cnt">0 条</span></div>
        <div class="lg-empty" id="fs-log-empty">按「开始推演」逐日回放：每笔入场/出场
          （绿盈红亏）、S2 拦截灰条、开关关闸/开闸与防守层状态会实时追加到这里；
          点击行可展开门控概率与结算明细。交易密，速度已预设 20x。</div>
        <div class="lg-scroll" id="fs-log" role="log" aria-live="polite"></div>
      </div>
      <div class="rp-tradecard">
        <div class="lg-h">S2 拦截对照<span class="lg-cnt">灰条 = 关闸期被拒的入场</span></div>
        {blk_tbl}
      </div>
    </div>
    <p class="footnote">逐笔交易由 research/e8a_trades_export.py 从 models/e8a/holdout_ref.csv
    的 62 笔留出入场重放（E9 悲观结算），与 outputs/e8_pooled/frontier_shift.csv、
    outputs/e11_bear_switch/switch_table.csv 存档核对一致后落盘 outputs/e8a_replay/。
    S2 关闸底纹 = 距 252 日高点回撤 &gt;20%（昨收判定）。防守层（N3-H 探测器）在本窗口
    全程失明（Musk 归档止于 2025-05-08）且无触发，日志首行如实标注。播放到达终点自动弹出
    总结卡（年化收益率置顶），也可随时点「总结」。假想推演，不碰真钱。</p>
  </div>
  {sum_card}
  <script type="application/json" id="fs-data">{data_json}</script>
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
            d_txt = (f"假想减仓生效（全仓→现金口径）· F{PERSIST_BDAYS} 至 "
                     f"{cur.get('risk_off_until') or '—'}")
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


# ------------------------------------------------- 晨间简报 + 事件日历（P1-5）

CAL_HORIZON_D = 30  # 日历条时间窗（天）；≤7 天高亮


def last_market_open(now: datetime) -> datetime:
    """最近一次已发生的美股开盘时刻（工作日 09:30 ET，busday 近似不剔假日）。"""
    et_now = now.astimezone(ET)
    d = et_now.date()
    if d.weekday() >= 5 or et_now.time() < dt_time(9, 30):
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
    return datetime.combine(d, dt_time(9, 30), tzinfo=ET).astimezone(timezone.utc)


def load_calendar(conn: sqlite3.Connection, finra: list[dict] | None,
                  det: dict | None, now: datetime) -> tuple[list[dict], list[dict]]:
    """未来事件日历：财报（earnings_cal）+ FOMC（fed_fomc 库内未来场次）+
    FINRA 下期空头发布（等待板同源推算）+ 标定期满日。

    返回 (窗口内条目, 更远条目)；每条 {date, kind, label, detail}。
    """
    today = now.astimezone(ET).date()
    horizon = today + timedelta(days=CAL_HORIZON_D)
    items: list[dict] = []
    if has_table(conn, "events"):
        # FOMC：未来场次（库内 fed_fomc 到 2027）
        for r in conn.execute(
            """SELECT event_time_utc, type, title FROM events
               WHERE source_id = 'fed_fomc' AND event_time_utc >= ?
               ORDER BY event_time_utc LIMIT 4""",
            (now.isoformat(timespec="seconds"),),
        ):
            dt = parse_ts(r["event_time_utc"])
            if dt is None:
                continue
            items.append({
                "date": dt.astimezone(ET).date(), "kind": "fomc",
                "label": "FOMC 决议" + ("（含 SEP）" if r["type"].endswith("_sep") else ""),
                "detail": "声明惯例 14:00 ET 发布",
            })
        # 财报：earnings_cal 采集的未来最近一场
        r = conn.execute(
            """SELECT event_time_utc, title, payload_json FROM events
               WHERE source_id = 'earnings_cal' AND event_time_utc >= ?
               ORDER BY event_time_utc LIMIT 1""",
            ((now - timedelta(days=1)).isoformat(timespec="seconds"),),
        ).fetchone()
        if r is not None:
            dt = parse_ts(r["event_time_utc"])
            if dt is not None:
                try:
                    win = json.loads(r["payload_json"] or "{}").get("dates") or []
                except ValueError:
                    win = []
                items.append({
                    "date": dt.astimezone(ET).date(), "kind": "earnings",
                    "label": "TSLA 财报（Yahoo 预计）",
                    "detail": ("窗口 " + " ~ ".join(win) + " · " if len(win) > 1 else "")
                              + "盘后惯例 · 官宣前可能漂移",
                })
    # FINRA 下期空头发布（等待板 next_short_period 同源）
    latest = finra[-1] if finra else None
    base_settle_s = (latest["settle"] if latest
                     else (det or {}).get("cur", {}).get("short_settlement"))
    try:
        nxt_settle, nxt_pub = next_short_period(date.fromisoformat(str(base_settle_s)))
        items.append({
            "date": nxt_pub, "kind": "finra",
            "label": "FINRA 空头利益发布",
            "detail": f"结算 {nxt_settle} · 结算+9 交易日 16:00 ET 口径（±1 日）",
        })
    except (TypeError, ValueError):
        pass
    # 标定期满日（探测器恢复出信号）
    if det and det["cur"].get("state") == "CALIBRATING":
        eta = calib_eta(str(det["cur"].get("state_date")),
                        int(det["cur"].get("baseline_days") or 0))
        if eta:
            items.append({"date": eta, "kind": "calib",
                          "label": "探测器标定期满",
                          "detail": f"{CALIB_BDAYS} 交易日基线集齐 · 恢复出信号"})
    items = [it for it in items if it["date"] >= today]
    items.sort(key=lambda it: it["date"])
    near = [it for it in items if it["date"] <= horizon]
    far = [it for it in items if it["date"] > horizon][:3]
    return near, far


def _cal_strip(near: list[dict], far: list[dict], today: date) -> str:
    """日历条：未来 7/30 天（≤7 天高亮）；窗口外「更远」压缩为一行。"""
    kind_zh = {"fomc": "宏观", "earnings": "财报", "finra": "空头", "calib": "值班"}
    rows = []
    for it in near:
        dd = (it["date"] - today).days
        soon = dd <= 7
        when = "今天" if dd == 0 else f"{dd} 天后"
        rows.append(
            f'<div class="cal-row{" soon" if soon else ""}">'
            f'<span class="cal-d num">{esc(str(it["date"]))}</span>'
            f'<span class="cal-in num">{esc(when)}</span>'
            f'<span class="cal-k">{esc(kind_zh.get(it["kind"], "事件"))}</span>'
            f'<span class="cal-l">{esc(it["label"])}</span>'
            f'<span class="cal-ref">{esc(it["detail"])}</span></div>'
        )
    if not rows:
        rows.append('<div class="cal-row"><span class="cal-ref">'
                    f"未来 {CAL_HORIZON_D} 天窗口内无已知日历事件</span></div>")
    far_html = ""
    if far:
        far_txt = " · ".join(
            f"{it['label']} {it['date']}（{(it['date'] - today).days} 天后）" for it in far
        )
        far_html = f'<div class="cal-far">更远：{esc(far_txt)}</div>'
    return (
        '<div class="cal-head">事件日历 · 未来 7 天高亮 / 30 天窗口'
        "<span class=\"h-sub\">财报（earnings_cal）· FOMC（fed_fomc）· "
        "FINRA 空头发布（推算）· 标定期满</span></div>"
        f'<div class="cal">{"".join(rows)}</div>{far_html}'
    )


def render_morning_brief(conn: sqlite3.Connection, det: dict | None,
                         px: dict | None, now: datetime,
                         cal: tuple[list[dict], list[dict]]) -> str:
    """② 晨间简报：自上次开盘以来的变化摘要（P1-5「值班员」动线）。

    内容：新 T0/T1 事件 · 探测器状态变化 · S2 读数变化 · shadow 昨晚会话 ·
    事件日历。无变化时如实显示「无新事件」，不装繁忙。
    """
    open_utc = last_market_open(now)
    open_et = open_utc.astimezone(ET)
    since_iso = open_utc.isoformat(timespec="seconds")
    today_et = now.astimezone(ET).date()

    # -- ① 新 T0/T1 事件（observed 口径：上次开盘后哨兵新看到的高信号情报）
    hi_rows: list[dict] = []
    if has_table(conn, "events") and has_table(conn, "sources"):
        hi_rows = [dict(r) for r in conn.execute(
            """SELECT e.event_time_utc, e.type, e.title, s.tier
               FROM events e JOIN sources s ON s.source_id = e.source_id
               WHERE s.tier IN ('T0','T1') AND e.source_id != 'detector'
                 AND e.observed_time_utc >= ?
               ORDER BY e.observed_time_utc DESC LIMIT 8""",
            (since_iso,),
        )]
    if hi_rows:
        lis = "".join(
            f'<div class="mb-ev">{tier_badge(r["tier"])}'
            f'<span class="mb-t">{esc(_type_label(r["type"]))}</span>'
            f'<span class="mb-title" title="{esc(r["title"] or "")}">'
            f'{esc((r["title"] or r["type"])[:70])}</span></div>'
            for r in hi_rows
        )
        cell_hi = (f'<div class="cx-cell"><div class="cx-k">新 T0/T1 事件'
                   f'<span class="pill sm warn"><span class="dot"></span>'
                   f'{len(hi_rows)} 条</span></div><div class="mb-list">{lis}</div></div>')
    else:
        cell_hi = ('<div class="cx-cell"><div class="cx-k">新 T0/T1 事件</div>'
                   '<div class="cx-v muted">无新事件</div>'
                   '<div class="cx-ref">上次开盘以来 T0/T1 渠道无新入库</div></div>')

    # -- ② 探测器状态变化
    det_sw: list[dict] = []
    if has_table(conn, "events"):
        det_sw = [dict(r) for r in conn.execute(
            """SELECT event_time_utc, title, payload_json FROM events
               WHERE source_id = 'detector' AND type = 'detector_state'
                 AND observed_time_utc >= ?
               ORDER BY observed_time_utc DESC LIMIT 3""",
            (since_iso,),
        )]
    cur_state = (det or {}).get("cur", {}).get("state") or "无状态"
    if det_sw:
        sw_lines = "".join(
            f'<div class="mb-ev"><span class="mb-t">'
            f'{fmt_local(parse_ts(s["event_time_utc"]))}</span>'
            f'<span class="mb-title" title="{esc(s["title"] or "")}">'
            f'{esc((s["title"] or "")[:70])}</span></div>'
            for s in det_sw
        )
        cell_det = (f'<div class="cx-cell warn-cell"><div class="cx-k">探测器状态变化'
                    f'<span class="pill sm warn"><span class="dot"></span>'
                    f'{len(det_sw)} 次</span></div><div class="mb-list">{sw_lines}</div>'
                    f'<div class="cx-ref">当前 {esc(cur_state)}</div></div>')
    else:
        base = (det or {}).get("cur", {}).get("baseline_days")
        sub = (f"标定 {base}/{CALIB_BDAYS} 交易日" if cur_state == "CALIBRATING"
               else "状态机照常")
        cell_det = ('<div class="cx-cell"><div class="cx-k">探测器状态变化</div>'
                    '<div class="cx-v muted">无切换</div>'
                    f'<div class="cx-ref">当前 {esc(cur_state)} · {esc(sub)}</div></div>')

    # -- ③ S2 读数变化（今日 vs 前一收盘：同序列去尾重算）
    s2 = (px or {}).get("s2")
    prev_s2 = None
    if px and s2_reading is not None and len(px.get("closes") or []) > 31:
        try:
            prev_s2 = s2_reading(px["dates"][:-1], px["closes"][:-1])
        except Exception:  # noqa: BLE001
            prev_s2 = None
    if s2:
        delta_txt = ""
        if prev_s2:
            dpp = s2["drawdown_pct"] - prev_s2["drawdown_pct"]
            delta_txt = (f'前一收盘 {prev_s2["drawdown_pct"]:+.1f}% → '
                         f'今 {s2["drawdown_pct"]:+.1f}%（{dpp:+.1f} pp）')
        stat = ("已触发（买入侧停用区）" if s2["triggered"]
                else f'未触发 · 距 −20% 线余量 {s2["margin_pp"]:.1f} pp')
        cell_s2 = (
            f'<div class="cx-cell{" crit" if s2["triggered"] else ""}">'
            '<div class="cx-k">S2 读数变化</div>'
            f'<div class="cx-v num{" crit-text" if s2["triggered"] else ""}">'
            f'{s2["drawdown_pct"]:+.1f}%</div>'
            f'<div class="cx-ref">{esc(delta_txt or "无前值可比")} · {esc(stat)}</div></div>'
        )
    else:
        cell_s2 = ('<div class="cx-cell crit"><div class="cx-k">S2 读数变化</div>'
                   '<div class="cx-v crit-text">无法计算</div>'
                   '<div class="cx-ref">取价失败——缺口不是安全</div></div>')

    # -- ④ shadow 两策略昨晚会话（shadow_status.json 全部策略逐条）
    status = load_shadow_status()
    strategies = (status or {}).get("strategies") or {}
    if strategies:
        s_lines = []
        n_bad = 0
        for name, s in sorted(strategies.items()):
            sess = parse_ts(s.get("session_end") or "")
            s2s = s.get("s2") or {}
            bits = [f"会话 {fmt_local(sess)}" if sess else "会话时刻缺失",
                    f"{s.get('signals', 0)} 信号"]
            if s2s.get("off"):
                bits.append("S2 停用中")
            if s.get("halted"):
                bits.append("已停机")
            cls = ""
            if s.get("error"):
                bits.append("会话异常")
                cls, n_bad = " crit-text", n_bad + 1
            elif sess is None or (now - sess).total_seconds() > 2 * 86400:
                bits.append("超过 2 天无会话")
                cls, n_bad = " warn-text", n_bad + 1
            s_lines.append(f'<div class="mb-ev"><span class="mb-t">{esc(name)}</span>'
                           f'<span class="mb-title{cls}">{esc(" · ".join(bits))}</span></div>')
        cell_sh = (f'<div class="cx-cell{" crit" if n_bad else ""}">'
                   f'<div class="cx-k">shadow 策略会话（{len(strategies)} 策略）</div>'
                   f'<div class="mb-list">{"".join(s_lines)}</div>'
                   f'<div class="cx-ref">shadow_status.json · 更新 '
                   f'{esc((status or {}).get("updated_at") or "—")}</div></div>')
    else:
        cell_sh = ('<div class="cx-cell crit"><div class="cx-k">shadow 策略会话</div>'
                   '<div class="cx-v crit-text">无记录</div>'
                   '<div class="cx-ref">shadow_status.json 缺失或为空——'
                   "昨晚会话结果无从谈起</div></div>")

    no_change = (not hi_rows and not det_sw and strategies
                 and not any(s.get("error") for s in strategies.values()))
    lede = ("自上次开盘以来<b>无新 T0/T1 事件、无状态切换</b>——安静的一夜。"
            if no_change else "自上次开盘以来的变化如下——只报有据，不装繁忙。")
    return f"""
<section>
  <h2><span class="sec-no">__NO__</span>晨间简报<span class="h-sub">自上次开盘 {esc(open_et.strftime("%m-%d %H:%M"))} ET 以来 · 生成于 {esc(str(today_et))}（ET）</span></h2>
  <div class="card cx">
    <p class="statement">{lede}</p>
    <div class="cx-grid">{cell_hi}{cell_det}{cell_s2}{cell_sh}</div>
    {_cal_strip(cal[0], cal[1], today_et)}
    <p class="footnote">口径：「新事件」按哨兵入库时刻（observed）统计，回填的老事件
    不会伪装成新闻；「上次开盘」为工作日 09:30 ET 的 busday 近似（不剔美股假日）。
    S2 前值 = 同一价格序列去掉最新一根重算，非独立存档。日历中财报/FINRA 发布日
    为推算或第三方预计，以官方公告为准。</p>
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
        '<p class="footnote" style="margin:2px 16px 6px">仓位口径（P1-6 声明）：'
        "REDUCE = 假想<b>全仓转现金</b>、RESTORE = 全仓买回——即 N3-H 应用 A 的判分"
        "口径（B&amp;H 对成本线），<b>不是部分减仓建议</b>；真实仓位梯度未经检验，"
        "出信号时减多少由人工决定。</p>"
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


def render_detector(data: dict | None, now: datetime, wait_html: str = "",
                    nitter_alert: str | None = None) -> str:
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
            "空头 up-jump × Musk 密集同窗命中，<b>假想减仓生效</b>"
            "（口径：假想全仓→现金，应用 A，非部分减仓建议）；"
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
    if nitter_alert:  # 放风腿依赖告警（P1-4）：x_nitter 连败 → 腿 B 断供
        legnote += (
            '<div class="legnote depwarn"><svg class="ic" aria-hidden="true">'
            f'<use href="#i-mega"/></svg><span><b>{esc(nitter_alert)}</b></span></div>'
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
  {wait_html}
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


def _src_card(h: dict, now: datetime, show_weight: bool = True,
              extra: dict | None = None) -> str:
    """渠道卡；extra（P1-4 上卡）：{"flag": (级别, 文本), "dep": 下游依赖文本}。"""
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
    extra = extra or {}
    flag_html = ""
    if extra.get("flag"):
        lvl, txt = extra["flag"]
        flag_html = f'<div class="sc-flag {esc(lvl)}">质量旗标 · {esc(txt)}</div>'
    dep_html = ""
    if extra.get("dep"):
        dep_lvl = extra.get("dep_lvl", "")
        dep_html = f'<div class="sc-flag {esc(dep_lvl)}">下游依赖 · {esc(extra["dep"])}</div>'
    return (
        f'<div class="src-card{card_cls}">'
        f'<div class="sc-top"><strong>{esc(h["source_id"])}</strong>'
        f'<span class="pill sm {status_cls}"><span class="dot"></span>{status_txt}</span></div>'
        f'<div class="sc-name" title="{esc(h["name"])}">{esc(h["name"])}</div>'
        f'<div class="sc-met">{weight_html}'
        f'<span>事件 <span class="num">{h["n_events"]:,}</span></span>{streak_html}</div>'
        f'<div class="sc-sub">{poll_html}' + (f" · {detail}" if detail else "") + "</div>"
        f"{flag_html}{dep_html}"
        "</div>"
    )


# 衍生信号（detector 等）：是"结论"不是"渠道"，移出层级矩阵单列（P0-4）
DERIVED_SOURCE_IDS = {"detector"}


def render_health(rows: list[dict], now: datetime,
                  extras: dict[str, dict] | None = None) -> str:
    """④ 渠道矩阵：按 T0-T3 分组的卡片组 + 衍生信号单列小组。

    extras（P1-4）：{source_id: {"flag"/"dep"...}} 质量旗标与下游依赖上卡。
    """
    if not rows:
        return ""
    extras = extras or {}
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
        cards = "".join(
            _src_card(h, now, extra=extras.get(h["source_id"])) for h in groups[t]
        )
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


# ---- 情报流治理（P1-1）：类型中文化 / 标题指纹去重 / 分层配额 / 层级筛选 ----

TYPE_LABEL = {  # 内部类型代号 → 人话（P2-3）；前缀规则见 _type_label()
    "short_interest": "空头利益", "ats_weekly": "暗池周报",
    "options_snapshot": "期权快照", "polymarket_odds": "预测市场",
    "form4": "Form 4", "form144": "Form 144",
    "sc13d": "13D", "sc13g": "13G", "sc13d_amend": "13D/A", "sc13g_amend": "13G/A",
    "fomc_meeting": "FOMC 日历", "fomc_meeting_sep": "FOMC 日历",
    "earnings_date": "财报日历", "detector_state": "探测器状态",
    "detector_split_guard": "拆股防护", "x_musk_post": "Musk 发帖",
    "x_musk_rt": "Musk 转推", "x_tesla_co_post": "Tesla 官号",
    "x_tesla_co_rt": "Tesla 官号转推",
}


def _type_label(t: str | None) -> str:
    t = t or ""
    if t in TYPE_LABEL:
        return TYPE_LABEL[t]
    if t.startswith("news_"):
        return "新闻"
    if t.startswith("youtube"):
        return "视频"
    if t.startswith("8k"):
        return "8-K"
    if t.startswith("uspto") or t.startswith("patent"):
        return "专利"
    return t or "—"


def _src_display(e: dict) -> str:
    """来源显示名（P2-3）：新闻取 payload.source_attr（媒体名）；否则渠道类型人话。"""
    t = e.get("type") or ""
    if t.startswith("news_"):
        try:
            attr = (json.loads(e.get("payload_json") or "{}").get("source_attr")
                    or "").strip()
        except ValueError:
            attr = ""
        if attr:
            return attr
        # Yahoo feed 无 source_attr，标题尾部也无媒体名——按 feed 标注
        return {"news_yahoo_tsla": "Yahoo Finance", "news_cnbc_top": "CNBC",
                "news_cnbc_markets": "CNBC", "news_marketwatch_top": "MarketWatch",
                "news_google_news_tsla": "Google News"}.get(t, "新闻")
    return _type_label(t)


_TITLE_SEP_RE = re.compile(r"\s+[-–—|]\s+")
_TITLE_WORD_RE = re.compile(r"[a-z0-9]+")


def _title_fingerprint(title: str) -> tuple[str, frozenset]:
    """标题指纹：去尾部「 - 媒体名」段 → 小写词序列 + 词集（去重判据用）。"""
    t = (title or "").strip()
    last = None
    for last in _TITLE_SEP_RE.finditer(t):
        pass
    if last is not None and 0 < len(t) - last.end() <= 42:
        t = t[: last.start()]
    words = _TITLE_WORD_RE.findall(t.lower())
    return " ".join(words), frozenset(words)


def _is_dup_title(a: tuple[str, frozenset], b: tuple[str, frozenset]) -> bool:
    """同题判据：词集 Jaccard ≥ 0.6，或规范化标题互为前缀（≥20 字符）。"""
    sa, wa = a
    sb, wb = b
    if not wa or not wb:
        return False
    if len(sa) >= 20 and len(sb) >= 20 and (sa.startswith(sb) or sb.startswith(sa)):
        return True
    inter = len(wa & wb)
    union = len(wa | wb)
    return union > 0 and inter / union >= 0.6


def dedupe_timeline(rows: list[dict]) -> list[dict]:
    """新闻类事件按「同 ET 日 × 标题相似」折叠：保留最早来源，其余记为「另 N 源」。

    仅对 news_* 类型做（短帖/申报等不适用相似合并）。每条产出行附加：
      dup_n     被折叠的重复条数（0 = 无重复）
      dup_srcs  被折叠条目的来源名列表（悬停展示）
    """
    groups: dict[date, list[dict]] = {}  # ET 日 → 该日已见组
    out: list[dict] = []
    for e in rows:
        e = dict(e)
        e["dup_n"], e["dup_srcs"] = 0, []
        t = e.get("type") or ""
        et_dt = parse_ts(e.get("event_time_utc"))
        if not t.startswith("news_") or et_dt is None or not e.get("title"):
            out.append(e)
            continue
        day = et_dt.astimezone(ET).date()
        fp = _title_fingerprint(e["title"])
        hit = None
        for g in groups.get(day, []):
            if _is_dup_title(fp, g["_fp"]):
                hit = g
                break
        if hit is None:
            e["_fp"] = fp
            e["_et"] = et_dt
            groups.setdefault(day, []).append(e)
            out.append(e)
            continue
        # 折叠进已有组；保留 event_time 最早的一条为代表
        if et_dt < hit["_et"]:  # 当前条更早 → 换代表（rows 按 observed 倒序，可能出现）
            hit["dup_srcs"].append(_src_display(hit))
            for k in ("observed_time_utc", "event_time_utc", "source_id", "type",
                      "title", "url", "payload_json", "tier"):
                hit[k] = e.get(k)
            hit["_et"] = et_dt
            hit["_fp"] = fp
        else:
            hit["dup_srcs"].append(_src_display(e))
        hit["dup_n"] += 1
    for e in out:
        e.pop("_fp", None)
        e.pop("_et", None)
    return out


def _feed_item(e: dict, now: datetime) -> str:
    """情报流单行：event 时刻相对时间（悬停看绝对/入库时刻）+ 层级徽章 +
    来源名 + 标题 + 类型人话 + 「另 N 源」折叠角标。"""
    ev_t = parse_ts(e.get("event_time_utc"))
    obs = parse_ts(e.get("observed_time_utc"))
    title = e.get("title") or e.get("type") or "(无标题)"
    url = e.get("url")
    title_html = (
        f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>'
        if url
        else esc(title)
    )
    derived = e["source_id"] in DERIVED_SOURCE_IDS
    badge = (
        '<span class="tier tier-d" title="衍生信号（探测器结论，非情报渠道）">信号</span>'
        if derived
        else tier_badge(e.get("tier"))
    )
    tier_key = "d" if derived else (e.get("tier") or "T3")[-1]
    fold = ""
    if e.get("dup_n"):
        srcs = "、".join(dict.fromkeys(e["dup_srcs"])) or "同题来源"
        fold = (f'<span class="fold" title="同题折叠：{esc(srcs)}">'
                f"另 {e['dup_n']} 源</span>")
    time_title = (f"事件时刻 {fmt_local(ev_t)} · 入库 {fmt_local(obs)}"
                  if ev_t else f"入库 {fmt_local(obs)}")
    t_show = ev_t or obs
    return (
        f'<li class="ev" data-tier="{esc(tier_key)}">'
        f'<span class="ev-time num" data-iso="{esc(t_show.isoformat() if t_show else "")}" '
        f'title="{esc(time_title)}"><span class="rel">{esc(fmt_ago(t_show, now))}</span></span>'
        f"{badge}"
        f'<span class="ev-src" title="{esc(e["source_id"])}">{esc(_src_display(e))}</span>'
        f'<span class="ev-title">{title_html}{fold}</span>'
        f'<span class="ev-type micro" title="{esc(e.get("type") or "")}">'
        f'{esc(_type_label(e.get("type")))}</span>'
        "</li>"
    )


T3_QUOTA = 25  # T2/T3 流默认展示条数（其余折叠进「展开更早」）


def render_timeline(rows: list[dict], hi_rows: list[dict], now: datetime) -> str:
    """⑤ 最新情报流（P1-1 治理版）：高信号置顶区（T0/T1/衍生信号，独立查询，
    永不被淹没）+ T2/T3 限额流（同题跨源去重折叠，可展开）+ 层级筛选按钮。"""
    if not rows and not hi_rows:
        return ""
    # 高信号区：独立时间线（load_hi_timeline），单一渠道限 5 条
    # （防 polymarket 快照类刷屏挤占 EDGAR/FINRA/探测器），共 20 条
    hi, per_src = [], {}
    for e in hi_rows:
        sid = e["source_id"]
        if sid not in DERIVED_SOURCE_IDS and per_src.get(sid, 0) >= 5:
            continue
        per_src[sid] = per_src.get(sid, 0) + 1
        hi.append(e)
        if len(hi) >= 20:
            break
    hi_ids = {e.get("event_id") for e in hi}
    deduped = dedupe_timeline(
        [e for e in rows if e.get("event_id") not in hi_ids]
    )
    n_folded = sum(e["dup_n"] for e in deduped)
    lo = deduped
    lo_head, lo_rest = lo[:T3_QUOTA], lo[T3_QUOTA:]

    chips = "".join(
        f'<button class="lg fd-chip{" act" if v == "all" else ""}" data-tf="{v}" '
        f'type="button">{lab}</button>'
        for v, lab in (("all", "全部"), ("0", "T0"), ("1", "T1"), ("2", "T2"),
                       ("3", "T3"), ("d", "信号"))
    )
    hi_html = ("".join(_feed_item(e, now) for e in hi)
               or '<li class="ev"><span class="ev-title muted">'
                  "窗口内无 T0/T1/衍生信号事件</span></li>")
    lo_html = "".join(_feed_item(e, now) for e in lo_head)
    more_html = (
        f'<details class="feed-more"><summary>展开更早 {len(lo_rest)} 条</summary>'
        f'<ol class="feed">{"".join(_feed_item(e, now) for e in lo_rest)}</ol></details>'
        if lo_rest else ""
    )
    return f"""
<section id="feed-sec">
  <h2><span class="sec-no">__NO__</span>最新情报流<span class="h-sub">近 {len(rows)} 条 → 去重折叠 {n_folded} 条 · 高信号置顶 · T2/T3 限额 {T3_QUOTA} 条可展开 · 时间列 = 事件时刻（悬停看入库时刻）</span></h2>
  <div class="card">
    <div class="feed-bar" id="feed-bar" role="group" aria-label="层级筛选">
      <span class="micro muted">层级筛选</span>{chips}
    </div>
    <div class="feed-lab">高信号置顶 · T0 布局痕迹 / T1 法定披露 / 衍生信号</div>
    <ol class="feed feed-hi">{hi_html}</ol>
    <div class="feed-lab">T2 / T3 流 · 同题跨源已折叠（「另 N 源」悬停看来源）</div>
    <ol class="feed">{lo_html}</ol>
    {more_html}
    <p class="footnote">去重口径：news_* 事件按「同 ET 日 + 标题指纹相似（词集
    Jaccard ≥ 0.6 或互为前缀）」折叠为一组，保留 event_time 最早的来源，其余计入
    「另 N 源」；申报/短帖类不做相似合并。置顶区固定收录 T0/T1/衍生信号
    （最多 20 条，单一渠道限 5 条防快照类刷屏），不受 T3 噪声挤占，
    溢出条目仍在下方流与层级筛选中。</p>
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
  --bg:#f2efe6; --surface:#faf8f1; --surface-2:#e9e3d3;
  --ink:#182028; --ink-2:#454f57; --muted:#77796c;
  --border:rgba(103,84,42,.28); --grid:#ddd5bf; --baseline:#b9ad8e;
  --good:#0c7a52; --good-text:#0a6746; --good-wash:rgba(12,122,82,.10);
  --warn:#b3880f; --warn-text:#84630a; --warn-wash:rgba(179,136,15,.13);
  --crit:#b13e06; --crit-text:#9a3708; --crit-wash:rgba(177,62,6,.08);
  --off:#77796c;
  --accent:#8a6520; --accent-mark:#a0761d; --accent-wash:rgba(154,116,32,.10);
  --accent-rule:rgba(138,101,32,.38);
  --tier0:#3a2e14; --tier0-ink:#f5efdd;
  --tier1:#6c573a; --tier1-ink:#f5efdd;
  --tier2:#a1762a; --tier2-ink:#ffffff;
  --tier3:#c9ac64; --tier3-ink:#3a2e14;
  --m-insider:#7961be; --m-pit:#418d59; --m-fake:#c45a10; --m-hypo:#ab3378;
  --link:#8a6520;
"""

_CSS = """
/* ===== tokens（设计稿色板：暗色默认，亮色经 data-theme 或系统偏好） ===== */
:root {
  color-scheme: dark;
  --bg:#0a0f14; --surface:#121b22; --surface-2:#18242e;
  --ink:#f2efe6; --ink-2:#c9c2ae; --muted:#8a8c82;
  --border:rgba(214,175,110,.22); --grid:#22303a; --baseline:#46525b;
  --good:#7fd4a0; --good-text:#7fd4a0; --good-wash:rgba(127,212,160,.13);
  --warn:#dc9838; --warn-text:#dc9838; --warn-wash:rgba(220,152,56,.13);
  --crit:#da544a; --crit-text:#e0685e; --crit-wash:rgba(224,104,94,.14);
  --off:#8a8c82;
  --accent:#d6af6e; --accent-mark:#d6af6e; --accent-wash:rgba(214,175,110,.10);
  --accent-rule:rgba(214,175,110,.32);
  --tier0:#f0e9dc; --tier0-ink:#3a2e14;
  --tier1:#d9c289; --tier1-ink:#3a2e14;
  --tier2:#d8a354; --tier2-ink:#2a1e08;
  --tier3:#6c573a; --tier3-ink:#f5efdd;
  --m-insider:#a590d8; --m-pit:#5fb287; --m-fake:#df8448; --m-hypo:#d4529a;
  --link:#d6af6e;
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
.brand > svg.ic { width: 26px; height: 26px; color: var(--accent); }
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
.themebtn:hover { color: var(--accent); border-color: var(--accent); }
.banner { flex: 1; padding: 8px 12px; border: 1px solid var(--crit);
  border-radius: 4px; background: var(--crit-wash); color: var(--crit-text);
  font-size: 13px; }

/* ===== sections（宋体衬线板块题 + 等宽序号 + 延展线） ===== */
section { margin-top: 48px; }
h2 { display: flex; align-items: baseline; gap: 10px; margin: 0 0 14px;
  font: 600 17px/1.4 var(--font-serif); letter-spacing: .05em; flex-wrap: wrap; }
h2 .sec-no { font: 400 11px var(--font-mono); letter-spacing: .18em; color: var(--accent); }
h2 .h-sub { font: 400 12px var(--font-sans); color: var(--muted); letter-spacing: 0; }
h2::after { content: ""; flex: 1; border-top: 1px solid var(--accent-rule);
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
.px b { font-size: 16px; color: var(--accent); font-weight: 600; }
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
.sym-tab.act { background: var(--accent-wash); color: var(--accent);
  border-color: var(--accent); }
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
.tv-rb.act { background: var(--accent-wash); color: var(--accent); }
.tv-layers { display: flex; gap: 8px; flex-wrap: wrap; }
.lg { display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
  user-select: none; appearance: none; font: inherit; border: 1px solid var(--border);
  border-radius: 4px; padding: 4px 10px; font-size: 12px; color: var(--ink-2);
  background: var(--surface-2); }
.lg svg { width: 20px; height: 14px; flex: none; }
.lg .ct { font-family: var(--font-mono); font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums; }
.lg.off { opacity: .38; border-style: dashed; }
.lg:hover:not([disabled]) { border-color: var(--accent); }
.lg[disabled] { opacity: .38; cursor: not-allowed; border-style: dashed; }
.tv-wrap { position: relative; padding: 2px 0 0; }
.tv-view { display: none; }
.tv-view.act { display: block; }
.tv-svg { width: 100%; height: auto; display: block; }
.tv-grid line { stroke: var(--grid); stroke-width: 1; }
.tv-tick { font-family: var(--font-mono); font-size: 10.5px; fill: var(--muted);
  font-variant-numeric: tabular-nums; }
.tv-price { fill: none; stroke: var(--accent-mark); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round; }
.tv-end { fill: var(--accent-mark); }
.tv-endlab { font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  fill: var(--accent); }
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
.xh circle { fill: none; stroke: var(--accent); stroke-width: 1.6; }
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
.wt-bar span { display: block; height: 100%; background: var(--accent-mark); }
.wtf-0, .wtf-1, .wtf-2, .wtf-3 { background: var(--accent-mark); }

/* ===== 晨间简报（cx 复用）+ 事件日历条 ===== */
.mb-list { display: flex; flex-direction: column; gap: 3px; margin-top: 8px; }
.mb-ev { display: flex; align-items: baseline; gap: 8px; font-size: 12px; min-width: 0; }
.mb-ev .tier { flex: none; }
.mb-t { flex: none; font-family: var(--font-mono); font-size: 10.5px;
  color: var(--muted); letter-spacing: .04em; }
.mb-title { min-width: 0; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; color: var(--ink-2); }
.cal-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  font: 600 11px var(--font-mono); letter-spacing: .14em; color: var(--muted);
  margin: 16px 0 8px; }
.cal { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.cal-row { display: grid; grid-template-columns: 100px 72px 44px minmax(120px, .8fr)
  minmax(0, 1.4fr); gap: 10px; align-items: baseline; padding: 7px 12px;
  border-bottom: 1px solid var(--grid); font-size: 12.5px; }
.cal-row:last-child { border-bottom: none; }
.cal-row.soon { background: var(--warn-wash); }
.cal-d { font-size: 12px; color: var(--ink-2); }
.cal-in { font-size: 11px; color: var(--muted); }
.cal-row.soon .cal-in { color: var(--warn-text); font-weight: 700; }
.cal-k { font-family: var(--font-mono); font-size: 10px; letter-spacing: .1em;
  color: var(--muted); border: 1px solid var(--border); border-radius: 3px;
  padding: 1px 6px; justify-self: start; }
.cal-l { color: var(--ink); }
.cal-ref { font-size: 11px; color: var(--muted); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
@media (max-width: 760px){ .cal-row { grid-template-columns: 92px 64px 1fr; }
  .cal-l { grid-column: 3; } .cal-ref { grid-column: 3; white-space: normal; } }
.cal-far { font-size: 11.5px; color: var(--muted); margin-top: 6px;
  font-family: var(--font-mono); }
.warn-cell { background: var(--warn-wash); }

/* ===== 渠道卡质量旗标 / 下游依赖（P1-4） ===== */
.sc-flag { font-size: 11px; line-height: 1.5; color: var(--muted);
  border-top: 1px dashed var(--border); padding-top: 4px; margin-top: 2px; }
.sc-flag.warn { color: var(--warn-text); }
.sc-flag.crit { color: var(--crit-text); font-weight: 600; }
.legnote.depwarn { margin-top: 8px; padding: 8px 10px; border: 1px solid var(--crit);
  border-radius: 5px; background: var(--crit-wash); color: var(--crit-text); }
.legnote.depwarn svg.ic { color: var(--crit); }

/* ===== 04 最新情报流 ===== */
.feed-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 10px 16px 2px; }
.fd-chip { padding: 3px 12px; }
.fd-chip.act { border-color: var(--accent); color: var(--accent);
  background: var(--accent-wash); box-shadow: inset 0 0 0 1px var(--accent); }
.feed-lab { font: 600 10.5px var(--font-mono); letter-spacing: .16em;
  color: var(--muted); padding: 10px 16px 0; display: flex; align-items: center;
  gap: 10px; }
.feed-lab::after { content: ""; flex: 1; border-top: 1px dashed var(--border); }
.feed-hi { background: var(--surface-2); border-block: 1px solid var(--grid);
  margin-top: 8px; }
.fold { font-family: var(--font-mono); font-size: 10px; color: var(--warn-text);
  border: 1px solid var(--warn); border-radius: 3px; background: var(--warn-wash);
  padding: 0 6px; margin-left: 8px; white-space: nowrap; cursor: help; }
.feed-more { margin: 0; }
.feed-more summary { cursor: pointer; color: var(--muted); font-size: 12px;
  padding: 8px 16px; }
.feed-more summary:hover { color: var(--ink-2); }
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

/* ===== 视图切换（实时值班｜模拟探测） ===== */
main > .sym-tabs { margin-top: 26px; }
.viewbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin: 12px 0 0; padding-bottom: 14px; border-bottom: 1px solid var(--border); }
.viewtabs { display: inline-flex; border: 1px solid var(--border); border-radius: 5px;
  overflow: hidden; }
.vt { appearance: none; border: none; background: transparent; cursor: pointer;
  color: var(--muted); font: 600 12.5px var(--font-sans); letter-spacing: .05em;
  padding: 8px 20px; border-right: 1px solid var(--border); }
.vt:last-child { border-right: none; }
.vt:hover { color: var(--ink-2); }
.vt.act { background: var(--accent-wash); color: var(--accent); }
.vw { display: none; } .vw.act { display: block; }

/* ===== 等待板（值班台账） ===== */
.waitboard { margin-top: 12px; padding: 16px 18px 8px; }
.wb-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 4px; }
.wb-head h3 { display: flex; align-items: center; gap: 8px; margin: 0; }
.wb-head h3 svg.ic { width: 15px; height: 15px; color: var(--muted); }
.wb-sec { font-family: var(--font-mono); font-size: 10.5px; letter-spacing: .18em;
  color: var(--muted); margin: 14px 0 8px; display: flex; align-items: center; gap: 10px; }
.wb-sec::after { content: ""; flex: 1; border-top: 1px dashed var(--border); }
.wb-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.wb-grid.three { grid-template-columns: repeat(3, 1fr); }
@media (max-width: 1080px){ .wb-grid, .wb-grid.three { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px){ .wb-grid, .wb-grid.three { grid-template-columns: 1fr; } }
.waitboard .cx-v { font-size: 17px; }
.playbook { display: flex; flex-direction: column; gap: 8px; }
.pb-row { display: grid; grid-template-columns: 40px minmax(0, 1.1fr) 52px minmax(0, 1.3fr) 118px;
  gap: 10px; align-items: baseline; background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 6px; padding: 9px 12px; }
@media (max-width: 900px){ .pb-row { grid-template-columns: 40px 1fr;
  grid-auto-rows: auto; } .pb-row .pb-src { grid-column: 2; } }
.pb-chip { font-family: var(--font-mono); font-size: 10px; font-weight: 700;
  letter-spacing: .1em; text-align: center; border-radius: 3px; padding: 2px 0;
  align-self: center; }
.pb-chip.if { background: var(--warn-wash); color: var(--warn-text);
  border: 1px solid var(--warn); }
.pb-chip.then { background: var(--crit-wash); color: var(--crit-text);
  border: 1px solid var(--crit); }
.pb-cond { font-size: 12.5px; color: var(--ink-2); line-height: 1.55; }
.pb-act { font-size: 12.5px; color: var(--ink-2); line-height: 1.55; }
.pb-src { font-family: var(--font-mono); font-size: 10.5px; color: var(--muted);
  letter-spacing: .04em; text-align: right; align-self: center; white-space: nowrap; }

/* ===== 模拟探测（历史推演） ===== */
.rp-spec { padding: 14px 18px; margin-bottom: 12px; }
.rp-spec-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
@media (max-width: 1080px){ .rp-spec-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 620px){ .rp-spec-grid { grid-template-columns: 1fr; } }
.rp-spec .bt { display: flex; gap: 8px; align-items: center; font-size: 12px;
  color: var(--ink-2); font-weight: 600; margin-bottom: 4px; }
.rp-spec p { margin: 0; font-size: 12px; color: var(--muted); line-height: 1.65; }
.rp-spec .chip { margin-right: 4px; }
.rp-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  margin-bottom: 12px; }
.rp-btn { appearance: none; cursor: pointer; font: 600 12px var(--font-mono);
  letter-spacing: .08em; color: var(--ink-2); background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 4px; padding: 7px 16px; }
.rp-btn:hover { border-color: var(--accent); color: var(--accent); }
.rp-btn.primary { background: var(--accent-wash); border-color: var(--accent);
  color: var(--accent); }
#rp-range { flex: 1; min-width: 160px; accent-color: var(--accent); }
.rp-date { font-size: 12.5px; color: var(--ink); }
.rp-count { font-size: 11px; color: var(--muted); }
.rp-trigmarks line { stroke: var(--warn); stroke-width: 2; }
.rp-cursor line { stroke: var(--ink); stroke-width: 1.2; stroke-dasharray: 4 3;
  opacity: .85; }
.rp-cursor circle { fill: var(--ink); stroke: var(--surface); stroke-width: 1.5; }
.rp-curlab { font-family: var(--font-mono); font-size: 11px; font-weight: 600;
  fill: var(--ink); }
.rp-panel { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;
  margin-top: 12px; }
@media (max-width: 860px){ .rp-panel { grid-template-columns: 1fr; } }
#rp-legb.blind { opacity: .55; }
#rp-legb.blind .val { color: var(--muted); font-size: 15px; }
.rp-statecell .val { margin: 10px 0 6px; }
.rp-statecell .pill { font-size: 13px; padding: 5px 14px; }
/* -- 推演日志 + 假想交易记录 -- */
.rp-lower { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: 12px; margin-top: 14px; }
@media (max-width: 980px){ .rp-lower { grid-template-columns: 1fr; } }
.rp-logcard, .rp-tradecard { border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); min-width: 0; }
.lg-h { display: flex; align-items: baseline; gap: 10px; justify-content: space-between;
  font: 600 11px var(--font-mono); letter-spacing: .1em; color: var(--ink-2);
  padding: 9px 12px; border-bottom: 1px solid var(--border); }
.lg-cnt { color: var(--muted); font-weight: 500; letter-spacing: .04em; }
.lg-empty { font-size: 12px; color: var(--muted); padding: 14px 12px; margin: 0;
  line-height: 1.7; }
.lg-scroll { overflow-y: auto; max-height: 320px; }
.lg-row { border-bottom: 1px solid var(--border); }
.lg-row:last-child { border-bottom: 0; }
.lg-line { display: flex; align-items: baseline; gap: 8px; padding: 6px 12px;
  font-size: 12px; }
.lg-row.click .lg-line { cursor: pointer; }
.lg-row.click .lg-line:hover { background: var(--surface-2); }
.lg-d { font-size: 11px; color: var(--muted); flex: 0 0 76px; }
.lg-tag { font: 600 10.5px var(--font-mono); letter-spacing: .05em; border-radius: 3px;
  padding: 2px 7px; white-space: nowrap; background: var(--surface-2);
  color: var(--ink-2); border: 1px solid var(--border); }
.lg-trig .lg-tag { background: var(--warn-wash); border-color: var(--warn);
  color: var(--warn-text); }
.lg-chain .lg-tag { border-color: var(--warn); color: var(--warn-text); }
.lg-on .lg-tag { background: var(--good-wash); border-color: var(--good);
  color: var(--good-text); }
.lg-blind .lg-tag { color: var(--muted); }
.lg-b { color: var(--ink-2); min-width: 0; line-height: 1.55; }
.lg-x { margin-left: auto; color: var(--muted); font-size: 10px; flex: 0 0 auto;
  transition: transform .15s; }
.lg-row.open .lg-x { transform: rotate(90deg); }
.lg-exp { padding: 2px 12px 10px 96px; }
.lg-kv { display: grid; grid-template-columns: 72px 1fr; gap: 8px; font-size: 12px;
  padding: 3px 0; }
.lg-kv > span:first-child { font-family: var(--font-mono); font-size: 10px;
  letter-spacing: .08em; color: var(--muted); padding-top: 2px; }
.lg-kv > span:last-child { color: var(--ink-2); line-height: 1.6; min-width: 0; }
.lg-trade .lg-line { font-weight: 600; }
.lg-trade.sell .lg-line { background: var(--crit-wash); }
.lg-trade.buy .lg-line { background: var(--good-wash); }
.lg-trade.sell .lg-tag { background: var(--surface); border-color: var(--crit);
  color: var(--crit-text); }
.lg-trade.buy .lg-tag { background: var(--surface); border-color: var(--good);
  color: var(--good-text); }
.rp-tradecard { overflow-x: auto; }
.rp-trades { width: 100%; min-width: 0; border-collapse: collapse;
  font-size: 12px; table-layout: fixed; }
.rp-trades th, .rp-trades td { white-space: normal; }
.rp-trades th { font: 600 10px var(--font-mono); letter-spacing: .06em;
  color: var(--muted); text-align: left; padding: 7px 8px;
  border-bottom: 1px solid var(--border); }
.rp-trades th:nth-child(1) { width: 96px; }
.rp-trades th:nth-child(2) { width: 58px; }
.rp-trades th:nth-child(3) { width: 64px; }
.rp-trades th:nth-child(5) { width: 78px; }
.rp-trades td.num { white-space: nowrap; }
.rp-trades td { padding: 7px 8px; border-bottom: 1px solid var(--border);
  color: var(--ink-2); vertical-align: top; line-height: 1.5;
  overflow-wrap: anywhere; }
.rp-trades tbody tr:last-child td { border-bottom: 0; }
.rp-trades .rp-basis { font-size: 11px; color: var(--muted); }
.rp-res { font-weight: 700; }
/* -- 推演总结卡 -- */
.rp-sum { position: fixed; z-index: 40; top: 50%; left: 50%;
  transform: translate(-50%, -50%); width: min(720px, 94vw); max-height: 86vh;
  overflow-y: auto; background: var(--surface); border: 1px solid var(--warn);
  border-radius: 8px; padding: 14px 20px 16px;
  box-shadow: 0 14px 44px rgba(0,0,0,.4); }
.rs-h { display: flex; align-items: center; gap: 10px; justify-content: space-between;
  border-bottom: 1px solid var(--border); padding-bottom: 9px; margin-bottom: 10px; }
.rs-h > span { font-size: 13.5px; font-weight: 700; color: var(--ink); }
.rs-model { margin: 0 0 4px; font-size: 12.5px; color: var(--muted); line-height: 1.7; }
.rs-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  margin: 12px 0; }
@media (max-width: 640px){ .rs-grid { grid-template-columns: 1fr 1fr; } }
.rs-cell { border: 1px solid var(--border); border-radius: 6px; padding: 9px 11px;
  background: var(--surface-2); }
.rs-k { font: 600 10px var(--font-mono); letter-spacing: .08em; color: var(--muted);
  margin-bottom: 5px; }
.rs-v { font-size: 16px; font-weight: 700; color: var(--ink);
  font-variant-numeric: tabular-nums; }
.rs-v .sub { font-size: 11px; color: var(--muted); font-weight: 500; }
.rs-ep { margin: 6px 0; font-size: 12.5px; color: var(--ink-2); line-height: 1.7; }
.rs-note { border: 1px solid var(--warn); background: var(--warn-wash);
  border-radius: 6px; padding: 10px 12px; margin-top: 12px; font-size: 12.5px;
  font-weight: 600; color: var(--warn-text); line-height: 1.75; }
/* -- 全系统推演（进攻层 E8-A+S2）与推演模式切换 -- */
.rmodebar { display: flex; align-items: flex-start; gap: 14px; flex-wrap: wrap;
  margin: 14px 0 16px; }
.rmodebar .viewtabs { flex: none; }
.rm-cmp { flex: 1 1 340px; font-size: 12px; color: var(--muted);
  line-height: 1.65; padding-top: 2px; }
.rm-cmp b { color: var(--ink-2); }
.rmode { display: none; } .rmode.act { display: block; }
.lg-tw .lg-line, .lg-tl .lg-line { font-weight: 600; }
.lg-tw .lg-tag { background: var(--good-wash); border-color: var(--good);
  color: var(--good-text); }
.lg-tl .lg-tag { background: var(--crit-wash); border-color: var(--crit);
  color: var(--crit-text); }
.lg-blk .lg-line { opacity: .62; }
.lg-blk .lg-tag { border-style: dashed; color: var(--muted); }
.lg-s2off .lg-tag { background: var(--crit-wash); border-color: var(--crit);
  color: var(--crit-text); }
.lg-s2on .lg-tag { background: var(--good-wash); border-color: var(--good);
  color: var(--good-text); }
.fs-marks line { stroke-width: 2; }
.fs-marks line.w { stroke: var(--good); }
.fs-marks line.l { stroke: var(--crit); }
.fs-marks line.b { stroke: var(--muted); opacity: .7; }
.fs-blktbl th:nth-child(1) { width: 132px; }
.fs-blktbl th:nth-child(2) { width: auto; }
.fs-blktbl th:nth-child(3) { width: auto; }
.fs-blktbl th:nth-child(5) { width: auto; }
.fs-blkrow td { color: var(--muted); }
.fs-blknote { font-size: 11px; color: var(--muted); margin: 8px 10px 10px;
  line-height: 1.6; }
.fs-ann { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  border: 1px solid var(--good); background: var(--good-wash); border-radius: 6px;
  padding: 9px 14px; margin: 10px 0 2px; }
.fs-ann-k { font: 600 10px var(--font-mono); letter-spacing: .08em;
  color: var(--muted); }
.fs-ann-v { font-size: 21px; font-weight: 800; color: var(--good-text);
  font-variant-numeric: tabular-nums; }
.fs-ann-sub { flex: 1 1 260px; font-size: 11px; color: var(--muted);
  line-height: 1.55; }

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

# 视图切换 + 推演播放器（模拟探测页）。不经 %-格式化，直接拼接。
_RP_JS = """
/* 视图切换：实时值班｜模拟探测（页内再分两种推演模式）；location.hash 记忆
   （#live / #replay 探测器推演 / #fullsys 全系统推演），meta refresh 后不丢 */
(function () {
  var tabs = document.querySelectorAll(".vt");
  if (!tabs.length) return;
  var modes = document.querySelectorAll(".rm");
  var curMode = "det";
  function setView(v) {
    tabs.forEach(function (b) {
      var on = b.getAttribute("data-view") === v;
      b.classList.toggle("act", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".vw").forEach(function (w) {
      w.classList.toggle("act", w.id === "view-" + v);
    });
  }
  function setMode(mo) {
    curMode = mo;
    modes.forEach(function (b) {
      var on = b.getAttribute("data-mode") === mo;
      b.classList.toggle("act", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".rmode").forEach(function (w) {
      w.classList.toggle("act", w.id === "rmode-" + mo);
    });
  }
  function writeHash(v) {
    try { history.replaceState(null, "", "#" + v); } catch (e) {}
  }
  tabs.forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.getAttribute("data-view");
      setView(v);
      writeHash(v === "replay" && curMode === "full" ? "fullsys" : v);
    });
  });
  modes.forEach(function (b) {
    b.addEventListener("click", function () {
      setMode(b.getAttribute("data-mode"));
      writeHash(curMode === "full" ? "fullsys" : "replay");
    });
  });
  var h = (location.hash || "").slice(1);
  if (h === "fullsys") { setView("replay"); setMode("full"); }
  else if (h === "replay" || h === "live") setView(h);
})();
/* 推演播放器：逐日回放 N3-H 考场；事件写入推演日志，播放不中断；播完弹总结 */
(function () {
  var sec = document.getElementById("rp");
  var dataEl = document.getElementById("rp-data");
  if (!sec || !dataEl) return;
  var D = null;
  try { D = JSON.parse(dataEl.textContent); } catch (e) {}
  if (!D) return;
  var m = D.meta, n = D.D.length;
  function $(id) { return document.getElementById(id); }
  var curLine = $("rp-cur-line"), curDot = $("rp-cur-dot"), curPx = $("rp-cur-px");
  var range = $("rp-range"), dateLab = $("rp-date"), countLab = $("rp-count");
  var playBtn = $("rp-play"), stepBtn = $("rp-step");
  var log = $("rp-log"), logEmpty = $("rp-log-empty"), logCnt = $("rp-log-cnt");
  var sumCard = $("rp-sum"), sumBtn = $("rp-sum-btn"), sumClose = $("rp-sum-close");
  var idx = 0, timer = null, speed = 1, BASE_MS = 320;
  var logPos = 0, nLog = 0, sumSeen = false;

  var tradeBy = {};
  (D.trades || []).forEach(function (t) {
    if (t.sd) (tradeBy[t.sd] = tradeBy[t.sd] || []).push(["s", t]);
    if (t.bd) (tradeBy[t.bd] = tradeBy[t.bd] || []).push(["b", t]);
  });

  function isoOf(i) {
    return new Date((D.D[i] - D.epoch) * 86400000).toISOString().slice(0, 10);
  }
  function fmtPct(v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; }
  function legBrief(i) {
    var s = [], t = D.T[i], h = D.H[i], si = D.SI[i];
    if (D.B[i] === 1) s.push("放风腿失明");
    else if (t != null && h != null)
      s.push("帖 " + t + (t > h ? " > " : " ≤ ") + h.toFixed(1));
    if (si >= 0) s.push("空头 " + fmtPct(D.R[si][2]));
    return s.join(" · ");
  }
  function mkRow(cls, iso, tag, brief, exp) {
    var d = document.createElement("div");
    d.className = "lg-row " + cls + (exp ? " click" : "");
    d.innerHTML = '<div class="lg-line"><span class="lg-d num">' + iso + "</span>" +
      '<span class="lg-tag">' + tag + "</span>" +
      '<span class="lg-b">' + brief + "</span>" +
      (exp ? '<span class="lg-x num" aria-hidden="true">\\u25b8</span>' : "") +
      "</div>" + (exp ? '<div class="lg-exp" hidden>' + exp + "</div>" : "");
    if (exp) {
      d.querySelector(".lg-line").addEventListener("click", function () {
        var e2 = d.querySelector(".lg-exp");
        e2.hidden = !e2.hidden;
        d.classList.toggle("open", !e2.hidden);
      });
    }
    return d;
  }
  function kv(k, v) {
    return '<div class="lg-kv"><span>' + k + "</span><span>" + (v || "—") +
      "</span></div>";
  }
  function cardExp(c) {
    if (!c) return kv("说明", "日记卡缺失");
    return kv("看到什么", c.s) + kv("决定", c.d) + kv("之后实际", c.a) +
      kv("判定", c.v);
  }
  function dayRows(i) {
    var iso = isoOf(i), out = [];
    if (D.B[i] === 1 && (i === 0 || D.B[i - 1] === 0))
      out.push(mkRow("lg-blind", iso, "失明期进入",
        "Musk 归档尽头——放风腿无数据，无触发 \\u2260 安全",
        kv("说明", "musk_tweets 归档止于 2025-05-08，此后放风腿无法评估、探测器无法" +
          "触发；该段覆盖为空，属数据缺口而非安全判定。")));
    var firstOff = D.S[i] === 1 && (i === 0 || D.S[i - 1] === 0);
    if (D.G[i] === 1) {
      out.push(mkRow(firstOff ? "lg-trig" : "lg-chain", iso,
        firstOff ? "触发 · 避险开始" : "连环确认",
        legBrief(i) + (firstOff ? " · risk_off 生效" : " · F20 顺延"),
        cardExp(D.cards[iso])));
    } else if (firstOff) {
      out.push(mkRow("lg-trig", iso, "避险开始", "risk_off 生效 · F20", ""));
    }
    if (i > 0 && D.S[i] === 0 && D.S[i - 1] === 1)
      out.push(mkRow("lg-on", iso, "避险解除", "F20 期满 · 回到 risk_on", ""));
    (tradeBy[iso] || []).forEach(function (p) {
      var t = p[1];
      if (p[0] === "s") {
        out.push(mkRow("lg-trade sell", iso, "假想卖出 E" + t.ep,
          t.sp == null ? "开盘价缺失" : "开盘 " + t.sp.toFixed(2) + " 全仓转现金",
          kv("依据", t.sb)));
      } else {
        var resTxt = t.res == null ? "" :
          ' · <b class="' + (t.res >= 0 ? "good-text" : "crit-text") + '">' +
          (t.res >= 0 ? "躲掉 " : "错过 ") + fmtPct(t.res) + "</b>";
        out.push(mkRow("lg-trade buy", iso, "假想买回 E" + t.ep,
          (t.bp == null ? "开盘价缺失" : "开盘 " + t.bp.toFixed(2) + " 买回") + resTxt,
          kv("依据", t.bb) + kv("该段结果", t.res == null ? "—" :
            "卖 " + t.sp.toFixed(2) + " → 买 " + t.bp.toFixed(2) + "，" +
            (t.res >= 0 ? "躲掉 " : "错过 ") + fmtPct(t.res) +
            (t.bh == null ? "" : "（该段 B&H " + fmtPct(t.bh) + "）"))));
      }
    });
    return out;
  }
  function appendDay(i) {
    var rs = dayRows(i), k;
    for (k = 0; k < rs.length; k++) log.appendChild(rs[k]);
    nLog += rs.length;
    return rs.length;
  }
  function syncLog(i) {
    var added = 0, k;
    if (i >= logPos) {
      for (k = logPos; k <= i; k++) added += appendDay(k);
    } else {
      log.innerHTML = ""; nLog = 0; sumSeen = false;
      for (k = 0; k <= i; k++) added += appendDay(k);
    }
    logPos = i + 1;
    if (logEmpty) logEmpty.hidden = nLog > 0;
    if (logCnt) logCnt.textContent = nLog + " 条";
    if (added) log.scrollTop = log.scrollHeight;
  }
  function X(i) { return m.ml + (D.D[i] - m.d0) / m.ds * m.pw; }
  function Y(i) {
    return m.mt + (1 - (Math.log(D.C[i]) / Math.LN10 - m.ly0) / m.lys) * m.ph;
  }
  function draw(i) {
    idx = i;
    var x = X(i), y = Y(i);
    curLine.setAttribute("x1", x); curLine.setAttribute("x2", x);
    curDot.setAttribute("cx", x); curDot.setAttribute("cy", y);
    curPx.textContent = D.C[i].toFixed(2);
    if (x > m.ml + m.pw * 0.75) {
      curPx.setAttribute("x", x - 8); curPx.setAttribute("text-anchor", "end");
    } else {
      curPx.setAttribute("x", x + 8); curPx.setAttribute("text-anchor", "start");
    }
    range.value = i;
    dateLab.textContent = isoOf(i);
    countLab.textContent = (i + 1) + " / " + n;
    var si = D.SI[i];
    if (si >= 0) {
      var r = D.R[si];
      $("rp-a-val").textContent = (r[2] >= 0 ? "+" : "") + r[2].toFixed(2) + "%";
      $("rp-a-ref").textContent = "结算 " + r[1] + " · 发布 " + r[0] + "（当日已见的最新一期）";
      var ah = r[2] >= 10;
      $("rp-a-pill").className = "pill sm" + (ah ? " warn" : "");
      $("rp-a-pilltxt").textContent = ah ? "≥+10% 命中" : "未命中";
    } else {
      $("rp-a-val").textContent = "—";
      $("rp-a-ref").textContent = "尚无已发布空头数据";
      $("rp-a-pill").className = "pill sm";
      $("rp-a-pilltxt").textContent = "无数据";
    }
    var legb = $("rp-legb");
    if (D.B[i] === 1) {
      legb.classList.add("blind");
      $("rp-b-val").textContent = "放风腿无数据";
      $("rp-b-ref").textContent = "Musk 归档缺口——该段无法触发，无触发 ≠ 判定安全";
      $("rp-b-pill").className = "pill sm";
      $("rp-b-pilltxt").textContent = "失明期";
    } else {
      legb.classList.remove("blind");
      var t = D.T[i], h = D.H[i];
      $("rp-b-val").textContent = t == null ? "—" : t + " 帖";
      $("rp-b-ref").textContent =
        h == null ? "阈值 —" : "阈值 " + h.toFixed(1) + " 帖/日（过去 365 日 90 分位）";
      var bh = t != null && h != null && t > h;
      $("rp-b-pill").className = "pill sm" + (bh ? " warn" : "");
      $("rp-b-pilltxt").textContent = bh ? ">阈值 命中" : "未命中";
    }
    var off = D.S[i] === 1;
    $("rp-st-pill").className = "pill " + (off ? "crit" : "good");
    $("rp-st-txt").textContent = off ? "RISK_OFF" : "RISK_ON";
    $("rp-st-ref").textContent = off
      ? "假想减仓生效（F20，重叠触发顺延）"
      : (D.B[i] === 1 ? "未见触发——但放风腿失明，覆盖不完整" : "两腿未同时命中");
    syncLog(i);
  }
  function showSum() { if (sumCard) sumCard.hidden = false; }
  function hideSum() { if (sumCard) sumCard.hidden = true; }
  function atEnd() {
    if (!sumSeen) { sumSeen = true; showSum(); }
  }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = idx >= n - 1 ? "重新推演" : "开始推演";
  }
  function tick() {
    if (idx >= n - 1) { pause(); atEnd(); return; }
    draw(idx + 1);
    if (idx >= n - 1) { pause(); atEnd(); }
  }
  function play() {
    if (idx >= n - 1) { hideSum(); draw(0); }
    if (timer) return;
    timer = setInterval(tick, BASE_MS / speed);
    playBtn.textContent = "暂停";
  }
  playBtn.addEventListener("click", function () { if (timer) pause(); else play(); });
  stepBtn.addEventListener("click", function () {
    pause();
    if (idx >= n - 1) return;
    draw(idx + 1);
    if (idx >= n - 1) { pause(); atEnd(); }
  });
  sec.querySelectorAll(".rp-speed").forEach(function (b) {
    b.addEventListener("click", function () {
      sec.querySelectorAll(".rp-speed").forEach(function (x) {
        x.classList.toggle("act", x === b);
      });
      speed = parseFloat(b.getAttribute("data-s")) || 1;
      if (timer) { clearInterval(timer); timer = setInterval(tick, BASE_MS / speed); }
    });
  });
  range.addEventListener("input", function () {
    pause(); hideSum();
    draw(Math.max(0, Math.min(n - 1, parseInt(range.value, 10) || 0)));
  });
  if (sumBtn) sumBtn.addEventListener("click", function () {
    if (sumCard) { if (sumCard.hidden) showSum(); else hideSum(); }
  });
  if (sumClose) sumClose.addEventListener("click", hideSum);
  window.__rp = { draw: draw, play: play, pause: pause, n: n,
                  showSum: showSum, hideSum: hideSum,
                  get idx() { return idx; }, get playing() { return !!timer; },
                  get logCount() { return nLog; } };
  draw(0);
})();
"""


# 全系统推演播放器（进攻层 E8-A+S2）。不经 %-格式化，直接拼接。
_FS_JS = """
/* 全系统推演播放器：E8-A+S2 留出段逐日回放；逐笔交易/S2 拦截/开关切换写入日志 */
(function () {
  var sec = document.getElementById("fs");
  var dataEl = document.getElementById("fs-data");
  if (!sec || !dataEl) return;
  var D = null;
  try { D = JSON.parse(dataEl.textContent); } catch (e) {}
  if (!D) return;
  var m = D.meta, n = D.D.length;
  function $(id) { return document.getElementById(id); }
  var curLine = $("fs-cur-line"), curDot = $("fs-cur-dot"), curPx = $("fs-cur-px");
  var range = $("fs-range"), dateLab = $("fs-date"), countLab = $("fs-count");
  var playBtn = $("fs-play"), stepBtn = $("fs-step");
  var log = $("fs-log"), logEmpty = $("fs-log-empty"), logCnt = $("fs-log-cnt");
  var sumCard = $("fs-sum"), sumBtn = $("fs-sum-btn"), sumClose = $("fs-sum-close");
  var idx = 0, timer = null, speed = 20, BASE_MS = 320; // 交易密：默认 20x
  var logPos = 0, nLog = 0, sumSeen = false;
  var TY = { tp: "止盈", sl: "止损", timeout: "超时平", dayend: "日终平" };

  var tradeBy = {}, detBy = {};
  (D.T || []).forEach(function (t) {
    (tradeBy[t.o] = tradeBy[t.o] || []).push(t);
  });
  (D.DET || []).forEach(function (r) {
    (detBy[r[0]] = detBy[r[0]] || []).push(r);
  });

  function isoOf(i) {
    return new Date((D.D[i] - D.epoch) * 86400000).toISOString().slice(0, 10);
  }
  function fmtPct(v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "%"; }
  function mkRow(cls, iso, tag, brief, exp) {
    var d = document.createElement("div");
    d.className = "lg-row " + cls + (exp ? " click" : "");
    d.innerHTML = '<div class="lg-line"><span class="lg-d num">' + iso + "</span>" +
      '<span class="lg-tag">' + tag + "</span>" +
      '<span class="lg-b">' + brief + "</span>" +
      (exp ? '<span class="lg-x num" aria-hidden="true">\\u25b8</span>' : "") +
      "</div>" + (exp ? '<div class="lg-exp" hidden>' + exp + "</div>" : "");
    if (exp) {
      d.querySelector(".lg-line").addEventListener("click", function () {
        var e2 = d.querySelector(".lg-exp");
        e2.hidden = !e2.hidden;
        d.classList.toggle("open", !e2.hidden);
      });
    }
    return d;
  }
  function kv(k, v) {
    return '<div class="lg-kv"><span>' + k + "</span><span>" + (v || "—") +
      "</span></div>";
  }
  function tradeRow(iso, t) {
    var geo = kv("门控", "GBDT prob " + t.p.toFixed(3) + " ≥ 0.4325（Top≈10%）") +
      kv("几何", "tp +0.5% / sl −2% / 48×5m / 日内强平 · 费 1bp/边 + 滑点 2bp");
    if (t.b === 1) {
      return mkRow("lg-blk", iso, "被开关拦截",
        t.e + " 事件过门（prob " + t.p.toFixed(2) + "）· S2 关闸未成交 · 若成交 " +
        fmtPct(t.r),
        geo + kv("拦截", "S2：距 252 日高点回撤 > 20%（昨收判定）——关闸期不开新仓；" +
          "本笔照常结算仅作对照，不入权益"));
    }
    var win = t.r >= 0;
    return mkRow(win ? "lg-tw" : "lg-tl", iso, "第 " + t.k + " 笔",
      t.e + " 入 " + t.ep.toFixed(2) + " → " + t.x + " 出 " + t.xp.toFixed(2) +
      " · " + (TY[t.ty] || t.ty) + ' · <b class="' +
      (win ? "good-text" : "crit-text") + '">' + fmtPct(t.r) + "</b> · 累计 " +
      fmtPct(t.q),
      geo + kv("结算", t.e + " 开盘入 " + t.ep.toFixed(2) + "（含滑点）→ " + t.x +
        " " + (TY[t.ty] || t.ty) + "出 " + t.xp.toFixed(2) + "，单笔 " + fmtPct(t.r) +
        "，S2 流累计 " + fmtPct(t.q)));
  }
  function dayRows(i) {
    var iso = isoOf(i), o = D.D[i], out = [];
    (detBy[o] || []).forEach(function (r) {
      out.push(mkRow("lg-blind", iso, r[1], r[2], r[3] ? kv("说明", r[3]) : ""));
    });
    if (D.S[i] === 1 && (i === 0 || D.S[i - 1] === 0))
      out.push(mkRow("lg-s2off", iso, "S2 关闸",
        "距 252 日高点回撤 > 20%（昨收判定）——进攻层停止新开仓", ""));
    if (i > 0 && D.S[i] === 0 && D.S[i - 1] === 1)
      out.push(mkRow("lg-s2on", iso, "S2 开闸", "回撤收窄回阈值内——恢复入场", ""));
    (tradeBy[o] || []).forEach(function (t) { out.push(tradeRow(iso, t)); });
    return out;
  }
  function appendDay(i) {
    var rs = dayRows(i), k;
    for (k = 0; k < rs.length; k++) log.appendChild(rs[k]);
    nLog += rs.length;
    return rs.length;
  }
  function syncLog(i) {
    var added = 0, k;
    if (i >= logPos) {
      for (k = logPos; k <= i; k++) added += appendDay(k);
    } else {
      log.innerHTML = ""; nLog = 0; sumSeen = false;
      for (k = 0; k <= i; k++) added += appendDay(k);
    }
    logPos = i + 1;
    if (logEmpty) logEmpty.hidden = nLog > 0;
    if (logCnt) logCnt.textContent = nLog + " 条";
    if (added) log.scrollTop = log.scrollHeight;
  }
  function X(i) { return m.ml + (D.D[i] - m.d0) / m.ds * m.pw; }
  function Y(i) {
    return m.mt + (1 - (Math.log(D.C[i]) / Math.LN10 - m.ly0) / m.lys) * m.ph;
  }
  function draw(i) {
    idx = i;
    var x = X(i), y = Y(i);
    curLine.setAttribute("x1", x); curLine.setAttribute("x2", x);
    curDot.setAttribute("cx", x); curDot.setAttribute("cy", y);
    curPx.textContent = D.C[i].toFixed(2);
    if (x > m.ml + m.pw * 0.75) {
      curPx.setAttribute("x", x - 8); curPx.setAttribute("text-anchor", "end");
    } else {
      curPx.setAttribute("x", x + 8); curPx.setAttribute("text-anchor", "start");
    }
    range.value = i;
    dateLab.textContent = isoOf(i);
    countLab.textContent = (i + 1) + " / " + n;
    var off = D.S[i] === 1;
    $("fs-s2-pill").className = "pill sm " + (off ? "crit" : "good");
    $("fs-s2-txt").textContent = off ? "关闸" : "开闸";
    $("fs-s2-val").textContent = off ? "停止新开仓" : "允许入场";
    var q = D.Q[i];
    var ev = $("fs-eq-val");
    ev.textContent = fmtPct(q);
    ev.className = "val num " + (q >= 0 ? "good-text" : "crit-text");
    $("fs-n-val").textContent = D.N[i] + " / " + m.nk + " 笔";
    $("fs-n-ref").textContent = "S2 拦截 " + D.K[i] + " / " + m.nb + " 笔";
    syncLog(i);
  }
  function showSum() { if (sumCard) sumCard.hidden = false; }
  function hideSum() { if (sumCard) sumCard.hidden = true; }
  function atEnd() { if (!sumSeen) { sumSeen = true; showSum(); } }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = idx >= n - 1 ? "重新推演" : "开始推演";
  }
  function tick() {
    if (idx >= n - 1) { pause(); atEnd(); return; }
    draw(idx + 1);
    if (idx >= n - 1) { pause(); atEnd(); }
  }
  function play() {
    if (idx >= n - 1) { hideSum(); draw(0); }
    if (timer) return;
    timer = setInterval(tick, BASE_MS / speed);
    playBtn.textContent = "暂停";
  }
  playBtn.addEventListener("click", function () { if (timer) pause(); else play(); });
  stepBtn.addEventListener("click", function () {
    pause();
    if (idx >= n - 1) return;
    draw(idx + 1);
    if (idx >= n - 1) { pause(); atEnd(); }
  });
  sec.querySelectorAll(".fs-speed").forEach(function (b) {
    b.addEventListener("click", function () {
      sec.querySelectorAll(".fs-speed").forEach(function (x2) {
        x2.classList.toggle("act", x2 === b);
      });
      speed = parseFloat(b.getAttribute("data-s")) || 1;
      if (timer) { clearInterval(timer); timer = setInterval(tick, BASE_MS / speed); }
    });
  });
  range.addEventListener("input", function () {
    pause(); hideSum();
    draw(Math.max(0, Math.min(n - 1, parseInt(range.value, 10) || 0)));
  });
  if (sumBtn) sumBtn.addEventListener("click", function () {
    if (sumCard) { if (sumCard.hidden) showSum(); else hideSum(); }
  });
  if (sumClose) sumClose.addEventListener("click", hideSum);
  window.__fs = { draw: draw, play: play, pause: pause, n: n,
                  showSum: showSum, hideSum: hideSum,
                  get idx() { return idx; }, get playing() { return !!timer; },
                  get logCount() { return nLog; } };
  draw(0);
})();
"""


# 情报流层级筛选（P1-1c）。不经 %-格式化，直接拼接。
_FEED_JS = """
(function () {
  var bar = document.getElementById("feed-bar");
  var sec = document.getElementById("feed-sec");
  if (!bar || !sec) return;
  bar.querySelectorAll(".fd-chip").forEach(function (b) {
    b.addEventListener("click", function () {
      bar.querySelectorAll(".fd-chip").forEach(function (x) {
        x.classList.toggle("act", x === b);
      });
      var v = b.getAttribute("data-tf");
      if (v !== "all") {
        sec.querySelectorAll("details.feed-more").forEach(function (d) {
          d.open = true;
        });
      }
      sec.querySelectorAll("li.ev").forEach(function (li) {
        li.hidden = v !== "all" && li.getAttribute("data-tier") !== v;
      });
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
        finra = load_finra_releases()
        body = [_ICON_SPRITE, render_topbar(conn, now, px)]
        symbol_block = render_symbol_view(
            "TSLA", dates, closes,
            load_risk_segments(), load_blind_segments(), load_pits(),
            load_musk_buys(),
            det["trades"] if det else [], has_trades_table, fresh_badge,
        )
        health_rows = load_health(conn, sources)
        extras, nitter_alert = load_health_extras(conn, health_rows, now)
        sections = [
            render_consensus(det, px, shadow, now),
            render_morning_brief(conn, det, px, now,
                                 load_calendar(conn, finra, det, now)),
            render_detector(det, now, render_waitboard(det, px, finra, now),
                            nitter_alert),
            symbol_block,
            render_health(health_rows, now, extras),
            render_timeline(load_timeline(conn), load_hi_timeline(conn), now),
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
        replay_days = load_replay_days()
        replay_html = render_replay_view(
            replay_days, dates, closes, finra, load_trigger_cards(),
            load_daily_opens(), load_app_results(),
        )
        fullsys_html = render_fullsys_view(load_e8a_replay(), replay_days)
        body.append(
            "<main>" + render_symbol_tabs() + render_view_tabs()
            + f'<div class="vw act" id="view-live">{"".join(rendered)}</div>'
            + f'<div class="vw" id="view-replay">{render_replay_modebar()}'
            + f'<div class="rmode act" id="rmode-det">{replay_html}</div>'
            + f'<div class="rmode" id="rmode-full">{fullsys_html}</div></div>'
            + "</main>"
        )
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
        + _RP_JS
        + _FS_JS
        + _FEED_JS
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
