"""因果探测器 · 哨兵 — 静态 HTML 仪表盘生成器.

读 data/intel/sentinel.sqlite（只读，mode=ro），渲染完全自包含的
data/intel/dashboard.html：内联全部 CSS/JS，无外部依赖，浏览器直接打开。

板块（按信息层级）：
  1. 顶栏：生成时刻 / 最后事件入库时刻 / 最后轮询时刻（>30 分钟标红）
  2. 因果探测器（detector_state 当前状态 + 两腿读数 + 标定进度/阈值
     + 最近状态切换 + 假想单判分；表缺失整面板隐藏）
  3. 走势与历史判断（标的视图，render_symbol_view 可复用：TSLA 日线收盘
     对数坐标折线 + 1/3/8 年时间刷 + 四层可开关标注——避险底纹 / 真假坑 /
     Musk 买入 / 假想单；事后研究口径已注明。页面预留多标的标签栏。）
  4. 渠道矩阵（按 T0-T3 分组的渠道卡片，每层配 SVG 小图标）
  5. 最新情报流（最近 50 条事件时间线，行首带层级小图标）+ 时延统计
  6. 今日/近 7 日事件计数按层级条形图（内联 SVG）

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

from intel.store import DB_PATH

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
DET_STATE = {
    "CALIBRATING": ("warn", "标定中", "累积 nitter 口径基线，不出信号"),
    "RISK_ON": ("good", "正常持仓", "两腿未同时命中，无风险规避"),
    "RISK_OFF": ("crit", "假想减仓", "空头 up-jump × Musk 密集命中"),
}


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
    t = tier if tier in TIERS else "T3"
    return f'<span class="tier tier-{t[1]}">{esc(tier or "T?")}</span>'


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
    """基于 v_latency 的渠道集合，p50/p90 在 Python 里补算。"""
    if not (has_table(conn, "v_latency") and has_table(conn, "events")):
        return []
    out = []
    for r in conn.execute("SELECT * FROM v_latency ORDER BY source_id"):
        lags = sorted(
            x[0]
            for x in conn.execute(
                """SELECT (julianday(observed_time_utc) - julianday(event_time_utc))
                          * 86400.0
                   FROM events WHERE source_id = ?""",
                (r["source_id"],),
            )
        )
        row = dict(r)
        row["p50"] = percentile(lags, 0.50)
        row["p90"] = percentile(lags, 0.90)
        out.append(row)
    return out


def load_tier_counts(conn: sqlite3.Connection, now: datetime) -> dict[str, dict]:
    """今日（UTC 日）与近 7 日按层级的事件计数（observed 口径）。"""
    if not (has_table(conn, "events") and has_table(conn, "sources")):
        return {}
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
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


def _render_view_svg(
    vid: str,
    dates: list[date],
    closes: list[float],
    i0: int,
    risk: list[tuple[date, date]],
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

    # -- 避险底纹（最底层）
    band_rects = []
    for a, b in risk:
        ao, bo = max(a.toordinal(), d0o), min(b.toordinal() + 1, d1o)
        if bo <= d0o or ao >= d1o:
            continue
        band_rects.append(
            f'<rect x="{x(ao):.1f}" y="{_MT}" width="{x(bo) - x(ao):.1f}" '
            f'height="{ph}"><title>避险区段（RISK_OFF）{a} → {b}</title></rect>'
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

    # -- 价格折线
    pts = " ".join(
        f"{x(d.toordinal()):.1f},{y(c):.1f}" for d, c in zip(sub_d, sub_c)
    )
    parts.append(f'<polyline class="tv-price" points="{pts}"/>')

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
                f"坑底收盘 {p['close']:,.2f}",
                f"深度 {_fmt_pct_pt(p['dd'])}",
                f"后 60 日最高 {_fmt_pct_pt(p['fwd'])}",
                f"空头 6 周变化 {_fmt_pct_pt(p['si6'])}",
                f"Musk 趋势比 {p['mtr']:.2f}" if p["mtr"] is not None else "Musk 趋势比 —",
            ]
        )
        if p["golden"]:
            shape = (
                f'<polygon class="mk mk-gold" points="{mx:.1f},{my - 7:.1f} '
                f'{mx - 6.5:.1f},{my + 5:.1f} {mx + 6.5:.1f},{my + 5:.1f}"/>'
            )
        else:
            shape = (
                f'<polygon class="mk mk-trap" points="{mx:.1f},{my + 7:.1f} '
                f'{mx - 6.5:.1f},{my - 5:.1f} {mx + 6.5:.1f},{my - 5:.1f}"/>'
            )
        hit = (
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my:.1f}" r="13" '
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
            f'<circle class="mk mk-musk" cx="{mx:.1f}" cy="{my:.1f}" r="5.5"/>'
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my:.1f}" r="13" '
            f'data-tip="{esc(tip)}"><title>{esc(tip.replace("|", chr(10)))}</title>'
            "</circle>"
        )
    marker_group("ly-musk", m_items)

    d_items = []
    for t in trades:
        try:
            td = date.fromisoformat(str(t.get("state_date")))
        except (TypeError, ValueError):
            continue
        px = t.get("price")
        if px is None or not (d0 <= td <= d1):
            continue
        mx, my = x(td.toordinal()), y(float(px))
        reduce_ = t.get("action") == "REDUCE"
        tip = _tip_attr(
            [
                f"假想单 {t.get('action')} · {td}",
                f"TSLA 快照 {float(px):,.2f}",
                (t.get("note") or "")[:60] or "—",
            ]
        )
        d_items.append(
            f'<rect class="mk {"mk-trd-r" if reduce_ else "mk-trd-g"}" '
            f'x="{mx - 5:.1f}" y="{my - 5:.1f}" width="10" height="10" '
            f'transform="rotate(45 {mx:.1f} {my:.1f})"/>'
            f'<circle class="tv-hit" cx="{mx:.1f}" cy="{my:.1f}" r="13" '
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


def _layer_btn(
    layer: str, sw: str, label: str, count_txt: str,
    disabled_note: str | None = None, title: str = "",
) -> str:
    if disabled_note:
        return (
            f'<button class="tv-lb" data-layer="{layer}" disabled '
            f'title="{esc(disabled_note)}">{sw}{esc(label)} '
            f'<span class="cnt">{esc(disabled_note)}</span></button>'
        )
    return (
        f'<button class="tv-lb" data-layer="{layer}" aria-pressed="true" '
        f'title="{esc(title)}">{sw}{esc(label)} '
        f'<span class="cnt">{esc(count_txt)}</span></button>'
    )


def _pits_table(pits: list[dict], buys_aligned: list[tuple[dict, str]]) -> str:
    """标注明细表——tooltip 之外的无悬停可达路径（可折叠）。"""
    rows = []
    for p in sorted(pits, key=lambda q: q["date"]):
        rows.append(
            "<tr>"
            f"<td>{'真坑 ▲' if p['golden'] else '假坑 ▽'}</td>"
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
        f"<td>Musk 买入 ●</td>"
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
    pits: list[dict] | None,
    buys: list[dict] | None,
    trades: list[dict],
    has_trades_table: bool,
) -> str:
    """③ 走势与历史判断——单标的完整区块（图 + 时间刷 + 图层开关 + 明细表）。"""
    if not dates:
        return f"""
<section>
  <h2>走势与历史判断 <span class="h-sub">{esc(symbol)} 日线收盘 · 对数刻度 · 事后标注</span></h2>
  <div class="card"><p class="empty">价格数据不可读（{esc(str(BARS_CSV))}）——走势区块降级为空。</p></div>
</section>"""

    risk_l = risk or []
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
            vid, dates, closes, i0, risk_l, pits_l, buys_l, trades, active=(vid == "3y")
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
    layer_btns = "".join(
        [
            _layer_btn(
                "risk", '<span class="sw-band"></span>', "避险底纹",
                f"{len(risk_l)} 段",
                None if risk is not None else "数据缺失", risk_title,
            ),
            _layer_btn(
                "pitg", '<span class="sw-tri up"></span>', "真坑",
                f"{n_gold}", None if pits is not None else "数据缺失",
                "label=golden（事后口径）",
            ),
            _layer_btn(
                "pitt", '<span class="sw-tri dn"></span>', "假坑",
                f"{n_trap}", None if pits is not None else "数据缺失",
                "label=trap（事后口径）",
            ),
            _layer_btn(
                "musk", '<span class="sw-dot"></span>', "Musk 买入",
                f"{len(buys_aligned)}", None if buys is not None else "数据缺失",
                musk_note,
            ),
            _layer_btn(
                "trade", '<span class="sw-dia"></span>', "假想单",
                f"{len(trades)}", trade_note, "探测器值班期的虚拟操作",
            ),
        ]
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
  <h2>走势与历史判断 <span class="h-sub">{esc(symbol)} 日线收盘（ET 交易日聚合） · 对数刻度 · {esc(str(dates[0]))} → {esc(str(last))}</span></h2>
  <div class="card" id="tv">
    <div class="tv-bar">
      <div class="tv-ranges" role="group" aria-label="时间范围">{range_btns}</div>
      <div class="tv-layers" role="group" aria-label="图层开关">{layer_btns}</div>
    </div>
    <div class="tv-wrap" id="tv-wrap">
      {"".join(svgs)}
      <div id="tv-tip" hidden></div>
    </div>
    {_pits_table(pits_l, buys_aligned)}
    <p class="footnote">真假坑与避险区段为事后研究口径，非当时可知信号；坑的标注含前视指标
    （后 60 日表现），仅作历史复盘。Musk 买入为 Form 4 申报事实（蓝点落在首个成交日的收盘价上）。
    {esc(foot_extra)}价格轴为对数刻度；悬停标记看明细，悬停曲线看逐日收盘。</p>
  </div>
  <script type="application/json" id="tv-data">{data_json}</script>
</section>"""


# ---------------------------------------------------------------- rendering

def render_topbar(conn: sqlite3.Connection, now: datetime) -> str:
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
            f'<div class="stat {extra_cls}">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value num" data-iso="{esc(iso)}"'
            + (f' id="{iso_id}"' if iso_id else "")
            + f">{fmt_local(dt)}</div>"
            f'<div class="stat-sub rel">{esc(fmt_ago(dt, now))}</div></div>'
        )

    return f"""
<header>
  <div class="brand">
    <h1>因果探测器 · 哨兵</h1>
    <span class="pill {pill_cls}" id="poll-pill"><span class="dot"></span>{pill_txt}</span>
  </div>
  <div class="stats">
    {stat("页面生成于", now, iso_id="gen-ts")}
    {stat("最后事件入库", last_obs)}
    {stat("最后轮询", last_poll, "poll-stat" + (" crit-text" if stale else ""), "poll-ts")}
  </div>
  <div class="banner" id="stale-banner" hidden>
    此页面生成已久，数据可能过期——请重新运行 <code>python -m intel.dashboard</code>
    或加载 launchd 定时任务。
  </div>
</header>"""


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


def render_detector(data: dict | None, now: datetime) -> str:
    if not data:
        return ""
    cur = data["cur"]
    state = cur["state"]
    cls, phrase, why = DET_STATE.get(state, ("off", state, ""))
    upd = parse_ts(cur.get("updated_utc"))

    # -- 状态徽章
    badge_sub = f"{esc(phrase)} · {esc(why)}"
    if state == "RISK_OFF" and cur.get("risk_off_until"):
        badge_sub += f" · F{PERSIST_BDAYS} 至 {esc(cur['risk_off_until'])}"
    badge = (
        f'<div class="det-state {cls}">'
        f'<div class="det-state-name">{esc(state)}</div>'
        f'<div class="det-state-sub">{badge_sub}</div>'
        f'<div class="det-state-meta num">状态日 {esc(cur["state_date"])}'
        f'<span class="sub rel" data-iso="{esc(upd.isoformat() if upd else "")}">'
        f"更新于 {esc(fmt_ago(upd, now))}</span></div></div>"
    )

    # -- 两腿读数
    musk_sub = "口径日 " + esc(cur.get("musk_count_day") or "—") + " · nitter 口径"
    leg_musk = (
        '<div class="det-leg"><div class="leg-label">放风腿 · Musk 日发帖</div>'
        f'<div class="leg-value num">{_fmt_count(cur.get("musk_count"))}'
        '<span class="leg-unit">帖/日</span></div>'
        f'<div class="leg-sub">{musk_sub}</div></div>'
    )
    chg = cur.get("short_chg_pct")
    upjump = bool(cur.get("short_upjump_recent"))
    jump_html = (
        f'<span class="warn-text">回看 {LOOKBACK_BDAYS} 交易日内有 up-jump</span>'
        if upjump
        else f"回看 {LOOKBACK_BDAYS} 交易日内无 up-jump"
    )
    leg_short = (
        '<div class="det-leg"><div class="leg-label">空头腿 · 最新期变化</div>'
        f'<div class="leg-value num">{f"{chg:+.2f}" if chg is not None else "—"}'
        '<span class="leg-unit">%</span></div>'
        f'<div class="leg-sub">结算 {esc(cur.get("short_settlement") or "—")} · '
        f"{jump_html}</div></div>"
    )

    # -- 第四格：标定进度条（标定期）或密集阈值（正式期）
    if state == "CALIBRATING":
        days = int(cur.get("baseline_days") or 0)
        pct = min(100.0, days / CALIB_BDAYS * 100)
        leg4 = (
            '<div class="det-leg"><div class="leg-label">标定进度 · 基线累积</div>'
            f'<div class="leg-value num">{days}<span class="leg-unit">/ {CALIB_BDAYS} '
            "交易日</span></div>"
            f'<div class="prog" role="img" aria-label="标定进度 {days}/{CALIB_BDAYS}">'
            f'<span style="width:{pct:.0f}%"></span></div>'
            '<div class="leg-sub">期满后阈值 = 基线同分位数，出信号</div></div>'
        )
    else:
        thr = cur.get("dense_thr")
        leg4 = (
            '<div class="det-leg"><div class="leg-label">密集阈值 · 当日生效</div>'
            f'<div class="leg-value num">{_fmt_count(thr)}'
            '<span class="leg-unit">帖/日</span></div>'
            '<div class="leg-sub">nitter 基线分位映射 · 扩张窗每日重算</div></div>'
        )

    # -- 最近状态切换
    switches_html = ""
    if data["switches"]:
        items = []
        for s in data["switches"]:
            t = parse_ts(s["event_time_utc"])
            s_cls = DET_STATE.get(s.get("state") or "", ("off",))[0]
            items.append(
                '<li class="ev">'
                f'<span class="ev-time num">{fmt_local(t)}</span>'
                f'<span class="pill sm {s_cls}"><span class="dot"></span>'
                f'{esc(s.get("state") or "?")}</span>'
                f'<span class="ev-title" title="{esc(s["title"])}">{esc(s["title"])}</span>'
                "</li>"
            )
        switches_html = (
            f'<div class="det-sub"><h3>状态切换 <span class="h-sub">最近 '
            f'{len(items)} 次</span></h3>'
            f'<ol class="feed">{"".join(items)}</ol></div>'
        )

    footnote = (
        f"冻结规则 N3-H：Musk 密集发帖（act=次交易日）× 回看 {LOOKBACK_BDAYS} 交易日内"
        f"空头 change_pct ≥ +{SHORT_JUMP_PCT:.0f}% 发布 → RISK_OFF 持续 "
        f"F{PERSIST_BDAYS}（{PERSIST_BDAYS} 交易日，重叠触发顺延）。"
        f"标定期 {CALIB_BDAYS} 交易日只累积基线，不出信号。假想推演，不碰真钱。"
    )
    return f"""
<section>
  <h2>因果探测器 <span class="h-sub">N3-H 冻结规则 · 空头 up-jump × Musk 密集 · 前向虚拟推演</span></h2>
  <div class="card">
    <div class="det-grid">
      {badge}
      {leg_musk}
      {leg_short}
      {leg4}
    </div>
    {switches_html}
    {_det_trades_html(data["trades"], now)}
    <p class="footnote">{footnote}</p>
  </div>
</section>"""


TIER_ICON = {  # 层级 → sprite symbol id（雷达/盾牌/天平/扩音器）
    "T0": "ic-t0", "T1": "ic-t1", "T2": "ic-t2", "T3": "ic-t3",
}


def tier_icon(tier: str | None, cls: str = "t-ic") -> str:
    sid = TIER_ICON.get(tier or "", "ic-t3")
    return (
        f'<svg class="{cls}" aria-hidden="true" focusable="false">'
        f'<use href="#{sid}"/></svg>'
    )


def _src_card(h: dict, now: datetime) -> str:
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
    if w is None:
        weight_html = ""
    else:
        t = h["tier"] if h["tier"] in TIERS else "T3"
        weight_html = (
            '<span class="wt-cell" title="权重">'
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


def render_health(rows: list[dict], now: datetime) -> str:
    """④ 渠道矩阵：按 T0-T3 分组的卡片组，每层配 SVG 小图标。"""
    if not rows:
        return ""
    groups: dict[str, list[dict]] = {}
    for h in rows:  # rows 已按 tier→权重排好序
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
    return f"""
<section>
  <h2>渠道矩阵 <span class="h-sub">按层级分组 · 组内按权重排序</span></h2>
  <div class="card">{"".join(blocks)}</div>
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
        items.append(
            '<li class="ev">'
            f"{tier_icon(e.get('tier'), 'ev-ic')}"
            f'<span class="ev-time num" title="观察时刻（本机时区）">{fmt_local(obs)}</span>'
            f"{tier_badge(e.get('tier'))}"
            f'<span class="ev-src">{esc(e["source_id"])}</span>'
            f'<span class="ev-title">{title_html}</span>'
            f'<span class="ev-type sub">{esc(e.get("type") or "")}</span>'
            "</li>"
        )
    return f"""
<section>
  <h2>最新情报流 <span class="h-sub">最近 {len(rows)} 条 · 按观察时刻倒序</span></h2>
  <div class="card">
    <ol class="feed">{"".join(items)}</ol>
  </div>
</section>"""


def render_latency(rows: list[dict], sources: dict) -> str:
    if not rows:
        return ""
    trs = []
    for r in rows:
        sid = r["source_id"]
        src = sources.get(sid, {})
        notes = []
        if (r.get("lag_min_s") or 0) < 0:
            notes.append("日历预告（lag 为负 = 事件在未来，正常）")
        interval = src.get("poll_interval_s") or 0
        if r["p50"] is not None and interval and r["p50"] > 12 * max(interval, 300):
            notes.append("回填主导，偏大")
        trs.append(
            "<tr>"
            f"<td>{tier_badge(src.get('tier'))}</td>"
            f"<td><strong>{esc(sid)}</strong></td>"
            f'<td class="num">{r["n_events"]:,}</td>'
            f'<td class="num">{esc(fmt_dur(r["p50"])) if r["p50"] is not None else "—"}</td>'
            f'<td class="num">{esc(fmt_dur(r["p90"])) if r["p90"] is not None else "—"}</td>'
            f'<td class="num">{esc(fmt_dur(r["lag_min_s"])) if r.get("lag_min_s") is not None else "—"}</td>'
            f'<td class="detail">{esc("；".join(notes))}</td></tr>'
        )
    return f"""
<section>
  <h2>渠道时延 <span class="h-sub">observed − event</span></h2>
  <div class="card">
    <div class="scroll-x">
      <table>
        <thead><tr><th>层级</th><th>渠道</th><th>n</th><th>中位</th><th>p90</th>
        <th>最小</th><th>备注</th></tr></thead>
        <tbody>{"".join(trs)}</tbody>
      </table>
    </div>
    <p class="footnote">口径注：首采期的 p50/p90 被回填支配（老事件今天才开始观察），
    显著高于稳态时延；稳态 ≈ 轮询间隔 + 源侧发布延迟，随增量事件积累自动收敛。
    「最小」列更接近该渠道的稳态下限。</p>
  </div>
</section>"""


def _bar_panel(title: str, tiers: list[str], counts: dict, key: str, maxc: int) -> str:
    """内联 SVG 手绘水平条形图（右端 4px 圆角、基线端方角）。"""
    row_h, bar_h, x0, vw = 30, 18, 52, 340
    max_w = vw - x0 - 44
    vh = len(tiers) * row_h + 6
    parts = [
        f'<svg viewBox="0 0 {vw} {vh}" role="img" aria-label="{esc(title)}按层级事件计数">'
    ]
    for i, t in enumerate(tiers):
        n = counts.get(t, {}).get(key, 0)
        y = i * row_h + 4
        cy = y + bar_h / 2
        parts.append(
            f'<text x="{x0 - 10}" y="{cy}" class="svg-label" text-anchor="end" '
            f'dominant-baseline="central">{esc(t)}</text>'
        )
        if n > 0 and maxc > 0:
            w = max(4.0, n / maxc * max_w)
            r = min(4.0, w)
            d = (
                f"M{x0},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{bar_h - 2 * r} "
                f"a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"
            )
            parts.append(f'<path d="{d}" class="bar-{t[1]}"/>')
            lx = x0 + w + 8
        else:
            lx = x0 + 8
        parts.append(
            f'<text x="{lx:.1f}" y="{cy}" class="svg-value" '
            f'dominant-baseline="central">{n:,}</text>'
        )
    parts.append("</svg>")
    return (
        f'<div class="chart"><h3>{esc(title)}</h3>{"".join(parts)}</div>'
    )


def render_tier_chart(counts: dict[str, dict]) -> str:
    if not counts:
        return ""
    tiers = [t for t in TIERS if t in counts]
    if not tiers:
        return ""
    maxc = max(
        [c["today"] for c in counts.values()] + [c["week"] for c in counts.values()]
    )
    legend = " · ".join(
        f'<span class="lg"><span class="sw sw-{t[1]}"></span>{esc(TIER_LABEL.get(t, t))}</span>'
        for t in tiers
    )
    return f"""
<section>
  <h2>事件计数 <span class="h-sub">按层级 · observed 口径 · 两图同标尺</span></h2>
  <div class="card">
    <div class="charts">
      {_bar_panel("今日（UTC 日）", tiers, counts, "today", maxc)}
      {_bar_panel("近 7 日", tiers, counts, "week", maxc)}
    </div>
    <p class="footnote legend-line">{legend}</p>
  </div>
</section>"""


# ---------------------------------------------------------------- page shell

# 层级小图标 sprite：T0 雷达（布局痕迹）/ T1 盾牌（法定披露）/
# T2 天平（官方承诺）/ T3 扩音器（放风叙事）。线条继承 CSS stroke。
_ICON_SPRITE = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="ic-t0" viewBox="0 0 24 24">
  <circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>
  <path d="M12 12 L18.4 5.7"/>
  <circle cx="15.8" cy="14.6" r="1.4" fill="currentColor" stroke="none"/>
</symbol>
<symbol id="ic-t1" viewBox="0 0 24 24">
  <path d="M12 3 L19.5 6 V11.5 C19.5 16.5 16.4 20 12 21.5 C7.6 20 4.5 16.5 4.5 11.5 V6 Z"/>
</symbol>
<symbol id="ic-t2" viewBox="0 0 24 24">
  <path d="M12 4.5 V19"/><path d="M5 7 H19"/><path d="M8.5 21 H15.5"/>
  <path d="M5 7 L2.5 12.5 M5 7 L7.5 12.5 M2 12.5 a3 3 0 0 0 6 0"/>
  <path d="M19 7 L16.5 12.5 M19 7 L21.5 12.5 M16 12.5 a3 3 0 0 0 6 0"/>
</symbol>
<symbol id="ic-t3" viewBox="0 0 24 24">
  <path d="M3.5 10 V14 H7 L16.5 19.5 V4.5 L7 10 Z"/>
  <path d="M19.5 9.5 a4.5 4.5 0 0 1 0 5"/>
</symbol>
</defs></svg>"""

_CSS = """
:root {
  color-scheme: light;
  --bg: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --border: rgba(11,11,11,0.10); --grid: #e1e0d9;
  --good: #0ca30c; --good-text: #006300;
  --warn: #fab219; --warn-text: #7a5200;
  --crit: #d03b3b; --crit-text: #b02a2a;
  --good-wash: rgba(12,163,12,0.08);
  --warn-wash: rgba(250,178,25,0.12);
  --crit-wash: rgba(208,59,59,0.07); --off: #898781;
  --tier0: #1c5cab; --tier0-ink: #ffffff;
  --tier1: #2a78d6; --tier1-ink: #ffffff;
  --tier2: #5598e7; --tier2-ink: #0b0b0b;
  --tier3: #86b6ef; --tier3-ink: #0b0b0b;
  --link: #1c5cab;
  /* 走势图（金/蓝已过配色六检；灰为刻意中性槽位，身份由形状▽承担） */
  --chart-line: #2f2e2b; --band: rgba(208,59,59,0.12);
  --mk-gold: #b07d00; --mk-blue: #1c5cab; --mk-trap: #898781;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --border: rgba(255,255,255,0.10); --grid: #2c2c2a;
    --good: #0ca30c; --good-text: #0ca30c;
    --warn: #fab219; --warn-text: #fab219;
    --crit: #d03b3b; --crit-text: #e66767;
    --good-wash: rgba(12,163,12,0.16);
    --warn-wash: rgba(250,178,25,0.14);
    --crit-wash: rgba(208,59,59,0.14);
    --tier0: #9ec5f4; --tier0-ink: #0b0b0b;
    --tier1: #5598e7; --tier1-ink: #0b0b0b;
    --tier2: #256abf; --tier2-ink: #ffffff;
    --tier3: #184f95; --tier3-ink: #ffffff;
    --link: #86b6ef;
    --chart-line: #e4e3da; --band: rgba(208,59,59,0.22);
    --mk-gold: #bd8600; --mk-blue: #4f94e4; --mk-trap: #898781;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
main, header { max-width: 1080px; margin: 0 auto; padding: 0 20px; }
header { padding-top: 22px; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
.num, td.num, .stat-value { font-variant-numeric: tabular-nums; }

.brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
h1 { font-size: 19px; margin: 0; letter-spacing: 0.02em; }
h2 { font-size: 15px; margin: 26px 0 10px; }
h3 { font-size: 13px; margin: 0 0 6px; color: var(--ink-2); font-weight: 600; }
.h-sub { font-size: 12px; font-weight: 400; color: var(--muted); margin-left: 8px; }

.stats { display: flex; gap: 28px; flex-wrap: wrap; margin-top: 14px; }
.stat-label { font-size: 12px; color: var(--muted); }
.stat-value { font-size: 17px; font-weight: 600; }
.stat-sub { font-size: 12px; color: var(--muted); }
.crit-text, .crit-text .stat-value, .stat.crit-text .stat-value { color: var(--crit-text); }
.stat.crit-text .stat-label { color: var(--crit-text); }
.good-text { color: var(--good-text); }
.warn-text { color: var(--warn-text); }

.banner {
  margin-top: 14px; padding: 8px 12px; border: 1px solid var(--crit);
  border-radius: 6px; background: var(--crit-wash); color: var(--crit-text);
  font-size: 13px;
}

.pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; font-weight: 600; padding: 2px 10px;
  border: 1px solid var(--border); border-radius: 999px; color: var(--ink-2);
  white-space: nowrap;
}
.pill.sm { padding: 1px 8px; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.pill.good .dot { background: var(--good); }
.pill.good { color: var(--good-text); }
.pill.warn .dot { background: var(--warn); }
.pill.warn { color: var(--warn-text); }
.pill.crit .dot { background: var(--crit); }
.pill.crit { color: var(--crit-text); }
.pill.off .dot { background: var(--off); }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 0;
}
.scroll-x { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 720px; }
th, td { text-align: left; padding: 7px 14px; white-space: nowrap; }
thead th {
  font-size: 12px; font-weight: 600; color: var(--muted);
  border-bottom: 1px solid var(--grid);
}
tbody tr { border-bottom: 1px solid var(--grid); }
tbody tr:last-child { border-bottom: none; }
tr.row-crit { background: var(--crit-wash); }
tr.row-off { color: var(--muted); }
td .sub, .src .sub { display: block; font-size: 11.5px; color: var(--muted); white-space: nowrap; }
td.detail { font-size: 12px; color: var(--ink-2); max-width: 340px;
            overflow: hidden; text-overflow: ellipsis; }

.tier {
  display: inline-block; min-width: 30px; text-align: center;
  font-size: 11.5px; font-weight: 700; padding: 1px 7px; border-radius: 5px;
  font-variant-numeric: tabular-nums;
}
.tier-0 { background: var(--tier0); color: var(--tier0-ink); }
.tier-1 { background: var(--tier1); color: var(--tier1-ink); }
.tier-2 { background: var(--tier2); color: var(--tier2-ink); }
.tier-3 { background: var(--tier3); color: var(--tier3-ink); }

/* ---- 探测器面板（状态用语义色：good/warn/crit，不用层级色） ---- */
.det-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 14px; padding: 14px;
}
.det-state {
  border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px;
  display: flex; flex-direction: column; justify-content: center; gap: 3px;
}
.det-state.good { background: var(--good-wash); border-color: var(--good); }
.det-state.good .det-state-name { color: var(--good-text); }
.det-state.warn { background: var(--warn-wash); border-color: var(--warn); }
.det-state.warn .det-state-name { color: var(--warn-text); }
.det-state.crit { background: var(--crit-wash); border-color: var(--crit); }
.det-state.crit .det-state-name { color: var(--crit-text); }
.det-state-name { font-size: 27px; font-weight: 800; letter-spacing: 0.04em;
                  line-height: 1.15; }
.det-state-sub { font-size: 12px; color: var(--ink-2); }
.det-state-meta { font-size: 12px; color: var(--muted); }
.det-state-meta .sub { display: inline; margin-left: 8px; }
.det-leg { padding: 12px 2px; }
.leg-label { font-size: 12px; color: var(--muted); }
.leg-value { font-size: 24px; font-weight: 700; margin: 2px 0; }
.leg-unit { font-size: 12px; font-weight: 400; color: var(--muted); margin-left: 5px; }
.leg-sub { font-size: 12px; color: var(--muted); }
.prog {
  height: 8px; border-radius: 4px; background: var(--grid);
  overflow: hidden; margin: 7px 0 5px; max-width: 220px;
}
.prog span { display: block; height: 100%; border-radius: 4px;
             background: var(--warn); min-width: 4px; }
.det-sub { border-top: 1px solid var(--grid); }
.det-sub h3 { margin: 10px 14px 4px; }
.det-sub .feed { padding-bottom: 6px; }
.det-table { min-width: 640px; }

/* ---- 渠道权重（数值条按层级色阶着色） ---- */
td.wt { min-width: 96px; }
.wt-cell { display: inline-flex; align-items: center; gap: 8px; }
.wt-bar {
  width: 52px; height: 6px; border-radius: 3px; background: var(--grid);
  overflow: hidden; flex: none;
}
.wt-bar span { display: block; height: 100%; border-radius: 3px; }
.wtf-0 { background: var(--tier0); } .wtf-1 { background: var(--tier1); }
.wtf-2 { background: var(--tier2); } .wtf-3 { background: var(--tier3); }
tr.row-off .wt-bar span { opacity: 0.45; }

.feed { list-style: none; margin: 0; padding: 2px 0; }
.ev {
  display: flex; align-items: baseline; gap: 10px;
  padding: 6px 14px; border-bottom: 1px solid var(--grid);
}
.ev:last-child { border-bottom: none; }
.ev-time { flex: none; width: 84px; color: var(--ink-2); font-size: 12.5px; }
.ev .tier { flex: none; }
.ev-src { flex: none; width: 74px; font-size: 12px; color: var(--muted); }
.ev-title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ev-type { flex: none; font-size: 11.5px; color: var(--muted); }

.charts {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 18px; padding: 12px 14px 4px;
}
.chart svg { width: 100%; height: auto; display: block; }
.svg-label { font-size: 12px; font-weight: 700; fill: var(--ink-2);
             font-variant-numeric: tabular-nums; }
.svg-value { font-size: 12px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
.bar-0 { fill: var(--tier0); } .bar-1 { fill: var(--tier1); }
.bar-2 { fill: var(--tier2); } .bar-3 { fill: var(--tier3); }
.legend-line { display: flex; gap: 16px; flex-wrap: wrap; }
.lg { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.sw-0 { background: var(--tier0); } .sw-1 { background: var(--tier1); }
.sw-2 { background: var(--tier2); } .sw-3 { background: var(--tier3); }

/* ---- 标的标签栏（多标的预留位） ---- */
.sym-tabs {
  display: flex; align-items: center; gap: 6px;
  margin-top: 26px; margin-bottom: -10px; padding-bottom: 2px;
}
.sym-tab {
  font-size: 13px; font-weight: 700; letter-spacing: 0.03em;
  padding: 5px 18px; border: 1px solid var(--border); border-radius: 7px;
  color: var(--muted); background: transparent;
}
.sym-tab.act { background: var(--tier0); color: var(--tier0-ink);
               border-color: var(--tier0); }
.sym-hint { font-size: 11.5px; color: var(--muted); margin-left: 6px; }

/* ---- 走势与历史判断 ---- */
.tv-bar {
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; flex-wrap: wrap; padding: 12px 14px 6px;
}
.tv-ranges {
  display: inline-flex; border: 1px solid var(--border);
  border-radius: 6px; overflow: hidden; flex: none;
}
.tv-rb {
  appearance: none; border: none; background: transparent; cursor: pointer;
  color: var(--ink-2); font: inherit; font-size: 12.5px; font-weight: 600;
  padding: 4px 14px; border-right: 1px solid var(--border);
}
.tv-rb:last-child { border-right: none; }
.tv-rb.act { background: var(--tier0); color: var(--tier0-ink); }
.tv-layers { display: flex; gap: 8px; flex-wrap: wrap; }
.tv-lb {
  display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
  appearance: none; font: inherit; font-size: 12px; font-weight: 600;
  color: var(--ink-2); background: transparent;
  border: 1px solid var(--border); border-radius: 999px; padding: 3px 11px;
}
.tv-lb .cnt { color: var(--muted); font-weight: 400;
              font-variant-numeric: tabular-nums; }
.tv-lb.off { opacity: 0.4; }
.tv-lb[disabled] { opacity: 0.4; cursor: not-allowed; }
.sw-band { width: 10px; height: 10px; border-radius: 2px; flex: none;
           background: var(--band); border: 1px solid var(--crit); }
.sw-tri { width: 0; height: 0; flex: none;
          border-left: 5px solid transparent; border-right: 5px solid transparent; }
.sw-tri.up { border-bottom: 9px solid var(--mk-gold); }
.sw-tri.dn { border-top: 9px solid var(--mk-trap); }
.sw-dot { width: 9px; height: 9px; border-radius: 50%; flex: none;
          background: var(--mk-blue); }
.sw-dia { width: 8px; height: 8px; flex: none; background: var(--crit);
          transform: rotate(45deg); }
.tv-wrap { position: relative; padding: 2px 8px 0; }
.tv-view { display: none; }
.tv-view.act { display: block; }
.tv-svg { width: 100%; height: auto; display: block; }
.tv-grid line { stroke: var(--grid); stroke-width: 1; }
.tv-tick { font-size: 11px; fill: var(--muted);
           font-variant-numeric: tabular-nums; }
.tv-price { fill: none; stroke: var(--chart-line); stroke-width: 2;
            stroke-linejoin: round; stroke-linecap: round; }
.ly-risk rect { fill: var(--band); }
.mk { stroke: var(--surface); stroke-width: 2; paint-order: stroke; }
.mk-gold { fill: var(--mk-gold); }
.mk-trap { fill: var(--mk-trap); }
.mk-musk { fill: var(--mk-blue); }
.mk-trd-r { fill: var(--crit); }
.mk-trd-g { fill: var(--good); }
.tv-hit { fill: transparent; cursor: pointer; }
.tv-ov { fill: transparent; }
.xh line { stroke: var(--muted); stroke-width: 1; }
.xh circle { fill: var(--chart-line); stroke: var(--surface); stroke-width: 2; }
#tv.hide-risk .ly-risk, #tv.hide-pitg .ly-pitg, #tv.hide-pitt .ly-pitt,
#tv.hide-musk .ly-musk, #tv.hide-trade .ly-trade { display: none; }
#tv-tip {
  position: absolute; z-index: 5; pointer-events: none;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; font-size: 12px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.18); max-width: 280px;
}
#tv-tip div { color: var(--ink-2); white-space: nowrap;
              font-variant-numeric: tabular-nums; }
#tv-tip .tt-head { font-weight: 700; color: var(--ink); }
.tv-table { margin: 4px 14px 8px; font-size: 12.5px; }
.tv-table summary { cursor: pointer; color: var(--muted); font-size: 12px;
                    padding: 4px 0; }
.tv-table table { min-width: 640px; }

/* ---- 层级小图标 + 渠道矩阵 ---- */
.t-ic, .ev-ic {
  fill: none; stroke: currentColor; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round; color: var(--ink-2);
}
.t-ic { width: 18px; height: 18px; flex: none; }
.ev-ic { width: 14px; height: 14px; flex: none; align-self: center;
         stroke-width: 2.2; color: var(--muted); }
.tier-group { border-bottom: 1px solid var(--grid); padding: 10px 14px 14px; }
.tier-group:last-child { border-bottom: none; }
.tier-head { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
.tier-head h3 { margin: 0; }
.src-cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
  gap: 10px;
}
.src-card {
  border: 1px solid var(--grid); border-radius: 8px; padding: 9px 12px;
  display: flex; flex-direction: column; gap: 3px; min-width: 0;
}
.src-card.crit { background: var(--crit-wash); border-color: var(--crit); }
.src-card.offline { color: var(--muted); }
.sc-top { display: flex; justify-content: space-between; align-items: center;
          gap: 8px; }
.sc-name { font-size: 11.5px; color: var(--muted); overflow: hidden;
           text-overflow: ellipsis; white-space: nowrap; }
.sc-met { display: flex; gap: 12px; align-items: center; font-size: 12px;
          color: var(--ink-2); flex-wrap: wrap; }
.sc-sub { font-size: 11.5px; color: var(--muted); }
.sc-sub .rel { margin-left: 6px; }

.footnote { font-size: 12px; color: var(--muted); margin: 8px 14px 10px; }
.empty { color: var(--muted); padding: 30px 14px; }
footer { max-width: 1080px; margin: 24px auto 40px; padding: 0 20px;
         font-size: 12px; color: var(--muted); }
@media (max-width: 640px) {
  .ev-title { white-space: normal; }
  .ev { flex-wrap: wrap; }
}
"""

_JS = """
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
      var stat = pt.closest(".stat");
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
        body = [_ICON_SPRITE, render_topbar(conn, now)]
        dates, closes = load_daily_closes()
        symbol_block = render_symbol_tabs() + render_symbol_view(
            "TSLA", dates, closes,
            load_risk_segments(), load_pits(), load_musk_buys(),
            det["trades"] if det else [], has_trades_table,
        )
        sections = [
            render_detector(det, now),
            symbol_block,
            render_health(load_health(conn, sources), now),
            render_timeline(load_timeline(conn), now),
            render_latency(load_latency(conn), sources),
            render_tier_chart(load_tier_counts(conn, now)),
        ]
        rendered = [s for s in sections if s]
        if not rendered:
            rendered = ['<p class="empty">数据库暂无可展示的表——待哨兵首采后刷新。</p>']
        body.append("<main>" + "".join(rendered) + "</main>")
    finally:
        conn.close()
    body.append(
        f"<footer>数据源：{esc(str(db_path))} · 只读渲染 · "
        f"重生成：<code>python -m intel.dashboard</code></footer>"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
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
