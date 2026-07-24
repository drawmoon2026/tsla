"""因果探测器 · 哨兵 — 静态 HTML 仪表盘生成器.

读 data/intel/sentinel.sqlite（只读，mode=ro），渲染完全自包含的
data/intel/dashboard.html：内联全部 CSS/JS，无外部依赖，浏览器直接打开。

板块（按重要性）：
  1. 顶栏：生成时刻 / 最后事件入库时刻 / 最后轮询时刻（>30 分钟标红）
  2. 因果探测器（detector_state 当前状态 + 两腿读数 + 标定进度/阈值
     + 最近状态切换 + 假想单判分；表缺失整面板隐藏）
  3. 渠道健康表（poll_log + sources + events 累计；含权重列，tier→权重排序）
  4. 最新情报流（最近 50 条事件时间线）
  5. 时延统计（observed - event 的 p50/p90；回填口径加注）
  6. 今日/近 7 日事件计数按层级条形图（内联 SVG）

容错：任一表/视图缺失则跳过对应板块，不炸。

用法：
    .venv/bin/python -m intel.dashboard              # 生成一次
    .venv/bin/python -m intel.dashboard --out x.html # 自定义输出路径

launchd 定时重生成模板见 intel/deploy/com.tsla.dashboard.plist（默认不 load）。
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sqlite3
from datetime import datetime, timedelta, timezone
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

TIERS = ["T0", "T1", "T2", "T3"]

TIER_LABEL = {
    "T0": "T0 布局痕迹",
    "T1": "T1 监管申报",
    "T2": "T2 官方发布",
    "T3": "T3 公开舆情",
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


def render_health(rows: list[dict], now: datetime) -> str:
    if not rows:
        return ""
    trs = []
    for h in rows:
        last = h["last"]
        if last is None:
            status_cls, status_txt, row_cls = "off", "未启用", "row-off"
            last_t = None
            detail = ""
        else:
            last_t = parse_ts(last["poll_time_utc"])
            age = (now - last_t).total_seconds() if last_t else None
            interval = h["poll_interval_s"] or 300
            if last["ok"] == 0:
                status_cls, status_txt, row_cls = "crit", "失败", "row-crit"
                detail = esc((last.get("error") or "")[:120])
            elif age is not None and age > max(2 * interval, STALE_S):
                status_cls, status_txt, row_cls = "warn", "迟滞", ""
                detail = f"距上次轮询 {esc(fmt_dur(age))}"
            else:
                status_cls, status_txt, row_cls = "good", "正常", ""
                detail = f"抓 {last['n_seen']} / 新 {last['n_new']}"
        streak = h["fail_streak"]
        streak_html = (
            f'<span class="crit-text num">{streak}</span>' if streak else '<span class="num">0</span>'
        )
        w = h.get("weight")
        if w is None:
            weight_html = "—"
        else:
            t = h["tier"] if h["tier"] in TIERS else "T3"
            weight_html = (
                '<span class="wt-cell">'
                f'<span class="wt-bar"><span class="wtf-{t[1]}" '
                f'style="width:{min(1.0, max(0.0, w)) * 100:.0f}%"></span></span>'
                f"{w:.1f}</span>"
            )
        trs.append(
            f'<tr class="{row_cls}">'
            f"<td>{tier_badge(h['tier'])}</td>"
            f'<td class="src"><strong>{esc(h["source_id"])}</strong>'
            f'<span class="sub">{esc(h["name"])}</span></td>'
            f'<td class="num wt">{weight_html}</td>'
            f'<td><span class="pill sm {status_cls}"><span class="dot"></span>{status_txt}</span></td>'
            f'<td class="num">{fmt_local(last_t)}<span class="sub rel" data-iso="'
            f'{esc(last_t.isoformat() if last_t else "")}">{esc(fmt_ago(last_t, now))}</span></td>'
            f"<td class=\"num\">{streak_html}</td>"
            f'<td class="num">{h["n_events"]:,}</td>'
            f'<td class="detail">{detail}</td></tr>'
        )
    return f"""
<section>
  <h2>渠道健康 <span class="h-sub">按层级 → 权重排序</span></h2>
  <div class="card">
    <div class="scroll-x">
      <table>
        <thead><tr><th>层级</th><th>渠道</th><th>权重</th><th>状态</th><th>最近轮询</th>
        <th>连续失败</th><th>累计事件</th><th>备注</th></tr></thead>
        <tbody>{"".join(trs)}</tbody>
      </table>
    </div>
  </div>
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


def render(db_path: Path = DB_PATH) -> str:
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sources = load_sources(conn)
        body = [render_topbar(conn, now)]
        sections = [
            render_detector(load_detector(conn), now),
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
