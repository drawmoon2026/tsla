"""N3 因果探测器 — 哨兵常驻值班组件（前向虚拟推演，不碰真钱）.

冻结规则（N3-H 2026-07-24 冻结版，本文件只做前向实现，参数不许改）：
- 触发 = Musk 密集发帖日（act = 次交易日）× 回看 20 交易日内有空头利益
  up-jump 发布（change_pct >= +10%，发布时刻早于密集日结束）
- risk-off 持续 = F20：触发日起 20 个交易日，重叠触发顺延
- 定位 = 窄谱风险过滤器（只防"空头知情型下跌"），不是全天候减仓开关

标定问题（docs/strategy-lab.md N3 条目预先声明）：
前向放风腿来自 x_nitter RSS（只含 post+RT，无 reply；每账号每次仅最近 20 帖），
与历史 Sprinklr 归档（含 reply）口径不同——历史密集参考值 65 帖/日不可直接用。
实现 = 分位数映射标定（这是标定实现，不是改规则）：
- 从 musk_tweets.csv 全档（2018-01-03 → 2025-05-08，2683 个自然日，0 补齐）
  算出 65 帖/日的经验分位数 P(count<=65) = 0.8789，写死为 DENSE_QUANTILE；
- 前 CALIB_BDAYS=20 个交易日为标定期：只累积 nitter 口径的日计数基线
  （逐日存 detector_state.musk_window_json），状态输出 CALIBRATING，不出信号；
- 期满后阈值 = nitter 基线（扩张窗，含标定期后继续累积）的同分位数，每日重算。

已知口径边界（如实声明，不修饰）：
- nitter RT 的时间是原帖时间；实例宕机（整自然日无成功轮询）的日子按 blind 处理：
  计数存 null，不入基线、不参与密集判定，出闸按「有效」基线日数计（P0-1 修复）；
- 交易日口径：2026-08-02 起改用统一 NYSE 日历（intel/market_calendar.py，含假日
  与半日市；此前为 numpy busday 周一至周五近似，彩排实证跨 Labor Day 的 F20 端点
  偏 1 天）。**这是口径修正非规则变更**：F20=「20 个交易日」语义不变，只是交易日
  的定义从 busday 近似修正为真实日历（类比 N6 拆股修正先例）；已落库的历史行
  不追溯改写；
- 首个标定日覆盖的自然日可能因轮询未满一天而低估——都是基线噪声，标定期本身
  就是为吸收这类口径差而设。

每次哨兵轮询后运行（run_sentinel 注册在全部采集器之后）：
- 逐交易日在 detector_state 写/更新一行状态（CALIBRATING / RISK_ON / RISK_OFF）
- 状态切换时向 events 表发一条 source=detector 事件（title 含状态与两腿数值，
  仪表盘情报流自动显示）
- RISK_OFF 触发记一条假想减仓单、解除记恢复单（detector_trades，含 yfinance
  TSLA 价格快照），周报判分用

影子配置 C077（N9 残值落地 2026-08-03，只观察不出信号）：
- 规则 = 空头单腿：up-jump（change_pct >= +10%，同主配置 J10 阈值与 N6 拆股
  防护口径）act 日（发布次交易日）起 risk-off F30（30 个交易日，重叠顺延）；
  无放风腿依赖——不依赖 Musk 数据，失明期照常运转，也无标定期
- 出身与考绩（outputs/n9_frontier 存档）：N9 前沿扫描发现段判对率 71%（5/7，
  现行同频 57%），但考场 3 段判对 67%、避险均值 -1.1%，v2 晋升三条线
  （段数>=4 / 判对>=75% / 避险均值<0）只过一条——发现段优势未出场外复现，
  不晋升
- 处置 = 影子运转攒前向样本：每日在主配置评估之后同步评估（复用同一份
  short_releases 与拆股防护），状态写 detector_shadow_state 独立表；
  **不发事件、不记假想单、不上指挥卡**——纯数据积累。若前向表现追平主配置
  再议升格（须另走 N 系列登记，不得在此暗改）
- 历史状态不回填（前向纪律，2026-08-03 起累积）；历史表现查 outputs/n9_frontier

用法：
    .venv/bin/python -m intel.detector --once     # 单跑一轮
    .venv/bin/python -m intel.detector_report     # 值班报告
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from intel import market_calendar as mcal  # 统一 NYSE 交易日历（口径修正 2026-08-02）
from intel import store
from intel.collectors.base import Collector, cli

ET = ZoneInfo("America/New_York")

# ---- 冻结规则参数（N3-H，不许改） -----------------------------------------
SHORT_JUMP_PCT = 10.0    # up-jump：空头利益双周 change_pct >= +10%
LOOKBACK_BDAYS = 20      # 密集日回看窗（交易日）
PERSIST_BDAYS = 20       # F20：触发日起 20 个交易日 risk-off，重叠触发顺延

# ---- 影子配置 C077（N9 残值，只观察不出信号，非值班规则参数） ---------------
SHADOW_CONFIG_ID = "C077"    # N9 前沿扫描工作点编号（J10 空头单腿 F30）
SHADOW_PERSIST_BDAYS = 30    # F30：act 日起 30 个交易日 risk-off，重叠顺延

# ---- 拆股防护（N6 复核加装，数据伪影防御，非规则参数） ----------------------
# FINRA short_interest 为未复权股数：拆股跨期会产生假 up-jump（历史实例：
# 2020-08-31 期 5:1 拆股 +345.8%、2022-08-31 期 3:1 拆股 +202.2%，见
# outputs/n6_split_audit）。修正后 2018-2026 真实双周变动最大 +34.0%；
# 最小拆股因子 2 在空头仓位不变时产生 ~+100% 跳变。阈值 +50% 居中：
# chg_pct >= SPLIT_GUARD_PCT 的发布不允许自动触发，改发人工复核事件
# （确认为真实跳变后由人工处置；防护只拦"疑似拆股"，不放宽也不收紧冻结规则）。
# 阈值定义在 intel/splits.py（采集器跳变告警与本防护同源引用，单一口径来源）。
from intel.splits import SPLIT_GUARD_PCT  # noqa: E402  同源阈值

# ---- 标定常量（分位映射实现，非规则参数） ----------------------------------
DENSE_REF_COUNT = 65     # 历史 Sprinklr 口径密集参考值（帖/日）
DENSE_QUANTILE = 0.8789  # P(日发帖数<=65)，musk_tweets.csv 2018-01-03→2025-05-08
                         # 全档 2683 自然日（0 补齐）经验 CDF，2026-07-24 算出写死
CALIB_BDAYS = 20         # 标定期：前 20 个交易日只累积基线，不出信号

MUSK_TYPES = ("x_musk_post", "x_musk_rt")
COST_LINE = -0.0006      # 判分成本线：往返 ~6bp（与 N3-H 日记同口径）

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detector_state (
    state_date          TEXT PRIMARY KEY,   -- ET 交易日 YYYY-MM-DD
    state               TEXT NOT NULL CHECK (state IN ('CALIBRATING','RISK_ON','RISK_OFF')),
    musk_count          REAL,               -- 信号窗内最大日发帖数（nitter 口径）
    musk_count_day      TEXT,               -- 该计数对应的自然日
    musk_window_json    TEXT,               -- 本行覆盖的自然日计数 {date: count}，基线即由此累积；
                                            -- count=null 表示该日 x_nitter 无成功采集（blind），
                                            -- 不入基线、不参与密集判定（P0-1）
    dense_thr           REAL,               -- 当日生效密集阈值（标定期 NULL）
    baseline_days       INTEGER NOT NULL,   -- 已累积「有效」基线交易日数（含当日；blind 行不计）
    short_settlement    TEXT,               -- 最新已发布空头结算日
    short_chg_pct       REAL,               -- 最新已发布空头 change_pct
    short_upjump_recent INTEGER NOT NULL DEFAULT 0,  -- 回看20交易日内有无 up-jump 发布
    triggered           INTEGER NOT NULL DEFAULT 0,  -- 本日是否命中触发（含顺延触发）
    risk_off_until      TEXT,               -- RISK_OFF 到期交易日（含）
    updated_utc         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detector_shadow_state (
    config_id        TEXT NOT NULL,      -- 影子配置编号（现仅 'C077'）
    state_date       TEXT NOT NULL,      -- ET 交易日 YYYY-MM-DD
    state            TEXT NOT NULL CHECK (state IN ('RISK_ON','RISK_OFF')),
    short_settlement TEXT,               -- 最新已发布空头结算日
    short_chg_pct    REAL,               -- 最新已发布空头 change_pct
    triggered        INTEGER NOT NULL DEFAULT 0,  -- act 日=今日的新 up-jump 触发
    risk_off_until   TEXT,               -- F30 到期交易日（含）；RISK_ON 为 NULL
    n_upjumps_active INTEGER NOT NULL DEFAULT 0,  -- F30 窗仍生效的 up-jump 份数
    updated_utc      TEXT NOT NULL,
    PRIMARY KEY (config_id, state_date)  -- 独立表：主配置 detector_state 零改动
);

CREATE TABLE IF NOT EXISTS detector_trades (
    trade_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_time_utc TEXT NOT NULL,
    state_date     TEXT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('REDUCE','RESTORE')),
    price          REAL,               -- yfinance TSLA 快照；获取失败 NULL 如实记
    price_time_utc TEXT,
    note           TEXT
);
"""


# ------------------------------------------------------------------ helpers


def prev_bday(d: date) -> date:
    # 原 busday 近似：np.busday_offset(d, -1, roll="forward")（不剔假日）
    return mcal.prev_trading_day(d)


def next_bday_after(d: date) -> date:
    """严格晚于 d 的第一个交易日（d 可为周末/假日）——act 日."""
    # 原 busday 近似：np.busday_offset(d, 1, roll="backward")（不剔假日）
    return mcal.next_trading_day(d)


def _et_date(iso_utc: str) -> date:
    return datetime.fromisoformat(iso_utc).astimezone(ET).date()


def musk_day_counts(conn, lo: date, hi: date) -> dict[str, int]:
    """[lo, hi] 闭区间自然日的 Musk 日发帖数（nitter 口径 post+RT，0 补齐）."""
    counts = {str(lo + timedelta(days=k)): 0 for k in range((hi - lo).days + 1)}
    q = ",".join("?" * len(MUSK_TYPES))
    for (t,) in conn.execute(
        f"SELECT event_time_utc FROM events WHERE source_id='x_nitter' AND type IN ({q})",
        MUSK_TYPES,
    ):
        d = str(_et_date(t))
        if d in counts:
            counts[d] += 1
    return counts


def nitter_covered_days(conn, lo: date, hi: date) -> set[str]:
    """[lo, hi] 内有 x_nitter 成功轮询（poll_log ok=1，按 ET 自然日归属）的日期集合.

    P0-1 blind 口径依据：某自然日整日无成功采集，则该日的事件计数不可信
    （断供期一律计 0，会污染基线并在断供窗内漏触发）——该日标 blind。
    """
    if hi < lo:
        return set()
    # 粗筛（UTC 字符串区间左右各放宽一天），逐条按 ET 日精确归属
    lo_s = (datetime.combine(lo, datetime.min.time(), tzinfo=ET)
            - timedelta(days=1)).astimezone(timezone.utc).isoformat(timespec="seconds")
    hi_s = (datetime.combine(hi, datetime.max.time(), tzinfo=ET)
            + timedelta(days=1)).astimezone(timezone.utc).isoformat(timespec="seconds")
    covered: set[str] = set()
    for (t,) in conn.execute(
        """SELECT poll_time_utc FROM poll_log
           WHERE source_id='x_nitter' AND ok=1
             AND poll_time_utc >= ? AND poll_time_utc <= ?""",
        (lo_s, hi_s),
    ):
        d = str(_et_date(t))
        if str(lo) <= d <= str(hi):
            covered.add(d)
    return covered


def effective_baseline_days(conn, before_date: str) -> tuple[int, int]:
    """(有效基线交易日数, 跳过的 blind 自然日数)，只数 before_date 之前的行.

    有效行 = musk_window_json 至少含一个非 null 计数的行（blind 日计数为 null）；
    blind 自然日数 = 各行窗口里 null 计数日去重后的总数。
    """
    n_eff = 0
    blind_days: set[str] = set()
    for (j,) in conn.execute(
        "SELECT musk_window_json FROM detector_state WHERE state_date < ?",
        (before_date,),
    ):
        try:
            win = json.loads(j) if j else {}
        except ValueError:
            win = {}
        if any(v is not None for v in win.values()):
            n_eff += 1
        blind_days.update(k for k, v in win.items() if v is None)
    return n_eff, len(blind_days)


def short_releases(conn, now_iso: str) -> list[dict]:
    """已发布（event_time <= now）的空头利益发布，按发布时刻升序."""
    out = []
    for r in conn.execute(
        """SELECT event_time_utc, payload_json FROM events
           WHERE source_id='finra_short' AND event_time_utc <= ?
           ORDER BY event_time_utc""",
        (now_iso,),
    ):
        p = json.loads(r["payload_json"] or "{}")
        chg = p.get("change_pct")
        if chg is None and p.get("prev_short_interest"):
            chg = (p["short_interest"] / p["prev_short_interest"] - 1) * 100
        if chg is None:
            continue
        out.append({
            "pub_time_utc": r["event_time_utc"],
            "settlement": p.get("settlement_date"),
            "chg_pct": float(chg),
            "short_interest": p.get("short_interest"),
            "days_to_cover": p.get("days_to_cover"),
        })
    return out


def recent_upjumps(shorts: list[dict], today: date) -> tuple[list[dict], list[dict]]:
    """act（发布次交易日）落在回看 20 交易日内的 up-jump 发布.

    返回 (可信 up-jump, 疑似拆股伪影)：chg_pct >= SPLIT_GUARD_PCT 的发布
    进第二个列表，不参与自动触发（N6 拆股防护，见文件头常量注释）。
    """
    out: list[dict] = []
    suspects: list[dict] = []
    for s in shorts:
        if s["chg_pct"] < SHORT_JUMP_PCT:
            continue
        act = next_bday_after(_et_date(s["pub_time_utc"]))
        # 原 busday 近似：int(np.busday_count(act, today))（不剔假日）
        if act <= today and mcal.trading_days_between(act, today) <= LOOKBACK_BDAYS:
            (suspects if s["chg_pct"] >= SPLIT_GUARD_PCT else out).append(s)
    return out, suspects


def baseline_from_rows(conn, before_date: str | None = None) -> dict[str, int]:
    """把历史 detector_state 行累积的日计数合并成基线分布（date -> count）."""
    sql = "SELECT musk_window_json FROM detector_state"
    args: tuple = ()
    if before_date:
        sql += " WHERE state_date < ?"
        args = (before_date,)
    base: dict[str, int] = {}
    for (j,) in conn.execute(sql, args):
        if j:
            # blind 日（无成功采集，计数存 null）不入基线（P0-1）
            base.update({k: v for k, v in json.loads(j).items() if v is not None})
    return base


def tsla_snapshot() -> tuple[float | None, str | None, str | None]:
    """(price, time_utc, error) —— yfinance 当前价快照，失败如实返回 None."""
    try:
        import yfinance as yf

        tk = yf.Ticker("TSLA")
        price = None
        try:
            price = float(tk.fast_info["last_price"])
        except Exception:  # noqa: BLE001
            h = tk.history(period="1d")
            if len(h):
                price = float(h["Close"].iloc[-1])
        if price and price > 0:
            return price, store.utcnow_iso(), None
        return None, None, "no price data"
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ 影子配置


def evaluate_shadow(conn, today: date, now_iso: str, shorts: list[dict]) -> dict:
    """C077 影子评估：空头单腿 up-jump >= +10% × F30，只落库不出信号.

    与主配置复用同一份 short_releases 结果与 N6 拆股防护口径（chg_pct >=
    SPLIT_GUARD_PCT 的疑似伪影不触发影子）。无放风腿、无标定期，状态只有
    RISK_ON / RISK_OFF；每个可信 up-jump 的 act 日（发布次交易日）起 F30，
    重叠取最远到期（与主配置顺延语义一致）。状态为当日从发布史整体重算的
    纯函数——只写 detector_shadow_state 当日行（同日重跑覆盖），不回填历史，
    不发事件、不记假想单、不上指挥卡。
    """
    until_best: str | None = None
    triggered = False
    n_active = 0
    for s in shorts:
        if not (SHORT_JUMP_PCT <= s["chg_pct"] < SPLIT_GUARD_PCT):
            continue  # 未达 J10 或疑似拆股伪影（N6 防护同口径）
        act = next_bday_after(_et_date(s["pub_time_utc"]))
        if act > today:
            continue  # act 未到，尚不生效
        until = str(mcal.add_trading_days(act, SHADOW_PERSIST_BDAYS - 1))
        if until >= str(today):
            n_active += 1
        if act == today:
            triggered = True
        until_best = max(until_best or "", until)
    state = "RISK_OFF" if (until_best and until_best >= str(today)) else "RISK_ON"
    latest = shorts[-1] if shorts else None
    conn.execute(
        """INSERT OR REPLACE INTO detector_shadow_state
           (config_id, state_date, state, short_settlement, short_chg_pct,
            triggered, risk_off_until, n_upjumps_active, updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (SHADOW_CONFIG_ID, str(today), state,
         latest["settlement"] if latest else None,
         latest["chg_pct"] if latest else None,
         int(triggered),
         until_best if state == "RISK_OFF" else None,
         n_active, now_iso),
    )
    conn.commit()
    return {"state": state,
            "until": until_best if state == "RISK_OFF" else None}


# ------------------------------------------------------------------ 状态机


def evaluate(conn, now: datetime | None = None) -> dict:
    """跑一轮状态机：写 detector_state 当日行，切换时发事件/记假想单.

    返回 {state, switched, n_new_events, note}。非交易日不评估。
    """
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="seconds")
    today = now.astimezone(ET).date()
    # 原 busday 近似：np.is_busday(today)（假日照常评估并写行——彩排 Labor Day 实证）
    if not mcal.is_trading_day(today):
        return {"state": None, "switched": False, "n_new_events": 0,
                "note": f"{today} 非交易日（NYSE 日历），跳过"}

    conn.executescript(_SCHEMA)

    # -- 前情：今日已有行则续写（盘中多次轮询），否则接最近一行
    row_today = conn.execute(
        "SELECT * FROM detector_state WHERE state_date=?", (str(today),)
    ).fetchone()
    row_prev = conn.execute(
        "SELECT * FROM detector_state WHERE state_date<? ORDER BY state_date DESC LIMIT 1",
        (str(today),),
    ).fetchone()
    ref = row_today or row_prev
    prev_state = ref["state"] if ref else None
    prev_until = ref["risk_off_until"] if ref else None
    # P0-1：出闸按「有效」基线交易日数计——blind 行（窗口全 null）不计入
    n_eff_before, _n_blind_before = effective_baseline_days(conn, str(today))
    active = n_eff_before >= CALIB_BDAYS       # 第 21 个有效交易日起出信号

    # -- 放风腿：信号窗 = 上一交易日（含）至昨日（含）的自然日，act 均为今日
    win_lo, win_hi = prev_bday(today), today - timedelta(days=1)
    window: dict[str, int | None] = (
        musk_day_counts(conn, win_lo, win_hi) if win_hi >= win_lo else {})
    # P0-1 blind 判定：当日无成功 x_nitter 轮询 → 该自然日计数标 null（blind）：
    # 不入基线、不参与密集判定（断供期计 0 既污染基线又假装「无密集」）
    if window:
        covered = nitter_covered_days(conn, win_lo, win_hi)
        window = {d: (c if d in covered else None) for d, c in window.items()}
    window_valid = {d: c for d, c in window.items() if c is not None}
    leg_b_blind = bool(window) and not window_valid
    baseline_days = n_eff_before + (1 if window_valid else 0)   # 含今日（今日有效才计）
    musk_day, musk_cnt = (max(window_valid.items(), key=lambda kv: kv[1])
                          if window_valid else (None, None))

    # -- 空头腿
    shorts = short_releases(conn, now_iso)
    latest = shorts[-1] if shorts else None
    upjumps, split_suspects = recent_upjumps(shorts, today)

    # -- 拆股防护（N6）：疑似拆股跳变不自动触发，发一次性人工复核事件
    n_guard = 0
    for s in split_suspects:
        n_guard += store.insert_events(conn, "detector", [{
            "dedupe_key": f"split_guard_{s['settlement']}",
            "event_time_utc": now_iso,
            "symbol": "TSLA",
            "type": "detector_split_guard",
            "title": (f"拆股防护拦截：空头 change_pct {s['chg_pct']:+.1f}% >= "
                      f"+{SPLIT_GUARD_PCT:.0f}%（结算 {s['settlement']}）疑似拆股伪影，"
                      "已从自动触发中排除——需人工复核（确认真实跳变后人工处置）"),
            "payload": {"settlement": s["settlement"], "chg_pct": s["chg_pct"],
                        "short_interest": s["short_interest"],
                        "guard_pct": SPLIT_GUARD_PCT,
                        "rule": "N6 split guard: excluded from auto-trigger"},
        }])

    # -- 状态决策
    thr = None
    triggered = False
    until = prev_until
    if not active:
        state = "CALIBRATING"
        until = None
    else:
        base = baseline_from_rows(conn, before_date=str(today))  # 不含今日窗（因果）
        vals = np.array(list(base.values()), float)
        thr = float(np.quantile(vals, DENSE_QUANTILE)) if len(vals) else None
        if thr is not None:
            for d, c in window_valid.items():   # blind 日不参与密集判定（P0-1）
                if c <= thr:
                    continue
                dense_end = datetime.combine(
                    date.fromisoformat(d), datetime.max.time().replace(microsecond=0)
                ).replace(tzinfo=ET)
                if any(datetime.fromisoformat(u["pub_time_utc"]) < dense_end
                       for u in upjumps):
                    triggered = True
                    musk_day, musk_cnt = d, c   # 展示触发那一天的读数
                    break
        if triggered:
            # F20 端点：触发日起第 20 个交易日（含触发日）。原 busday 近似：
            # np.busday_offset(today, PERSIST_BDAYS - 1)——跨假日会早 1 天结束
            new_until = str(mcal.add_trading_days(today, PERSIST_BDAYS - 1))
            until = max(until or "", new_until)  # 重叠触发顺延
            state = "RISK_OFF"
        elif until and until >= str(today):
            state = "RISK_OFF"                   # F20 延续
        else:
            state, until = "RISK_ON", None

    # -- 写当日状态行（同日重跑覆盖更新）
    conn.execute(
        """INSERT OR REPLACE INTO detector_state
           (state_date, state, musk_count, musk_count_day, musk_window_json,
            dense_thr, baseline_days, short_settlement, short_chg_pct,
            short_upjump_recent, triggered, risk_off_until, updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (str(today), state, musk_cnt, musk_day, json.dumps(window),
         thr, baseline_days, latest["settlement"] if latest else None,
         latest["chg_pct"] if latest else None, int(bool(upjumps)),
         int(triggered), until, now_iso),
    )
    conn.commit()

    # -- 状态切换：发 detector 事件 + 假想单
    switched = prev_state != state
    n_new = 0
    if switched:
        musk_s = (f"Musk {musk_day} 发帖 {musk_cnt}（nitter 口径）" if musk_cnt is not None
                  else "Musk 窗口 blind（x_nitter 无成功采集）" if leg_b_blind
                  else "Musk 无窗口数据")
        if thr is not None:
            musk_s += f"，阈值 {thr:.1f}"
        elif state == "CALIBRATING":
            musk_s += f"，基线标定 {baseline_days}/{CALIB_BDAYS} 交易日"
        short_s = (f"空头 {latest['chg_pct']:+.1f}%（结算 {latest['settlement']}）"
                   if latest else "空头无已发布数据")
        if state == "RISK_OFF":
            title = (f"探测器 RISK_OFF 触发：{musk_s} × {short_s}，"
                     f"回看{LOOKBACK_BDAYS}日内 up-jump {len(upjumps)} 份 → F20 至 {until}")
        elif prev_state == "RISK_OFF":
            title = f"探测器 RISK_OFF 解除 → RISK_ON：F20 到期；{musk_s}；{short_s}"
        elif state == "RISK_ON" and prev_state == "CALIBRATING":
            title = f"探测器标定完成 → RISK_ON：{musk_s}；{short_s}"
        else:
            title = f"探测器上线 CALIBRATING：{musk_s}；{short_s}"
        n_new = store.insert_events(conn, "detector", [{
            "dedupe_key": f"{today}_{prev_state}->{state}",
            "event_time_utc": now_iso,
            "symbol": "TSLA",
            "type": "detector_state",
            "title": title,
            "payload": {
                "state": state, "prev_state": prev_state,
                "musk_count": musk_cnt, "musk_count_day": musk_day,
                "dense_thr": thr, "baseline_days": baseline_days,
                "short_chg_pct": latest["chg_pct"] if latest else None,
                "short_settlement": latest["settlement"] if latest else None,
                "n_upjumps_recent": len(upjumps), "risk_off_until": until,
                "rule": f"N3-H frozen: upjump>=+{SHORT_JUMP_PCT:.0f}% x dense, F{PERSIST_BDAYS}",
            },
        }])
        action = ("REDUCE" if state == "RISK_OFF"
                  else "RESTORE" if prev_state == "RISK_OFF" else None)
        if action:
            price, ptime, err = tsla_snapshot()
            # 仓位口径注明（P1-6）：REDUCE = 假想全仓→现金（N3-H 应用 A 判分口径），
            # 不是部分减仓；RESTORE = 全仓买回。判分（B&H vs 成本线）即按此口径。
            note = ("假想减仓（口径：全仓→现金，应用 A）：" if action == "REDUCE"
                    else "假想恢复（全仓买回）：") + title
            if err:
                note += f"（价格快照失败：{err}）"
            conn.execute(
                """INSERT INTO detector_trades
                   (trade_time_utc, state_date, action, price, price_time_utc, note)
                   VALUES (?,?,?,?,?,?)""",
                (now_iso, str(today), action, price, ptime, note),
            )
            conn.commit()

    # -- 影子配置 C077（N9 残值）：主配置评估完毕后同步评估，纯观察不出信号；
    #    失败只入 note，不影响值班主路径
    try:
        sh = evaluate_shadow(conn, today, now_iso, shorts)
        shadow_note = (f" shadow[{SHADOW_CONFIG_ID}]={sh['state']}"
                       + (f"→{sh['until']}" if sh["until"] else "") + "(观察)")
    except Exception as e:  # noqa: BLE001
        shadow_note = f" shadow[{SHADOW_CONFIG_ID}]=ERROR({type(e).__name__}: {e})"

    return {"state": state, "switched": switched, "n_new_events": n_new + n_guard,
            "note": f"{state} baseline={baseline_days}/{CALIB_BDAYS}(有效)"
                    + (f" thr={thr:.1f}" if thr is not None else "")
                    + (f" until={until}" if state == "RISK_OFF" else "")
                    + (" 腿B失明(blind：窗口内无成功采集，不入基线不判定)"
                       if leg_b_blind else "")
                    + (f" split_guard={n_guard}" if n_guard else "")
                    + shadow_note}


# ------------------------------------------------------------------ 组件封装


class CausalDetector(Collector):
    """哨兵值班组件：非采集渠道，每轮在全部采集器之后运行一次状态机."""

    SOURCE = {
        "source_id": "detector",
        "name": "N3 因果探测器（空头up-jump×Musk密集，F20）",
        "tier": "T1",
        "method": "derived",
        "poll_interval_s": 0,   # 不节流：每轮哨兵都评估
        "cost": "free",
        "weight_source": 0.5,
        "notes": "衍生信号非采集渠道；冻结规则 N3-H；nitter 口径标定期 20 交易日，"
                 "期满前状态 CALIBRATING",
    }

    def run_once(self, verbose: bool = True) -> dict:
        sid = self.SOURCE["source_id"]
        conn = store.connect()
        store.upsert_source(conn, self.SOURCE)
        t0 = time.time()
        try:
            res = evaluate(conn)
            dur = int((time.time() - t0) * 1000)
            store.record_poll(conn, sid, ok=True, n_seen=1,
                              n_new=res["n_new_events"], duration_ms=dur)
            stats = {"source_id": sid, "ok": True, "n_seen": 1,
                     "n_new": res["n_new_events"], "duration_ms": dur, "error": None}
            if verbose:
                print(f"[{sid}] ok  {res['note']}"
                      + ("  ->state event" if res["n_new_events"] else "")
                      + f" ({dur}ms)")
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
        return stats


if __name__ == "__main__":
    cli(CausalDetector)
