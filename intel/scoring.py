"""四维评分 v1 —— 意外分接入期权 IV 基线（其余仍为纯规则，无 LLM）.

框架依据 docs/intel-framework.md 第二节：每条情报打四个分（0-1），乘积为权重。
v1 实现状态（如实标注，不装）：

  身位分  已实现  渠道 tier 映射（POSITION_BY_TIER；衍生信号单列）
  事实分  已实现  事件 type 规则映射（FACT_BY_TYPE + 前缀规则，可查表）
  时效分  已实现  半衰期指数衰减（HALF_LIFE_DAYS 注册表，按渠道类型标定）
  意外分  v1      期权 IV 基线：事件日 TSLA ATM IV 相对前 5 个有效快照日均值
                  的相对变化，线性归一化到 0-1（SURPRISE_REL_FULL 处封顶）。
                  数据源 options_chain 表（2026-07-24 起前向积累）；
                  Yahoo IV 有抽风日（整链近零，实测 07-24 / 07-27）——按
                  质量门槛（IV_VALID_RANGE + 最少报价数）过滤，坏日按缺数据。
                  事件日无有效快照覆盖 → 诚实返回 None，不乘入。

  总权重 = 四项齐时 身位 × 事实 × 时效 × 意外（partial=False）；
           意外分缺席时 = 身位 × 事实 × 时效（partial=True，与 v0 同）。

设计约束：
- 只读计算，不回写 events 表（事件表保持采集原貌，评分在查询层实时算）；
- 输入是 events 行 dict（source_id / type / event_time_utc，tier 可选——
  缺 tier 时按 SOURCE_CLASS/保守默认处理）；
- 意外分需要 DB（options_chain）：调用方先 load_atm_iv_series(conn) 取
  日级 IV 序列，再传入 score_event(..., iv_series=...)；不传 = v0 行为；
- 全部映射表与阈值放在模块顶层，可直接查阅与人工复核。

自测： .venv/bin/python -m intel.scoring   （单调性断言 + 样例分数表）
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

_LN2 = math.log(2.0)

# ---------------------------------------------------------------- 身位分
# 框架第二节：真金白银行为 > 有约束力承诺 > 放风叙事。
# 按渠道 tier 映射；衍生信号（探测器结论，非情报渠道）单列 0.5。
DERIVED_SOURCE_IDS = {"detector"}

POSITION_BY_TIER = {
    "T0": 1.0,  # 布局痕迹（13D/G、Form144、空头利益、暗池、期权快照…）
    "T1": 1.0,  # 法定披露（Form4 / 8-K）
    "T2": 0.7,  # 官方承诺（FOMC / 财报日历 / 专利）
    "T3": 0.3,  # 放风叙事（新闻 / X / YouTube）
}
POSITION_DERIVED = 0.5   # 衍生信号：自家算法结论，非高位者行为
POSITION_DEFAULT = 0.3   # 未知层级 → 按最弱处理（保守）

# ---------------------------------------------------------------- 事实分
# 框架第二节：已发生的事实 1.0 > 可验证的承诺 0.6 > 观点/预期 0.2。
FACT_DISCLOSURE = 1.0   # 法定披露 / 数据发布（已发生、可证伪）
FACT_CALENDAR = 0.6     # 日历预告（可验证的承诺，事件尚未发生）
FACT_NARRATIVE = 0.2    # 新闻报道 / 发帖（观点、转述、叙事）

FACT_BY_TYPE = {  # 精确匹配表（事件 type → 事实分）
    # 法定披露（SEC/FINRA 强制申报，事实）
    "form4": FACT_DISCLOSURE, "form144": FACT_DISCLOSURE,
    "sc13d": FACT_DISCLOSURE, "sc13g": FACT_DISCLOSURE,
    "sc13d_amend": FACT_DISCLOSURE, "sc13g_amend": FACT_DISCLOSURE,
    "short_interest": FACT_DISCLOSURE, "ats_weekly": FACT_DISCLOSURE,
    # 数据发布（快照/赔率是当刻可复核的数值事实）
    "options_snapshot": FACT_DISCLOSURE, "polymarket_odds": FACT_DISCLOSURE,
    # 衍生信号：算法产出的状态读数，按数据发布计
    "detector_state": FACT_DISCLOSURE, "detector_split_guard": FACT_DISCLOSURE,
    # 日历预告（官方排期，事件在未来）
    "fomc_meeting": FACT_CALENDAR, "fomc_meeting_sep": FACT_CALENDAR,
    "earnings_date": FACT_CALENDAR,
    # 发帖（Musk/官号 X 帖：放风，需最强怀疑折扣）
    "x_musk_post": FACT_NARRATIVE, "x_musk_rt": FACT_NARRATIVE,
    "x_tesla_co_post": FACT_NARRATIVE, "x_tesla_co_rt": FACT_NARRATIVE,
}

_FACT_BY_PREFIX = (  # 前缀规则（精确表未命中时按序匹配）
    ("8k", FACT_DISCLOSURE),        # 8-K 各 items 变体
    ("uspto", FACT_DISCLOSURE),     # 专利授权/申请公示（官方公报）
    ("patent", FACT_DISCLOSURE),
    ("news_", FACT_NARRATIVE),
    ("youtube", FACT_NARRATIVE),
    ("x_", FACT_NARRATIVE),
)
FACT_DEFAULT = FACT_NARRATIVE  # 未知类型 → 按观点处理（保守）

# ---------------------------------------------------------------- 时效分
# 半衰期指数衰减：score = exp(-ln2 · age / half_life)，age=半衰期时恰为 0.5。
# 半衰期按渠道类型标定（注册表）；日历类不衰减（预告的价值在事件临近而非发布新旧）。
HALF_LIFE_DAYS = {
    "data": 14.0,      # 数据类（申报/快照/衍生信号）：双周节奏，衰减慢
    "news": 2.0,       # 新闻类（报道/发帖/视频）：叙事热度两天减半
    "calendar": None,  # 日历类：不衰减（恒 1.0）
}

SOURCE_CLASS = {  # 渠道 → 类型（时效半衰期选择用）
    "edgar": "data", "edgar_t0": "data",
    "finra_short": "data", "finra_ats": "data",
    "options_snapshot": "data", "polymarket": "data",
    "detector": "data", "uspto": "data",
    "fed_fomc": "calendar", "earnings_cal": "calendar",
    "news_rss": "news", "youtube": "news",
    "x_nitter": "news", "x_api_paid": "news",
}

_CLASS_BY_TYPE_PREFIX = (  # 渠道未注册时按事件 type 前缀兜底
    ("news_", "news"), ("youtube", "news"), ("x_", "news"),
    ("fomc", "calendar"), ("earnings", "calendar"),
)
CLASS_DEFAULT = "news"  # 未知渠道 → 按衰减最快处理（保守）

# ---------------------------------------------------------------- 意外分 v1
# 共识基线 = 期权隐含波动：事件日 ATM IV 相对前 5 个有效快照日均值的相对变化，
# 线性归一化（|ΔIV|/基线 达 SURPRISE_REL_FULL 记满分 1.0）。IV 动得越大 = 市场
# 越"没想到"。日级口径：同一 ET 日的所有事件共享当日意外分（v1 已知粗糙点）。
#
# 质量门槛（Yahoo IV 有抽风日，07-24/07-27 实测整链近零——同 OI suspect_zero）：
#   目标到期：DTE ∈ [15, 60] 中最接近 30 的到期日（缺则退而取最近的 DTE≥7）；
#   ATM 带：行权价距现价（ITM/OTM 边界估计）±2%，call+put 合并；
#   有效报价：IV ∈ IV_VALID_RANGE 才计入；有效数 < ATM_MIN_QUOTES → 该日无效；
#   基线：前 5 个有效快照日的均值，有效日 < BASELINE_MIN_DAYS → 当日算不出。
SURPRISE_REL_FULL = 0.30      # ATM IV 相对变化 ≥30% → 意外分 1.0
IV_VALID_RANGE = (0.10, 3.0)  # 年化 10%-300% 之外视为坏点（TSLA ATM 应在 40-90% 量级）
ATM_BAND_PCT = 0.02           # ATM 带宽 ±2%
ATM_MIN_QUOTES = 3            # 当日 ATM 有效报价最少条数
DTE_LO, DTE_HI, DTE_TARGET = 15, 60, 30
BASELINE_N = 5                # 基线回看的快照日数
BASELINE_MIN_DAYS = 2         # 基线最少有效日数


def load_atm_iv_series(conn) -> dict[str, float]:
    """options_chain → {ET 快照日: 当日 ATM IV}（只含过质量门槛的日）.

    只读查询；表缺失/无数据返回空 dict（调用方按意外分缺席处理）。
    """
    try:
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='options_chain'"
        ).fetchone()
        if not has:
            return {}
        days = [r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM options_chain ORDER BY 1")]
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for sd in days:
        iv = _atm_iv_one_day(conn, sd)
        if iv is not None:
            out[sd] = iv
    return out


def _atm_iv_one_day(conn, snap_date: str) -> float | None:
    """单快照日 ATM IV（质量门槛内的均值）；算不出/不过门槛 → None."""
    try:
        d0 = date.fromisoformat(snap_date)
        # 现价估计：ITM call 最高行权价与 OTM call 最低行权价的中点
        itm, otm = conn.execute(
            """SELECT (SELECT MAX(strike) FROM options_chain
                       WHERE snapshot_date=? AND opt_right='C' AND in_the_money=1),
                      (SELECT MIN(strike) FROM options_chain
                       WHERE snapshot_date=? AND opt_right='C' AND in_the_money=0)""",
            (snap_date, snap_date),
        ).fetchone()
        if not itm or not otm:
            return None
        spot = (itm + otm) / 2.0
        exps = [r[0] for r in conn.execute(
            "SELECT DISTINCT expiration FROM options_chain WHERE snapshot_date=?",
            (snap_date,))]
        dte = [(e, (date.fromisoformat(e) - d0).days) for e in exps]
        cand = ([x for x in dte if DTE_LO <= x[1] <= DTE_HI]
                or [x for x in dte if x[1] >= 7])
        if not cand:
            return None
        exp = min(cand, key=lambda x: abs(x[1] - DTE_TARGET))[0]
        ivs = [r[0] for r in conn.execute(
            """SELECT iv FROM options_chain
               WHERE snapshot_date=? AND expiration=? AND iv IS NOT NULL
                 AND ABS(strike - ?) <= ?""",
            (snap_date, exp, spot, spot * ATM_BAND_PCT))]
        good = [v for v in ivs if IV_VALID_RANGE[0] <= v <= IV_VALID_RANGE[1]]
        if len(good) < ATM_MIN_QUOTES:
            return None  # 抽风日（近零垃圾值）或报价太薄——如实缺席
        return sum(good) / len(good)
    except Exception:  # noqa: BLE001
        return None


def surprise_score(event_date_et: str, iv_series: dict[str, float]) -> float | None:
    """事件 ET 日的意外分：|当日 ATM IV − 前 5 有效快照日均值| / 基线，封顶 1.

    当日无有效快照、或基线有效日 < BASELINE_MIN_DAYS → None（诚实缺席）。
    """
    iv = iv_series.get(event_date_et)
    if iv is None:
        return None
    prior = [iv_series[k] for k in sorted(iv_series)
             if k < event_date_et][-BASELINE_N:]
    if len(prior) < BASELINE_MIN_DAYS:
        return None
    base = sum(prior) / len(prior)
    if base <= 0:
        return None
    rel = abs(iv - base) / base
    return min(1.0, rel / SURPRISE_REL_FULL)


# ---------------------------------------------------------------- scoring

def position_score(event: dict) -> float:
    """身位分：衍生信号单列，其余按渠道 tier 映射。"""
    if (event.get("source_id") or "") in DERIVED_SOURCE_IDS:
        return POSITION_DERIVED
    return POSITION_BY_TIER.get(event.get("tier") or "", POSITION_DEFAULT)


def fact_score(event_type: str | None) -> float:
    """事实分：精确表 → 前缀规则 → 保守默认。"""
    t = event_type or ""
    if t in FACT_BY_TYPE:
        return FACT_BY_TYPE[t]
    for prefix, score in _FACT_BY_PREFIX:
        if t.startswith(prefix):
            return score
    return FACT_DEFAULT


def source_class(event: dict) -> str:
    """渠道类型（时效半衰期选择）：注册表 → type 前缀兜底 → 保守默认。"""
    sid = event.get("source_id") or ""
    if sid in SOURCE_CLASS:
        return SOURCE_CLASS[sid]
    t = event.get("type") or ""
    for prefix, cls in _CLASS_BY_TYPE_PREFIX:
        if t.startswith(prefix):
            return cls
    return CLASS_DEFAULT


def recency_score(event: dict, now: datetime) -> float:
    """时效分：exp(-ln2·age/半衰期)；日历类恒 1.0；未来事件（age<0）不加成、按 1.0。
    event_time 缺失/不可解析 → 0.0（无时间戳的情报按失效处理，保守）。"""
    hl = HALF_LIFE_DAYS[source_class(event)]
    if hl is None:
        return 1.0
    raw = event.get("event_time_utc")
    try:
        et = datetime.fromisoformat(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if et.tzinfo is None:
        et = et.replace(tzinfo=timezone.utc)
    age_days = (now - et).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return math.exp(-_LN2 * age_days / hl)


def _event_date_et(event: dict) -> str | None:
    """event_time_utc → ET 日（意外分按日对齐快照）；解析失败 None."""
    raw = event.get("event_time_utc")
    try:
        et = datetime.fromisoformat(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if et.tzinfo is None:
        et = et.replace(tzinfo=timezone.utc)
    return str(et.astimezone(_ET).date())


def score_event(event: dict, now: datetime | None = None,
                iv_series: dict[str, float] | None = None) -> dict:
    """四维评分 v1：身位/事实/时效规则计算，意外分 = 期权 IV 基线（可缺席）。

    输入：events 行 dict（source_id / type / event_time_utc，tier 可选）；
          iv_series = load_atm_iv_series(conn) 的结果（不传 = 意外分缺席，v0 行为）。
    输出：
      position / fact / recency  三个规则分（0-1）
      surprise                   0-1（事件日有有效 IV 覆盖时）或 None（缺席）
      total                      四项齐 = 四乘积；意外缺席 = 三乘积
      partial                    True = 意外分缺席的部分评分
      source_class / half_life_days  时效分口径（可追溯）
    """
    now = now or datetime.now(timezone.utc)
    cls = source_class(event)
    pos = position_score(event)
    fact = fact_score(event.get("type"))
    rec = recency_score(event, now)
    sur = None
    if iv_series:
        d_et = _event_date_et(event)
        if d_et is not None:
            sur = surprise_score(d_et, iv_series)
    total = pos * fact * rec * (sur if sur is not None else 1.0)
    return {
        "position": pos,
        "fact": fact,
        "recency": rec,
        "surprise": sur,
        "total": total,
        "partial": sur is None,
        "source_class": cls,
        "half_life_days": HALF_LIFE_DAYS[cls],
    }


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    """单调性自测：tier / type / age 三个维度的分数排序符合设计预期。"""
    from datetime import timedelta

    now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)

    def ev(sid: str, tier: str | None, typ: str, age_days: float) -> dict:
        return {"source_id": sid, "tier": tier, "type": typ,
                "event_time_utc": (now - timedelta(days=age_days)).isoformat()}

    # 1) 身位单调：T0/T1 > T2 > T3；衍生 0.5 介于 T2 与 T3
    s_t1 = score_event(ev("edgar", "T1", "form4", 0), now)
    s_t2 = score_event(ev("fed_fomc", "T2", "fomc_meeting", 0), now)
    s_t3 = score_event(ev("news_rss", "T3", "news_yahoo_tsla", 0), now)
    s_dv = score_event(ev("detector", "T1", "detector_state", 0), now)
    assert s_t1["position"] == POSITION_BY_TIER["T0"] == 1.0
    assert s_t1["position"] > s_t2["position"] > s_dv["position"] > s_t3["position"]

    # 2) 事实单调：披露 1.0 > 日历 0.6 > 叙事 0.2；前缀规则命中 8-K 变体
    assert fact_score("form4") == 1.0 > fact_score("fomc_meeting") == 0.6
    assert fact_score("fomc_meeting") > fact_score("x_musk_post") == 0.2
    assert fact_score("8k_items_2.02_9.01") == FACT_DISCLOSURE
    assert fact_score("news_google_news_tsla") == FACT_NARRATIVE
    assert fact_score("完全未知类型") == FACT_DEFAULT  # 保守兜底

    # 3) 时效单调：同渠道越老越低；age=半衰期 → 恰 0.5；日历类不衰减
    r0 = score_event(ev("edgar", "T1", "form4", 0), now)["recency"]
    r14 = score_event(ev("edgar", "T1", "form4", 14), now)["recency"]
    r28 = score_event(ev("edgar", "T1", "form4", 28), now)["recency"]
    assert r0 == 1.0 and abs(r14 - 0.5) < 1e-9 and abs(r28 - 0.25) < 1e-9
    n2 = score_event(ev("news_rss", "T3", "news_yahoo_tsla", 2), now)["recency"]
    assert abs(n2 - 0.5) < 1e-9  # 新闻半衰期 2 天
    cal = score_event(ev("fed_fomc", "T2", "fomc_meeting", 400), now)
    assert cal["recency"] == 1.0 and cal["half_life_days"] is None
    fut = score_event(ev("fed_fomc", "T2", "fomc_meeting", -90), now)
    assert fut["recency"] == 1.0  # 未来事件不加成
    # 同龄跨渠道：新闻衰减快于数据
    assert (score_event(ev("news_rss", "T3", "news_yahoo_tsla", 5), now)["recency"]
            < score_event(ev("edgar", "T1", "form4", 5), now)["recency"])

    # 4) 总权重 = 三分乘积；意外分诚实缺席
    for s in (s_t1, s_t2, s_t3, s_dv, cal):
        assert s["surprise"] is None and s["partial"] is True
        assert abs(s["total"] - s["position"] * s["fact"] * s["recency"]) < 1e-12
    assert s_t1["total"] > s_t2["total"] > s_t3["total"]

    # 5) 缺时间戳 → 时效 0（保守失效）
    assert score_event({"source_id": "news_rss", "tier": "T3",
                        "type": "news_yahoo_tsla"}, now)["total"] == 0.0

    # 6) 意外分 v1：合成 IV 序列——基线 0.48，当日 0.552（+15% 相对变化）→ 0.5 分；
    #    覆盖日四乘积（partial=False）；无覆盖日/基线不足 → None（partial=True）
    iv = {"2026-07-25": 0.48, "2026-07-26": 0.48, "2026-07-28": 0.48,
          "2026-07-29": 0.48, "2026-07-30": 0.48, "2026-07-31": 0.552}
    assert abs(surprise_score("2026-07-31", iv) - 0.5) < 1e-9
    assert surprise_score("2026-07-25", iv) is None          # 基线不足
    assert surprise_score("2026-07-27", iv) is None          # 当日无有效快照
    big = dict(iv, **{"2026-07-31": 0.48 * 1.5})             # +50% ≥ 封顶
    assert surprise_score("2026-07-31", big) == 1.0
    e31 = ev("edgar", "T1", "form4", 0)                      # now=07-31 12:00 UTC
    s31 = score_event(e31, now, iv_series=iv)
    assert s31["partial"] is False and abs(s31["surprise"] - 0.5) < 1e-9
    assert abs(s31["total"] - s31["position"] * s31["fact"]
               * s31["recency"] * s31["surprise"]) < 1e-12
    s_no = score_event(ev("edgar", "T1", "form4", 20), now, iv_series=iv)
    assert s_no["surprise"] is None and s_no["partial"] is True  # 覆盖期外

    print("selftest OK — 身位/事实/时效单调性通过；意外分 v1（IV 基线）覆盖/缺席行为通过")
    print()
    print(f"{'样例':<38}{'身位':>6}{'事实':>6}{'时效':>6}{'总权重':>8}")
    samples = [
        ("T1 form4，今天", ev("edgar", "T1", "form4", 0)),
        ("T1 form4，14 天前", ev("edgar", "T1", "form4", 14)),
        ("T0 13G 增补，30 天前", ev("edgar_t0", "T0", "sc13g_amend", 30)),
        ("T2 FOMC 日历（不衰减）", ev("fed_fomc", "T2", "fomc_meeting", 60)),
        ("T3 新闻，刚发", ev("news_rss", "T3", "news_yahoo_tsla", 0)),
        ("T3 新闻，4 天前", ev("news_rss", "T3", "news_yahoo_tsla", 4)),
        ("T3 Musk 发帖，1 天前", ev("x_nitter", "T3", "x_musk_post", 1)),
        ("衍生 探测器状态，7 天前", ev("detector", "T1", "detector_state", 7)),
    ]
    for label, e in samples:
        s = score_event(e, now)
        print(f"{label:<38}{s['position']:>6.2f}{s['fact']:>6.2f}"
              f"{s['recency']:>6.2f}{s['total']:>8.3f}")


if __name__ == "__main__":
    _selftest()
