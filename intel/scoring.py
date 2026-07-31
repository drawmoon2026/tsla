"""四维评分 v0 —— 最小诚实版（纯规则，无 LLM）.

框架依据 docs/intel-framework.md 第二节：每条情报打四个分（0-1），乘积为权重。
v0 实现状态（如实标注，不装）：

  身位分  已实现  渠道 tier 映射（POSITION_BY_TIER；衍生信号单列）
  事实分  已实现  事件 type 规则映射（FACT_BY_TYPE + 前缀规则，可查表）
  时效分  已实现  半衰期指数衰减（HALF_LIFE_DAYS 注册表，按渠道类型标定）
  意外分  未实现  返回 None——需要共识基线数据
                  （待期权隐含波动或分析师预期数据源），缺席时不乘入总权重

  总权重 = 身位 × 事实 × 时效；意外分缺席 → partial=True（部分评分）。

设计约束：
- 只读计算，不回写 events 表（事件表保持采集原貌，评分在查询层实时算）；
- 输入是 events 行 dict（source_id / type / event_time_utc，tier 可选——
  缺 tier 时按 SOURCE_CLASS/保守默认处理）；
- 全部映射表放在模块顶层，可直接查阅与人工复核。

自测： .venv/bin/python -m intel.scoring   （单调性断言 + 样例分数表）
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

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

# ---------------------------------------------------------------- 意外分
# 未实现：需要共识基线（发布前后期权隐含波动 / 分析师预期偏差）才能度量
# "与市场共识的偏离度"。诚实返回 None，总权重不乘、标 partial=True。
SURPRISE_TODO = "待期权隐含波动或分析师预期数据源"


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


def score_event(event: dict, now: datetime | None = None) -> dict:
    """四维评分 v0：身位/事实/时效已计算，意外分未实现（None）。

    输入：events 行 dict（source_id / type / event_time_utc，tier 可选）。
    输出：
      position / fact / recency  三个已实现分（0-1）
      surprise                   None（未实现，见 SURPRISE_TODO）
      total                      身位×事实×时效（意外分缺席不乘）
      partial                    True = 部分评分（意外分缺席）
      source_class / half_life_days  时效分口径（可追溯）
    """
    now = now or datetime.now(timezone.utc)
    cls = source_class(event)
    pos = position_score(event)
    fact = fact_score(event.get("type"))
    rec = recency_score(event, now)
    return {
        "position": pos,
        "fact": fact,
        "recency": rec,
        "surprise": None,  # 未实现：需共识基线（SURPRISE_TODO）
        "total": pos * fact * rec,
        "partial": True,
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

    print("selftest OK — 身位/事实/时效单调性全部通过；意外分 None（partial）")
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
