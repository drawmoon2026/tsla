"""统一美股（NYSE）交易日历 — 全项目单一口径来源.

背景（outputs/ops_review/review.md P2-1 / P1-8、outputs/detector_rehearsal/report.md
Labor Day 专项）：全项目原用 numpy busday（周一至周五）近似交易日，不剔美股假日、
不识别半日市——F20 端点跨 Labor Day 偏 1 天（彩排实证）、半日市 13:00 后 shadow
不收工。本模块把 2026–2027 的 NYSE 假日表与半日市表按官方日历硬编码，提供统一
API，替换零散的 busday 近似——一次修，全链准。

口径说明（重要）：冻结规则里"F20 = 20 个交易日"等语义**不变**——只是"交易日"
的定义从 busday 近似修正为真实 NYSE 日历。此为口径修正非规则变更（类比 N6
拆股修正先例：修的是数据口径伪影，不动冻结参数）。2026-08-02 接线；此前已
落库的历史行按当时口径产生，不追溯改写。

越界 fallback：2026–2027 之外的年份查询时退回 busday 近似（周一至周五）。
**未来**年份（2028 起）每年份打一次警告——假日表需人工年度续期（NYSE 通常提前
2-3 年公布），维护事项已记入 intel/README.md；**过去**年份（2025 及更早）静默
fallback 不告警——历史数据本就在 busday 近似口径下产生（如 finra 历史发布的
act 日推算），刷警告只会淹没日志。

出处：NYSE 官方假日与交易时间页 https://www.nyse.com/markets/hours-calendars
（2026 与 2027 年表，含半日市脚注"1:00 p.m. early close"）。

API（date 均为 ET 交易日语义的 datetime.date）：
    is_trading_day(d)          是否交易日
    next_trading_day(d)        严格晚于 d 的第一个交易日
    prev_trading_day(d)        严格早于 d 的最后一个交易日
    add_trading_days(d, n)     n>0：先把 d 滚到首个 >= d 的交易日，再前进 n 个交易日
                               （n=0 只滚不进；n<0 反向：滚到 <= d 的交易日再后退）
    trading_days_between(a, b) [a, b) 半开区间内的交易日数（b<=a 返回 0）
    close_time_et(d)           当日收盘时刻 ET（16:00；半日市 13:00）。
                               对非交易日返回 16:00 仅作占位，调用方自行先判交易日。

自测：.venv/bin/python -m intel.market_calendar --selftest
"""

from __future__ import annotations

from datetime import date, time as dt_time, timedelta

FULL_CLOSE_ET = dt_time(16, 0)   # 常规收盘
HALF_CLOSE_ET = dt_time(13, 0)   # 半日市收盘（NYSE "1:00 p.m. early close"）

# ---- NYSE 全日休市（出处见文件头；按年硬编码，人工年度续期） -----------------
HOLIDAYS: dict[int, tuple[date, ...]] = {
    2026: (
        date(2026, 1, 1),    # New Year's Day（周四）
        date(2026, 1, 19),   # Martin Luther King, Jr. Day
        date(2026, 2, 16),   # Washington's Birthday / Presidents' Day
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth National Independence Day（周五）
        date(2026, 7, 3),    # Independence Day 观察日（7/4 为周六 → 周五休市；
                             #   注意 2026 年 7 月 3 日是全日休市，不是半日市）
        date(2026, 9, 7),    # Labor Day —— 出闸后首个假日（F20 修正的主案例）
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day（周五）
    ),
    2027: (
        date(2027, 1, 1),    # New Year's Day（周五）
        date(2027, 1, 18),   # Martin Luther King, Jr. Day
        date(2027, 2, 15),   # Washington's Birthday / Presidents' Day
        date(2027, 3, 26),   # Good Friday
        date(2027, 5, 31),   # Memorial Day
        date(2027, 6, 18),   # Juneteenth 观察日（6/19 为周六 → 周五休市）
        date(2027, 7, 5),    # Independence Day 观察日（7/4 为周日 → 周一休市）
        date(2027, 9, 6),    # Labor Day
        date(2027, 11, 25),  # Thanksgiving Day
        date(2027, 12, 24),  # Christmas 观察日（12/25 为周六 → 周五休市）
        # 备注：2028-01-01 为周六，但 NYSE Rule 7.2 年末例外（yearly accounting
        # period）—— 2027-12-31（周五）照常全日交易，不前移休市（先例：2021-12-31）。
    ),
}

# ---- NYSE 半日市（13:00 ET 收盘；出处同上） --------------------------------
HALF_DAYS: dict[int, tuple[date, ...]] = {
    2026: (
        date(2026, 11, 27),  # 感恩节次日（周五）
        date(2026, 12, 24),  # 圣诞前夕（周四；12/25 休市）
        # 2026 年无 7/3 半日市——该日是 Independence Day 观察日全日休市（见上）
    ),
    2027: (
        date(2027, 11, 26),  # 感恩节次日（周五）
        # 2027 年无 12/24 半日市——该日是 Christmas 观察日全日休市；
        # 12/23（周四）为全日交易（先例：2021-12-23）。7 月亦无半日市（7/5 休市）。
    ),
}

_HOLIDAY_SET = frozenset(d for ds in HOLIDAYS.values() for d in ds)
_HALF_SET = frozenset(d for ds in HALF_DAYS.values() for d in ds)
_COVERED_YEARS = frozenset(HOLIDAYS)
_warned_years: set[int] = set()


def _covered(d: date) -> bool:
    """该日期年份是否在硬编码表内；未来年份越界打一次性警告并 fallback.

    过去年份（表覆盖起点之前）静默 fallback——历史即 busday 近似口径，不告警。
    """
    if d.year in _COVERED_YEARS:
        return True
    if d.year > max(_COVERED_YEARS) and d.year not in _warned_years:
        _warned_years.add(d.year)
        print(f"[market_calendar] 警告：{d.year} 年不在硬编码 NYSE 日历表内"
              f"（表覆盖 {min(_COVERED_YEARS)}–{max(_COVERED_YEARS)}）——"
              "退回 busday 近似（周一至周五，不剔假日/半日市）。"
              "请按 NYSE 官方日历人工续期 intel/market_calendar.py"
              "（维护事项见 intel/README.md）。")
    return False


def is_trading_day(d: date) -> bool:
    """是否 NYSE 交易日（周末与表内假日为 False；越界年份退回周一至周五近似）."""
    if d.weekday() >= 5:
        return False
    if _covered(d):
        return d not in _HOLIDAY_SET
    return True  # fallback：busday 近似


def next_trading_day(d: date) -> date:
    """严格晚于 d 的第一个交易日."""
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    """严格早于 d 的最后一个交易日."""
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def add_trading_days(d: date, n: int) -> date:
    """交易日位移（与原 dashboard.add_bdays / np.busday_offset 用法对齐）.

    n>0：先把 d 向前滚到首个 >= d 的交易日，再前进 n 个交易日；
    n=0：只滚不进；n<0：先向后滚到首个 <= d 的交易日，再后退 |n| 个交易日。
    """
    if n >= 0:
        while not is_trading_day(d):
            d += timedelta(days=1)
        for _ in range(n):
            d = next_trading_day(d)
    else:
        while not is_trading_day(d):
            d -= timedelta(days=1)
        for _ in range(-n):
            d = prev_trading_day(d)
    return d


def trading_days_between(a: date, b: date) -> int:
    """[a, b) 半开区间内的交易日数（与 np.busday_count 同语义）；b<=a 返回 0."""
    if b <= a:
        return 0
    n, d = 0, a
    while d < b:
        if is_trading_day(d):
            n += 1
        d += timedelta(days=1)
    return n


def close_time_et(d: date) -> dt_time:
    """当日收盘时刻（ET）：常规 16:00；半日市 13:00.

    仅对交易日有意义；非交易日返回 16:00 占位（调用方先用 is_trading_day 判断）。
    越界年份 fallback 恒 16:00（busday 近似无半日市概念）。
    """
    if _covered(d) and d in _HALF_SET:
        return HALF_CLOSE_ET
    return FULL_CLOSE_ET


# ------------------------------------------------------------------ selftest

def _selftest() -> None:
    import numpy as np

    # 表自洽：假日/半日不落周末、互不重叠、年份对齐
    for year, ds in HOLIDAYS.items():
        for d in ds:
            assert d.year == year and d.weekday() < 5, d
    for year, ds in HALF_DAYS.items():
        for d in ds:
            assert d.year == year and d.weekday() < 5 and d not in _HOLIDAY_SET, d
    # 基础判定
    assert not is_trading_day(date(2026, 9, 7))          # Labor Day
    assert not is_trading_day(date(2026, 9, 5))          # 周六
    assert is_trading_day(date(2026, 9, 8))
    assert next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)   # 周五→跳假日周一
    assert prev_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)
    # F20 修正主案例（彩排 Labor Day 专项）：08-26 起第 19 个交易日
    # busday 近似 = 09-22；真实日历（跳 09-07）= 09-23
    assert add_trading_days(date(2026, 8, 26), 19) == date(2026, 9, 23)
    assert str(np.busday_offset(date(2026, 8, 26), 19)) == "2026-09-22"  # 旧口径对照
    # 计数语义与 busday_count 对齐（无假日区间完全一致）
    a, b = date(2026, 8, 3), date(2026, 8, 29)
    assert trading_days_between(a, b) == int(np.busday_count(a, b)) == 20
    # 含 Labor Day 的区间少 1 个交易日
    assert trading_days_between(date(2026, 9, 4), date(2026, 9, 9)) == 2   # 09-04, 09-08
    assert int(np.busday_count(date(2026, 9, 4), date(2026, 9, 9))) == 3
    # 半日市收盘
    assert close_time_et(date(2026, 11, 27)) == dt_time(13, 0)   # 感恩节次日
    assert close_time_et(date(2026, 12, 24)) == dt_time(13, 0)   # 圣诞前夕
    assert close_time_et(date(2027, 11, 26)) == dt_time(13, 0)
    assert close_time_et(date(2026, 11, 30)) == dt_time(16, 0)
    # 2026-07-03 是全日休市（非半日市）；2027-12-24 同理
    assert not is_trading_day(date(2026, 7, 3))
    assert not is_trading_day(date(2027, 12, 24))
    assert is_trading_day(date(2027, 12, 31))    # Rule 7.2 年末例外，照常交易
    # 感恩节 2026：11-26 休市、11-27 半日
    assert not is_trading_day(date(2026, 11, 26))
    assert is_trading_day(date(2026, 11, 27))
    # add_trading_days 负位移与 n=0 滚动
    assert add_trading_days(date(2026, 9, 7), 0) == date(2026, 9, 8)    # 假日滚到次日
    assert add_trading_days(date(2026, 9, 8), -1) == date(2026, 9, 4)
    # 越界 fallback：未来年份（2028）busday 近似 + 一次性警告；过去年份静默
    assert is_trading_day(date(2028, 1, 3))      # 周一，fallback 视为交易日
    assert "2028" in " ".join(str(y) for y in _warned_years)
    assert close_time_et(date(2028, 11, 24)) == dt_time(16, 0)
    assert is_trading_day(date(2025, 9, 1))      # 2025 Labor Day：历史静默 fallback
    assert 2025 not in _warned_years             # 过去年份不刷续期警告
    print("market_calendar selftest: OK（假日表 2026–2027，Labor Day/半日市/越界 fallback 全过）")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="跑内置自测")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        today = date.today()
        print(f"{today}: trading={is_trading_day(today)} "
              f"close={close_time_et(today)} next={next_trading_day(today)}")
