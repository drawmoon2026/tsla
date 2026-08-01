"""拆股折算共享模块 — FINRA 空头利益的全系统正式口径（N5/N6 产物升级，2026-07-24）.

背景：FINRA consolidatedShortInterest 的 short_interest 为**未复权股数**，拆股
跨期的 change_pct 会出现假跳变（实测 TSLA 2020-08-31 期 5:1 拆股 +345.8%、
2022-08-31 期 3:1 拆股 +202.2%，见 outputs/n6_split_audit）。本模块把 N5
（research/n5_cross_pit.py）与 N6（research/n6_split_audit.py）审计时的修正逻辑
沉淀为唯一口径来源，采集器入库前调用（治本），历史 CSV / sentinel.sqlite 已按
同口径批量修正（2026-07-24，原件备份 *.precorrection）。

口径要点：
- SPLITS：硬编码拆股表（effective_date：该日及之后的结算期为拆后股本 × 因子）。
  出处 N5，与 SI×ADV 同步跳变侦测交叉核对（2026-07-24）；COIN 2022-05、
  QCOM 2019-04 等大跳变为真实空头波动，不在表内。
- adjust_change_pct：只重算**跨拆股行**的 change_pct（本期与上期折算因子不同），
  其余行保留 FINRA 原值（因子约掉，避免无谓舍入扰动）；原值存 change_pct_raw、
  标记 split_adjusted=True。幂等（已标记行不重复折算）。
- adjust_levels：水平折算（N5 口径）——short_interest 除以累计因子，统一到
  拆前股本基准，供跨期水平比较（如 si_chg_6wk_pct）；原值存 si_unadjusted。
  水平折算不落盘（CSV/库保留 FINRA 公布的真实股数），由下游读取时调用。
- unexplained_jumps：跳变侦测告警——折算后 change_pct >= SPLIT_GUARD_PCT 且
  拆股表无法解释 → 疑似未登记拆股或极端真实跳变，采集器必须发告警事件/
  显式打印，不得静默入库。

SPLIT_GUARD_PCT 在此定义（intel/detector.py 同源引用，防前向假触发）：
N6 标定——修正后 2018-2026 TSLA 真实双周变动最大 +34.0%、最小拆股因子 2
产生 ~+100% 假跳变，+50% 居中。
"""

from __future__ import annotations

# 疑似拆股伪影阈值（%）：detector.py 的拆股防护与采集器告警共用此值
SPLIT_GUARD_PCT = 50.0

# 拆股表：symbol -> [(effective_date, factor), ...]（按日期升序）
SPLITS: dict[str, list[tuple[str, float]]] = {
    "TSLA": [("2020-08-31", 5), ("2022-08-25", 3)],
    "AAPL": [("2020-08-31", 4)],
    "NVDA": [("2021-07-20", 4), ("2024-06-10", 10)],
    "AMZN": [("2022-06-06", 20)],
    "GOOGL": [("2022-07-18", 20)],
    "WMT": [("2024-02-26", 3)],
    "AVGO": [("2024-07-15", 10)],
    "NFLX": [("2025-11-17", 10)],
}


def split_factor(symbol: str, settlement_date: str) -> float:
    """结算日 >= effective_date 的拆股因子累计乘积（无拆股 = 1.0）。"""
    f = 1.0
    for eff, k in SPLITS.get(symbol, []):
        if settlement_date >= eff:
            f *= k
    return f


def adjust_change_pct(payloads: list[dict], symbol: str) -> list[dict]:
    """就地折算跨拆股行的 change_pct（N6 口径），返回修正明细列表.

    payloads 必须按 settlement_date 升序（首行视为无跨期，不折算——
    需保证抓取覆盖到拆股前的期次，全历史拉取天然满足）。幂等。
    """
    fixes: list[dict] = []
    prev_f: float | None = None
    for p in payloads:
        f = split_factor(symbol, p["settlement_date"])
        if prev_f is None:
            prev_f = f
        if (f != prev_f
                and not p.get("split_adjusted")
                and p.get("short_interest") and p.get("prev_short_interest")):
            corrected = round(
                ((p["short_interest"] / f)
                 / (p["prev_short_interest"] / prev_f) - 1) * 100, 2)
            fixes.append({
                "symbol": symbol,
                "settlement_date": p["settlement_date"],
                "change_pct_raw": p["change_pct"],
                "change_pct_corrected": corrected,
                "factor_prev": prev_f,
                "factor_cur": f,
            })
            p["change_pct_raw"] = p["change_pct"]
            p["change_pct"] = corrected
            p["split_adjusted"] = True
        prev_f = f
    return fixes


def adjust_levels(payloads: list[dict], symbol: str) -> None:
    """就地水平折算（N5 口径）：short_interest /= 累计因子 → 拆前股本基准.

    仅供下游跨期水平比较（si_chg_6wk 等）使用，不落盘。幂等。
    """
    for p in payloads:
        f = split_factor(symbol, p["settlement_date"])
        if f != 1.0 and p.get("short_interest") is not None \
                and "si_unadjusted" not in p:
            p["si_unadjusted"] = p["short_interest"]
            p["short_interest"] = p["short_interest"] / f


def price_split_factor(symbol: str, day: str) -> float:
    """价格层折算因子（P0-2 备用接口）：day（拆前口径日）之后生效的拆股因子累计乘积.

    用法：把 day 当天的拆前价格折算到当前（拆后）口径 → price / price_split_factor。
    与 split_factor 互补：split_factor 数「day 及之前已生效」的因子（股数口径），
    本函数数「day 之后才生效」的因子（价格折算需要的是后续拆股）。
    """
    f = 1.0
    for eff, k in SPLITS.get(symbol, []):
        if day < eff:
            f *= k
    return f


def adjust_price_series(symbol: str, dates: list, closes: list[float]) -> list[float]:
    """把混合口径/拆前口径的日收盘序列统一折算到当前（拆后）口径（P0-2 备用）.

    dates 元素可为 date 或 ISO 字符串；返回新列表，不改原序列。已知拆股表
    （SPLITS）驱动，供价格线在确认拆股后人工重建/校验时调用——正常路径不启用
    （replay_refresh / prices.py 的防线是「检出断裂即中止并告警」，折算须人工确认）。
    """
    return [c / price_split_factor(symbol, str(d)) for d, c in zip(dates, closes)]


def unexplained_jumps(payloads: list[dict], symbol: str) -> list[dict]:
    """折算后仍 change_pct >= SPLIT_GUARD_PCT 的行——拆股表无法解释的大跳变.

    须在 adjust_change_pct 之后调用；命中者为疑似未登记拆股或极端真实跳变，
    采集器据此发告警事件（sqlite 渠道）或显式打印（CSV 渠道），不得静默。
    """
    out: list[dict] = []
    for p in payloads:
        chg = p.get("change_pct")
        if chg is not None and chg >= SPLIT_GUARD_PCT:
            out.append({
                "symbol": symbol,
                "settlement_date": p["settlement_date"],
                "change_pct": chg,
                "short_interest": p.get("short_interest"),
                "guard_pct": SPLIT_GUARD_PCT,
            })
    return out
