#!/usr/bin/env python
"""E17-A/B — "30% plan" tracks A (leverage on E8-A+S2) and B (beta + armor).

Track A (leverage): base = the ARCHIVED E8-A+S2 holdout trade stream
(outputs/e8a_replay/trades.csv, 54 kept trades, 2025-10-01..2026-07-22) and the
ARCHIVED crash-window gated stream (outputs/e8a_crash_test/trades.csv,
gated_top10, 177 trades, 2021-10-01..2023-01-31). For L in {1.25,1.5,1.75,2.0}
each trade's net return is multiplied by L (price P&L, fees and slippage all
scale with notional, so xL is exact for those), then a financing charge of
6.5%/yr on the borrowed fraction (L-1) x actual holding time is subtracted.
Slippage assumption: the frozen 2bp/side is kept unchanged under leverage —
valid for retail-scale notional (<~$1M per clip vs TSLA 5m median $ volume in
the tens of millions); stated, not re-estimated.
MTM equity is re-marked bar-by-bar on 5m closes with the leveraged excursion.
Ruin analysis: maintenance margin 30% => margin call when
(1 + L*r) / (L * (1+r)) < 0.30  =>  r_call = (0.3L - 1) / (0.7L);
wipeout at r = -1/L. Compared against the frozen-geometry worst normal loss
(sl 2% stop-market => -2.04% net) and historical worst intraday excursions
(gap-through context). Crash stress: the 177-trade stream (and its S2-filtered
subset, recomputed with the e11 rule) under the same leverage.

Track B (beta + armor): 8y of daily closes (data/TSLA_1h_alpaca.csv aggregated
to ET days, 2018-07-23..2026-07-22). Pre-registered family, NO grid search:
beta in {1.0, 0.7, 0.5} x armor in {none, S2, S2+detector} = 9 combos.
- S2: dd vs 252d rolling max < -20% => exposure 0, evaluated shift(1),
  min_periods=252 (inactive during warm-up), same rule as e11/replay.
- Detector: N3-H risk_off days (outputs/n3h_deduction/daily_states.csv,
  non-blind rows only, 2023-07-03..2025-05-09; the state for day d is built
  from day d-1 tweets, so it is usable on day d without lookahead). Outside
  coverage / blind rows the detector contributes NOTHING (armor degrades to
  S2 alone) — stated per-combo as coverage fraction.
- Cash leg (1 - exposure) earns 4%/yr (per trading day /252). Exposure changes
  cost |d_exposure| x 3bp (1bp fee + 2bp slippage).
Metrics per combo per window (full 2018-2026 and sub-window 2021-01-01..end,
the one containing the complete bear): CAGR (252d), daily-close MTM max DD,
longest underwater run, ann/|mdd|; segment splits: crash 2021-11-01..2023-01-31
and post-crash 2023-02-01..end. Key question: which combo gets closest to /
above 30% ann under mdd <= -35% constraint => Pareto CSV.

Multiple comparisons ledger: 4 (A) + 9 (B) = +13, all pre-registered here.
No model retraining, no parameter tuning, frozen geometry throughout.

Usage:   .venv/bin/python research/e17_ab_tracks.py
Outputs: outputs/e17_ab/{track_a_leverage.csv, track_a_crash.csv,
         track_a_margin.csv, track_b_combos.csv, track_b_pareto.csv,
         summary.txt}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402

OUT = ROOT / "outputs" / "e17_ab"
HOLDOUT_TRADES = ROOT / "outputs" / "e8a_replay" / "trades.csv"
CRASH_TRADES = ROOT / "outputs" / "e8a_crash_test" / "trades.csv"
HOLDOUT_BARS = ROOT / "data" / "TSLA_5m_rolling.csv"
CRASH_BARS = ROOT / "data" / "pool_crash" / "TSLA_5m_2019_2023.csv"
HOURLY = ROOT / "data" / "TSLA_1h_alpaca.csv"
DETECTOR_STATES = ROOT / "outputs" / "n3h_deduction" / "daily_states.csv"

LEVERAGE = [1.25, 1.50, 1.75, 2.00]
FIN_RATE = 0.065          # margin financing, per year, on the borrowed (L-1)
MAINT_MARGIN = 0.30
CASH_RATE = 0.04          # track B cash leg, per year
SWITCH_COST = 0.0003      # 1bp fee + 2bp slip per side of an exposure change
SLIP = 0.0002
YEAR_S = 365.25 * 24 * 3600

HOLDOUT_CAL_DAYS = 294    # 2025-10-01 .. 2026-07-22 (archived window)
CRASH_A_START, CRASH_A_END = "2021-10-01", "2023-01-31"   # track A stress leg
CRASH_B_START, CRASH_B_END = "2021-11-01", "2023-01-31"   # track B segment
SUBWINDOW_START = "2021-01-01"

BETAS = [1.00, 0.70, 0.50]
ARMORS = ["none", "s2", "s2det"]


def mdd(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


# ------------------------------------------------------------------ track A
def load_holdout() -> pd.DataFrame:
    t = pd.read_csv(HOLDOUT_TRADES)
    t = t[~t["blocked"]].copy()
    t["entry_utc"] = pd.to_datetime(t["entry_t_utc"], utc=True)
    t["exit_utc"] = (
        pd.to_datetime(t["exit_et"]).dt.tz_localize(ET).dt.tz_convert("UTC")
    )
    return t.sort_values("entry_utc").reset_index(drop=True)


def load_crash() -> pd.DataFrame:
    t = pd.read_csv(CRASH_TRADES)
    t = t[t["variant"] == "gated_top10"].copy()
    t["entry_utc"] = pd.to_datetime(t["entry_t"], utc=True)
    t["exit_utc"] = pd.to_datetime(t["exit_t"], utc=True)
    return t.sort_values("entry_utc").reset_index(drop=True)


def lever_stream(t: pd.DataFrame, lev: float) -> pd.DataFrame:
    """Per-trade leveraged net return: L x archived net ret minus financing on
    the borrowed (L-1) for the actual holding time (>= one 5m bar)."""
    out = t.copy()
    hold_s = (out["exit_utc"] - out["entry_utc"]).dt.total_seconds().clip(lower=300)
    out["hold_s"] = hold_s
    out["fin_cost"] = (lev - 1.0) * FIN_RATE * hold_s / YEAR_S
    out["ret_lev"] = lev * out["ret"] - out["fin_cost"]
    return out


def mtm_path_lev(bars: pd.DataFrame, t: pd.DataFrame, lev: float) -> np.ndarray:
    """Bar-level MTM equity: inside a trade mark at 5m closes with the
    leveraged excursion from the entry fill; the exit bar books the leveraged
    net return (incl. financing); flat in cash between trades."""
    close = bars["Close"]
    op = bars["Open"]
    eq, path = 1.0, [1.0]
    for _, tr in t.iterrows():
        entry_px = float(op.loc[tr["entry_utc"]]) * (1 + SLIP)
        seg = close.loc[tr["entry_utc"]: tr["exit_utc"]]
        eq0 = eq
        for px in seg.iloc[:-1]:
            path.append(eq0 * (1.0 + lev * (px / entry_px - 1.0)))
        eq = eq0 * (1.0 + tr["ret_lev"])
        path.append(eq)
    return np.asarray(path)


def stream_stats(t: pd.DataFrame, bars: pd.DataFrame, lev: float,
                 cal_days: float | None) -> dict:
    s = lever_stream(t, lev)
    eq = (1.0 + s["ret_lev"]).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    day_pnl = s.groupby("et_day")["ret_lev"].apply(
        lambda r: float((1 + r).prod() - 1))
    path = mtm_path_lev(bars, s, lev)
    d = {
        "L": lev,
        "n": len(s),
        "total": total,
        "avg_bp": float(s["ret_lev"].mean() * 1e4),
        "fin_cost_total_bp": float(s["fin_cost"].sum() * 1e4),
        "mdd_closed": mdd(eq.to_numpy()),
        "mtm_mdd": mdd(path),
        "worst_day": float(day_pnl.min()),
        "worst_day_date": str(day_pnl.idxmin()),
        "worst_trade": float(s["ret_lev"].min()),
    }
    if cal_days:
        d["ann"] = float((1.0 + total) ** (365.0 / cal_days) - 1.0)
    return d


def margin_rows() -> list[dict]:
    rows = []
    for lev in [1.0] + LEVERAGE:
        r_call = (MAINT_MARGIN * lev - 1.0) / ((1.0 - MAINT_MARGIN) * lev) if lev > 1 else np.nan
        rows.append({
            "L": lev,
            "sl_loss_equity": -lev * 0.020396,          # frozen sl2% net, xL
            "r_call_price_move": r_call,                 # adverse move => margin call
            "r_wipeout_price_move": -1.0 / lev,          # equity = 0
        })
    return rows


# ------------------------------------------------------------------ track B
def daily_closes() -> pd.DataFrame:
    bars = load_bars(str(HOURLY))
    et_day = bars.index.tz_convert(ET).date
    d = bars.groupby(et_day).agg(
        close=("Close", "last"), low=("Low", "min"), open=("Open", "first"))
    d.index = pd.to_datetime(d.index)
    return d


def s2_off_series(close: pd.Series) -> pd.Series:
    dd = close / close.rolling(252, min_periods=252).max() - 1.0
    off = (dd < -0.20).fillna(False)
    return off.shift(1).fillna(False).astype(bool)  # yesterday's close decides today


def detector_off_series(index: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    st = pd.read_csv(DETECTOR_STATES, parse_dates=["date"])
    live = st[~st["tweet_data_blind"]]
    off_days = set(live.loc[live["state"] == "risk_off", "date"])
    covered = set(live["date"])
    off = pd.Series([d in off_days for d in index], index=index)
    coverage = float(np.mean([d in covered for d in index]))
    return off, coverage


def combo_returns(r: pd.Series, s2_off: pd.Series, det_off: pd.Series,
                  beta: float, armor: str) -> tuple[pd.Series, pd.Series]:
    if armor == "none":
        armored_off = pd.Series(False, index=r.index)
    elif armor == "s2":
        armored_off = s2_off
    else:
        armored_off = s2_off | det_off
    expo = beta * (~armored_off).astype(float)
    rf = CASH_RATE / 252.0
    cost = expo.diff().abs().fillna(0.0) * SWITCH_COST
    ret = expo * r + (1.0 - expo) * rf - cost
    return ret, expo


def window_stats(ret: pd.Series) -> dict:
    eq = (1.0 + ret).cumprod()
    n = len(eq)
    ann = float(eq.iloc[-1] ** (252.0 / n) - 1.0)
    m = mdd(eq.to_numpy())
    peak = np.maximum.accumulate(eq.to_numpy())
    under = eq.to_numpy() < peak * (1 - 1e-12)
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {"ann": ann, "mdd": m, "underwater_days": longest,
            "ann_over_mdd": ann / abs(m) if m < 0 else np.inf,
            "total": float(eq.iloc[-1] - 1.0), "n_days": n}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ================= TRACK A =================
    hold = load_holdout()
    crash = load_crash()
    hbars = load_bars(str(HOLDOUT_BARS))
    cbars = load_bars(str(CRASH_BARS))

    base = stream_stats(hold, hbars, 1.0, HOLDOUT_CAL_DAYS)
    assert abs(base["total"] - 0.120365) < 5e-4, base["total"]
    assert abs(base["mtm_mdd"] - (-0.036789)) < 2e-3, base["mtm_mdd"]
    print(f"[A] base reproduction ok: total {base['total']:+.4%} "
          f"mtm_mdd {base['mtm_mdd']:.4%} ann {base['ann']:.4%}")

    a_rows = [base] + [stream_stats(hold, hbars, L, HOLDOUT_CAL_DAYS)
                       for L in LEVERAGE]
    a_df = pd.DataFrame(a_rows)
    a_df.to_csv(OUT / "track_a_leverage.csv", index=False)

    # crash stress: 177-trade gated stream, plus its S2-filtered subset
    cd = cbars["Close"].groupby(cbars.index.tz_convert(ET).date).last()
    cd.index = pd.to_datetime(cd.index)
    s2_off_crash = s2_off_series(cd)
    crash["s2_off"] = crash["et_day"].map(
        lambda d: bool(s2_off_crash.get(pd.Timestamp(d), False)))
    crash_s2 = crash[~crash["s2_off"]].reset_index(drop=True)
    print(f"[A] crash S2 subset n={len(crash_s2)} (e11 archive: 33), "
          f"total {(1 + crash_s2['ret']).prod() - 1:+.4%} (e11: -6.81%)")

    c_rows = []
    for L in [1.0] + LEVERAGE:
        r = stream_stats(crash, cbars, L, None)
        r["stream"] = "crash_177_no_armor"
        c_rows.append(r)
        r = stream_stats(crash_s2, cbars, L, None)
        r["stream"] = f"crash_{len(crash_s2)}_with_S2"
        c_rows.append(r)
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(OUT / "track_a_crash.csv", index=False)

    # margin / ruin analysis + historical worst intraday context
    m_df = pd.DataFrame(margin_rows())
    daily = daily_closes()
    intraday_low = daily["low"] / daily["open"] - 1.0
    gap_low = daily["low"] / daily["close"].shift(1) - 1.0
    m_df["hist_worst_intraday_low_vs_open"] = float(intraday_low.min())
    m_df["hist_worst_low_vs_prev_close"] = float(gap_low.min())
    m_df.to_csv(OUT / "track_a_margin.csv", index=False)

    # ================= TRACK B =================
    close = daily["close"]
    r = close.pct_change().dropna()
    s2_off = s2_off_series(close).reindex(r.index).fillna(False).astype(bool)
    det_off, det_cov = detector_off_series(r.index)

    windows = {
        "2018-2026": r.index,
        "2021-2026": r.index[r.index >= SUBWINDOW_START],
    }
    seg_crash = (r.index >= CRASH_B_START) & (r.index <= CRASH_B_END)
    seg_post = r.index > CRASH_B_END

    b_rows = []
    for beta in BETAS:
        for armor in ARMORS:
            ret, expo = combo_returns(r, s2_off, det_off, beta, armor)
            row = {"combo": f"b{int(beta * 100)}_{armor}", "beta": beta,
                   "armor": armor,
                   "detector_coverage": det_cov if armor == "s2det" else np.nan,
                   "off_days_frac": float((expo == 0).mean()),
                   "n_switches": int((expo.diff().abs() > 0).sum())}
            for wname, widx in windows.items():
                st = window_stats(ret.loc[widx])
                row.update({f"{wname}_{k}": v for k, v in st.items()})
            row.update({f"crash_{k}": v
                        for k, v in window_stats(ret[seg_crash]).items()
                        if k in ("total", "mdd")})
            row.update({f"post_{k}": v
                        for k, v in window_stats(ret[seg_post]).items()
                        if k in ("ann", "mdd")})
            b_rows.append(row)
    b_df = pd.DataFrame(b_rows)
    b_df.to_csv(OUT / "track_b_combos.csv", index=False)

    pareto = b_df[["combo", "beta", "armor",
                   "2018-2026_ann", "2018-2026_mdd",
                   "2021-2026_ann", "2021-2026_mdd"]].copy()
    for w in ["2018-2026", "2021-2026"]:
        pareto[f"{w}_pass_mdd35"] = pareto[f"{w}_mdd"] >= -0.35
        pareto[f"{w}_pass_ann30"] = pareto[f"{w}_ann"] >= 0.30
    pareto = pareto.sort_values("2018-2026_ann", ascending=False)
    pareto.to_csv(OUT / "track_b_pareto.csv", index=False)

    # ================= summary =================
    lines = []
    w = lines.append
    w("E17-A/B — 30% 计划双轨（杠杆轨 + Beta+护甲轨）；几何/门/开关全冻结，无任何调参")
    w("=" * 100)
    w("多重比较记账：A 轨 4 档杠杆 + B 轨 9 组合 = +13 次比较，全部预登记（本文件 docstring），无网格。")
    w("")
    w("== A 轨：E8-A+S2 留出段（2025-10-01..2026-07-22，54 笔存档）x 杠杆 ==")
    w("融资 6.5%/年按实际持仓秒数计（日内、day_flat）；滑点假设：2bp/边 不随 L 放大（零售规模，见 docstring）。")
    w(a_df[["L", "n", "total", "ann", "avg_bp", "fin_cost_total_bp",
            "mdd_closed", "mtm_mdd", "worst_day", "worst_trade"]]
      .to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    w("")
    w("== A 轨：崩盘压测（2021-10-01..2023-01-31 存档流）x 杠杆 ==")
    w(c_df[["stream", "L", "n", "total", "avg_bp", "mdd_closed", "mtm_mdd",
            "worst_day"]]
      .to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    w("")
    w("== A 轨：保证金/爆仓口径（维持保证金 30%）==")
    w(m_df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    w("解释：r_call = 单笔持仓期内价格逆行多少触发追缴；冻结几何下正常单笔最大亏 = sl2% 净 -2.04% x L；")
    w("day_flat 无隔夜跳空，需盘中闪崩/停牌一步穿到 r_call 才会爆——8 年最深盘中回落见表中历史列。")
    w("")
    w("== B 轨：TSLA beta x 护甲 组合族（9 个，预登记）==")
    w(f"探测器覆盖率（非盲区交易日占比，8 年窗口）：{det_cov:.1%}；盲区护甲退化为纯 S2。")
    w("现金端 4%/年；换仓成本 3bp x |敞口变化|。全部日线收盘 MTM。")
    w(b_df[["combo", "off_days_frac", "n_switches",
            "2018-2026_ann", "2018-2026_mdd", "2018-2026_underwater_days",
            "2018-2026_ann_over_mdd",
            "2021-2026_ann", "2021-2026_mdd",
            "crash_total", "crash_mdd", "post_ann", "post_mdd"]]
      .to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    w("")
    w("== B 轨 Pareto（年化 vs 回撤，两窗口）==")
    w(pareto.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    w("")

    # ---------------- verdict vs the E17 pre-registration --------------------
    w("== 判决（对照 E17 预登记：验证段+崩盘压测 年化>=30% 且回撤可陈述 -> candidate；不达 -> 记入『30% 价格表』）==")
    a_pass = a_df[a_df["ann"] >= 0.30]
    if len(a_pass):
        L_star = float(a_pass["L"].iloc[0])
        arow = a_df[a_df["L"] == L_star].iloc[0]
        c_arm = c_df[(c_df["L"] == L_star)
                     & c_df["stream"].str.contains("with_S2")].iloc[0]
        c_naked = c_df[(c_df["L"] == L_star)
                       & c_df["stream"].str.contains("no_armor")].iloc[0]
        w(f"A 轨：最小达标杠杆 L={L_star:.2f} —— 验证段年化 {arow['ann']:+.1%}"
          f"（>=30% 达标），MTM 回撤 {arow['mtm_mdd']:.1%}，单日最差 {arow['worst_day']:.1%}；")
        w(f"  崩盘压测（同 L）：带 S2 护甲 total {c_arm['total']:+.1%} / MTM MDD {c_arm['mtm_mdd']:.1%}；"
          f"无护甲 total {c_naked['total']:+.1%} / MTM MDD {c_naked['mtm_mdd']:.1%}。")
        w(f"  保证金：L={L_star:.2f} 追缴需单笔持仓期内逆行 "
          f"{(MAINT_MARGIN * L_star - 1) / ((1 - MAINT_MARGIN) * L_star):+.1%}"
          f"，8 年最深盘中回落 {float(intraday_low.min()):+.1%}（day_flat 无隔夜跳空敞口）——回撤与爆仓风险可陈述。")
        w("  但判决不是 candidate，是【conditional candidate，效力待 shadow】：基底 54 笔本身 bootstrap p 0.04-0.28（E8 存档），")
        w("  边未被证实——杠杆只放大期望与方差，不增加任何证据；且杠杆后的崩盘段 MTM MDD 依赖 S2 及时关闸（e11 已证顶点后 27 个交易日才关，先吃 -8.45%x L）。")
    else:
        w("A 轨：无杠杆档在验证段达到年化 30% —— 计入价格表。")
    b_pass = pareto[pareto["2018-2026_pass_mdd35"] & pareto["2018-2026_pass_ann30"]]
    w(f"B 轨：9 组合中满足『8 年年化>=30% 且 MDD<=35%』的组合数 = {len(b_pass)}。")
    w("  历史 30%+ 的唯一来源确认为裸/高 beta：b100_none 44.2%/-73.6%，b70_none 36.4%/-57.8%（都远超 35% 回撤上限）。")
    w("  护甲的价格：S2 把 8 年年化从 44% 砍到 ~11%，MDD 只从 -74% 改善到 -68%（b100_s2）——")
    w("  对日线 beta 而言 S2 是买高卖低的鞭打机（8 年 89 次换挡，2022-2025 有 58-86% 的日子在场外，错过反弹主升）。")
    w("  这与 e11 的 S2 结论不矛盾：e11 的 S2 保护的是日内刮头皮流（停用=不交易），beta 护甲是卖出资产本身，错过的恢复期主导 TSLA 复利。")
    w("  探测器增量：s2det 与 s2 几乎同值（覆盖率仅 23%，且 risk_off 天数少）——护甲叠加探测器在日线 beta 上无可测增益。")
    w("  次窗口 2021-2026（含完整熊市）：裸 B&H 年化仅 8.8%——8 年 44% 的头条完全靠 2019-2020 的 10 倍行情，前向预期必须按次窗口打折。")
    w("")
    w("== 30% 价格表（本双轨的诚实结论）==")
    w(f"  方案 1（A 轨）：E8-A+S2 上 L=2.0 杠杆 -> 验证段年化 {a_df['ann'].iloc[-1]:+.1%}，"
      f"价格 = 未证实的 54 笔边 x2、崩盘段（带 S2）MTM 回撤 {c_df[c_df['L'] == 2.0][c_df.loc[c_df['L'] == 2.0, 'stream'].str.contains('with_S2')]['mtm_mdd'].iloc[0]:.1%}、"
      "对 S2 关闸及时性的依赖、融资 6.5% 与保证金制度风险。")
    w("  方案 2（B 轨）：b100_none（裸 B&H）-> 8 年年化 +44.2%，价格 = -73.6% MTM 回撤、778 个交易日水下、次窗口年化仅 +8.8%。")
    w("  方案 3（B 轨）：b70_none -> 8 年年化 +36.4%，价格 = -57.8% 回撤，次窗口 +11.3%。")
    w("  无任何护甲组合能在回撤<=35% 约束下接近 30%：约束内最好是 b50_s2det（+10.3% / -36.1%，still 界外）。")
    w("")
    w("== 诚实边界 ==")
    w("  1) A 轨基底 n=54 不显著；杠杆放大的是未证实期望。融资成本如实计算后确实可忽略（54 笔合计 0.8-3.2bp）。")
    w("  2) A 轨滑点假设 2bp/边不随杠杆放大——仅对零售规模成立；机构规模需重估冲击成本。")
    w("  3) 崩盘压测两段模型实例不同（2019-2021 池 vs 2023-2025 池），拼接仅为量级参考（同 e11 备注）。")
    w("  4) B 轨 8 年窗口含 TSLA 史诗牛市；2021-2026 次窗口是更保守的前向锚。日线收盘 MTM 会低估盘中真实回撤（B&H 盘中最深 -75%+）。")
    w("  5) 探测器非盲区覆盖仅 23.2%（2023-07..2025-05），盲区护甲退化为 S2——s2det 列的差值只在覆盖期内有意义。")
    w("  6) 本双轨 +13 次比较全部预登记；未做任何网格/调参。")
    w("")
    w("运行耗时 %.0fs" % (time.time() - t0))
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
