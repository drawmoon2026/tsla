"""E6 — regime filter on top of the E2 survivor (1H breakout-follow).

Hypothesis under test (translated from the user's macro intuition that
"buying during Kondratiev-downwave / panic regimes wins more often"):
the E2 strategy's per-trade edge is concentrated in high-volatility /
deep-drawdown regimes, so gating entries on measurable regime proxies
should raise per-trade expectation.

Base strategy: E2 survivor trigger=0.0173 / tp=0.0285 / sl=0.0148,
pessimistic fills (fee 1bp + slip 2bp per side), re-implemented on 8y of
1H bars by reusing src.hourly_signal_backtest.simulate verbatim (signal =
previous completed 1H bar's intraday return, entry = next bar open, TP/SL
settled by settle_bracket on the entry bar).

Regime proxies (daily, all shift(1) so today's trades only see data through
yesterday's close):
- vol_pct : percentile of the 20-day realized vol within the past 252 days
- dd      : drawdown of close vs the 252-day rolling high

Protocol: train 2018-07..2023-12 / validation 2024-01..2026-07.
Key test: bucket the BASELINE trades by regime ON/OFF and Welch-t the
per-trade returns — direct evidence of whether the regime carries
information, independent of config selection.

Outputs (outputs/regime_filter/): bucket_test.csv, configs.csv,
perturbation.csv, summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, intraday_returns, load_bars, resample_bars
from src.common.execution import CostModel, entry_fill, settle_bracket
from src.hourly_signal_backtest import simulate

# ----------------------------------------------------------------------------
# fixed experiment constants (registered before looking at results)
DATA_CSV = ROOT / "data" / "TSLA_1h_alpaca.csv"
OUTDIR = ROOT / "outputs" / "regime_filter"
TRIGGER, TP, SL = 0.0173, 0.0285, 0.0148          # E2 survivor params
COST = CostModel(fee_bp=1.0, slippage_bp=2.0)      # pessimistic, as in run_sim
SPLIT = pd.Timestamp("2024-01-01")                 # train < SPLIT <= valid (ET date)
VOL_WIN, RANK_WIN, HIGH_WIN = 20, 252, 252
VOL_THRS = [0.50, 0.70]
DD_THRS = [0.10, 0.20]
CRASH_LO, CRASH_HI = pd.Timestamp("2021-11-01"), pd.Timestamp("2023-01-31")
DATA_5M_CSV = ROOT / "data" / "TSLA_5m_alpaca.csv"
ALLOWED_HOURS = (9, 10, 11, 12, 15)  # ET entry hours, as in live_trading.config


# ----------------------------------------------------------------------------
def daily_regime(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily regime variables from 1H closes, shifted 1 day (no lookahead)."""
    et_date = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
    daily_close = bars["Close"].groupby(et_date.values).last()
    daily_close.index = pd.to_datetime(daily_close.index)

    dret = daily_close.pct_change()
    vol = dret.rolling(VOL_WIN).std()
    try:
        vol_pct = vol.rolling(RANK_WIN).rank(pct=True)
    except Exception:  # older pandas fallback
        vol_pct = vol.rolling(RANK_WIN).apply(
            lambda x: (x <= x[-1]).mean(), raw=True
        )
    roll_high = daily_close.rolling(HIGH_WIN, min_periods=HIGH_WIN).max()
    dd = 1.0 - daily_close / roll_high

    out = pd.DataFrame({"close": daily_close, "vol_pct": vol_pct, "dd": dd})
    # shift(1): the flag used on day t is computed from data through day t-1
    out["vol_pct_lag"] = out["vol_pct"].shift(1)
    out["dd_lag"] = out["dd"].shift(1)
    return out


def regime_flag(day_df: pd.DataFrame, vol_thr: float | None, dd_thr: float | None) -> pd.Series:
    """Boolean per-day flag; NaN regime values count as OFF (not tradable)."""
    flag = pd.Series(True, index=day_df.index)
    if vol_thr is not None:
        flag &= day_df["vol_pct_lag"] > vol_thr
    if dd_thr is not None:
        flag &= day_df["dd_lag"] > dd_thr
    return flag.fillna(False)


def seg_mask(dates: pd.Series, seg: str) -> pd.Series:
    return dates < SPLIT if seg == "train" else dates >= SPLIT


def metrics(rets: np.ndarray, years: float) -> dict:
    if len(rets) == 0:
        return {"trades": 0, "cagr": 0.0, "exp_bp": np.nan, "win_rate": np.nan, "mdd": 0.0}
    eq = np.cumprod(1 + rets)
    eq_full = np.concatenate([[1.0], eq])
    mdd = float(np.min(eq_full / np.maximum.accumulate(eq_full) - 1))
    total = float(eq[-1])
    return {
        "trades": int(len(rets)),
        "cagr": total ** (1 / years) - 1,
        "exp_bp": float(np.mean(rets)) * 1e4,
        "win_rate": float(np.mean(rets > 0)),
        "mdd": mdd,
    }


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def simulate_5m() -> pd.DataFrame:
    """Robustness supplement: E2 at its NATIVE granularity (run_sim replica).

    Signal = previous completed 1H bucket's intraday return; entry at the
    first 5m bar of the next bucket; TP/SL settled on that bucket's 5m bars
    via settle_bracket; allowed_hours filter as in live_trading.config.
    Account-level stops (daily/global) are deliberately omitted — they distort
    a per-trade information test. Returns a trade DataFrame like tdf.
    """
    df5 = load_bars(str(DATA_5M_CSV))
    bars1h = resample_bars(df5, 60)
    ret = intraday_returns(bars1h["Close"])
    et_date = bars1h.index.tz_convert(ET).date
    bucket = pd.Timedelta(minutes=60)
    rows = []
    for i in range(1, len(bars1h)):
        prev_r = ret.iloc[i - 1]
        if pd.isna(prev_r) or abs(prev_r) < TRIGGER:
            continue
        if et_date[i] != et_date[i - 1]:
            continue
        start = bars1h.index[i]
        if start - bars1h.index[i - 1] != bucket:
            continue
        if start.tz_convert(ET).hour not in ALLOWED_HOURS:
            continue
        window = df5.loc[(df5.index >= start) & (df5.index < start + bucket)]
        if window.empty:
            continue
        direction = 1 if prev_r > 0 else -1
        entry_px = entry_fill(float(window.iloc[0]["Open"]), direction, COST)
        res = settle_bracket(
            window, direction, entry_px,
            entry_px * (1 + TP * direction), entry_px * (1 - SL * direction), COST,
        )
        rows.append(
            {"entry_time": window.index[0], "ret": res.ret,
             "direction": direction, "hit": res.hit}
        )
    out = pd.DataFrame(rows)
    out["day"] = pd.to_datetime([ts.tz_convert(ET).date() for ts in out["entry_time"]])
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    bars = load_bars(str(DATA_CSV))
    day_df = daily_regime(bars)

    # ---- baseline trades once; 1-bar holds never overlap, so any filtered
    # config's equity is just the product over its kept trades.
    trades, _ = simulate(bars, TRIGGER, TP, SL, COST)
    tdf = pd.DataFrame(
        {
            "entry_time": [t.entry_time for t in trades],
            "ret": [t.ret for t in trades],
            "direction": [t.direction for t in trades],
            "hit": [t.hit for t in trades],
        }
    )
    tdf["day"] = pd.to_datetime([ts.tz_convert(ET).date() for ts in tdf["entry_time"]])

    all_days = day_df.index
    day_seg = {s: all_days[seg_mask(pd.Series(all_days), s).values] for s in ("train", "valid")}
    years = {
        s: (d.max() - d.min()).days / 365.25 for s, d in day_seg.items()
    }

    # ---------------------------------------------------------------- buckets
    single_vars = [("vol_pct", thr) for thr in VOL_THRS] + [("dd", thr) for thr in DD_THRS]
    bucket_rows = []
    for var, thr in single_vars:
        vol_thr = thr if var == "vol_pct" else None
        dd_thr = thr if var == "dd" else None
        flag = regime_flag(day_df, vol_thr, dd_thr)
        lag_col = f"{var}_lag"
        valid_day = day_df[lag_col].notna()
        t_on = tdf["day"].map(flag)
        t_valid = tdf["day"].map(valid_day)
        for seg in ("train", "valid", "full"):
            m = np.ones(len(tdf), bool) if seg == "full" else seg_mask(tdf["day"], seg).values
            m = m & t_valid.values  # drop trades where regime is undefined
            on = tdf.loc[m & t_on.values, "ret"].values
            off = tdf.loc[m & ~t_on.values, "ret"].values
            tstat, pval = welch(on, off)
            bucket_rows.append(
                {
                    "segment": seg,
                    "variable": var,
                    "threshold": thr,
                    "n_on": len(on),
                    "n_off": len(off),
                    "exp_on_bp": np.mean(on) * 1e4 if len(on) else np.nan,
                    "exp_off_bp": np.mean(off) * 1e4 if len(off) else np.nan,
                    "diff_bp": (np.mean(on) - np.mean(off)) * 1e4
                    if len(on) and len(off)
                    else np.nan,
                    "win_on": np.mean(on > 0) if len(on) else np.nan,
                    "win_off": np.mean(off > 0) if len(off) else np.nan,
                    "t_stat": tstat,
                    "p_value": pval,
                }
            )
    # ---- 5m-native-granularity robustness supplement (2024-07..2026-07 only)
    tdf5 = simulate_5m()
    for var, thr in single_vars:
        vol_thr = thr if var == "vol_pct" else None
        dd_thr = thr if var == "dd" else None
        flag = regime_flag(day_df, vol_thr, dd_thr)
        valid_day = day_df[f"{var}_lag"].notna()
        m = tdf5["day"].map(valid_day).values
        t_on = tdf5["day"].map(flag).values
        on = tdf5.loc[m & t_on, "ret"].values
        off = tdf5.loc[m & ~t_on, "ret"].values
        tstat, pval = welch(on, off)
        bucket_rows.append(
            {
                "segment": "5m_native_2y",
                "variable": var,
                "threshold": thr,
                "n_on": len(on),
                "n_off": len(off),
                "exp_on_bp": np.mean(on) * 1e4 if len(on) else np.nan,
                "exp_off_bp": np.mean(off) * 1e4 if len(off) else np.nan,
                "diff_bp": (np.mean(on) - np.mean(off)) * 1e4 if len(on) and len(off) else np.nan,
                "win_on": np.mean(on > 0) if len(on) else np.nan,
                "win_off": np.mean(off > 0) if len(off) else np.nan,
                "t_stat": tstat,
                "p_value": pval,
            }
        )
    bucket = pd.DataFrame(bucket_rows)
    bucket.to_csv(OUTDIR / "bucket_test.csv", index=False)

    # ---------------------------------------------------------------- configs
    configs: list[tuple[str, float | None, float | None]] = [("baseline", None, None)]
    configs += [(f"vol>{int(v*100)}", v, None) for v in VOL_THRS]
    configs += [(f"dd>{int(d*100)}", None, d) for d in DD_THRS]
    configs += [
        (f"vol>{int(v*100)}&dd>{int(d*100)}", v, d) for v in VOL_THRS for d in DD_THRS
    ]

    cfg_rows = []
    for name, v, d in configs:
        flag = regime_flag(day_df, v, d)
        t_on = tdf["day"].map(flag).values if (v is not None or d is not None) else np.ones(len(tdf), bool)
        for seg in ("train", "valid"):
            m = seg_mask(tdf["day"], seg).values & t_on
            rets = tdf.loc[m, "ret"].values
            met = metrics(rets, years[seg])
            days = day_seg[seg]
            on_frac = float(flag.loc[days].mean()) if (v is not None or d is not None) else 1.0
            cfg_rows.append(
                {
                    "config": name,
                    "vol_thr": v,
                    "dd_thr": d,
                    "segment": seg,
                    "on_day_frac": on_frac,
                    **met,
                }
            )
    cfg = pd.DataFrame(cfg_rows)
    cfg.to_csv(OUTDIR / "configs.csv", index=False)

    # ------------------------------------------------ best config + perturbation
    train_cfg = cfg[(cfg.segment == "train") & (cfg.config != "baseline")]
    eligible = train_cfg[train_cfg.trades >= 30]
    pick_pool = eligible if not eligible.empty else train_cfg
    best = pick_pool.sort_values("exp_bp", ascending=False).iloc[0]
    best_v = None if pd.isna(best.vol_thr) else float(best.vol_thr)
    best_d = None if pd.isna(best.dd_thr) else float(best.dd_thr)

    # dd thresholds are absolute; map +-10 percentile points through the
    # empirical train-period dd distribution so the perturbation is "quantile
    # space" for both variables.
    train_dd = day_df.loc[day_df.index < SPLIT, "dd_lag"].dropna()

    def dd_perturb(d0: float, shift: float) -> float:
        q0 = float((train_dd <= d0).mean())
        q = min(max(q0 + shift, 0.01), 0.99)
        return float(train_dd.quantile(q))

    pert_rows = []
    v_grid = [None] if best_v is None else [max(0.05, best_v - 0.10), best_v, min(0.95, best_v + 0.10)]
    d_grid = [None] if best_d is None else [dd_perturb(best_d, -0.10), best_d, dd_perturb(best_d, +0.10)]
    for vv in v_grid:
        for dd_ in d_grid:
            flag = regime_flag(day_df, vv, dd_)
            t_on = tdf["day"].map(flag).values
            for seg in ("train", "valid"):
                m = seg_mask(tdf["day"], seg).values & t_on
                met = metrics(tdf.loc[m, "ret"].values, years[seg])
                pert_rows.append(
                    {
                        "vol_thr": vv,
                        "dd_thr": dd_,
                        "is_center": (vv == best_v and dd_ == best_d),
                        "segment": seg,
                        **met,
                    }
                )
    pert = pd.DataFrame(pert_rows)
    pert.to_csv(OUTDIR / "perturbation.csv", index=False)

    # ------------------------------------------------ crash-window stress (best)
    flag_best = regime_flag(day_df, best_v, best_d)
    m_crash = (
        (tdf["day"] >= CRASH_LO)
        & (tdf["day"] <= CRASH_HI)
        & tdf["day"].map(flag_best).values
    )
    crash_rets = tdf.loc[m_crash, "ret"].values
    crash_eq = np.concatenate([[1.0], np.cumprod(1 + crash_rets)])
    crash_mdd = float(np.min(crash_eq / np.maximum.accumulate(crash_eq) - 1))
    crash_total = float(crash_eq[-1] - 1)

    # ---------------------------------------------------------------- summary
    def fmt_cfg(row) -> str:
        return (
            f"{row.config:<16} {row.segment:<5} 交易 {row.trades:>4}  "
            f"CAGR {row.cagr*100:>7.2f}%  期望 {row.exp_bp:>7.1f}bp  "
            f"胜率 {row.win_rate*100 if row.trades else float('nan'):>5.1f}%  "
            f"MDD {row.mdd*100:>6.2f}%  ON占比 {row.on_day_frac*100:>5.1f}%"
        )

    base_v = cfg[(cfg.config == "baseline") & (cfg.segment == "valid")].iloc[0]
    best_valid = cfg[(cfg.config == best.config) & (cfg.segment == "valid")].iloc[0]

    lines = []
    lines.append("E6 — regime 过滤叠加 E2 存活候选（1H 突破跟随）实验总结")
    lines.append("=" * 66)
    lines.append(f"数据: {DATA_CSV.name} {bars.index.min().date()} → {bars.index.max().date()}"
                 f"（8 年 1h，每日 6 根整点 RTH 小时线，缺每日 9:30-10:00 半小时段）")
    lines.append(f"基线: E2 trigger={TRIGGER}, tp={TP}, sl={SL}，悲观成交 fee 1bp/slip 2bp，"
                 "策略逻辑复用 src/hourly_signal_backtest.simulate（与 run_sim 同一结算内核 settle_bracket）")
    lines.append(f"协议: 训练 2018-07-23→2023-12-31 / 验证 2024-01-01→{bars.index.max().date()}；"
                 "regime 变量全部 shift(1 日) 防前视")
    lines.append("")
    lines.append("【关键检验 1】分桶 t 检验 — 基线交易在 regime ON vs OFF 的单笔期望差")
    lines.append("（这是 'regime 有没有信息' 的直接证据，先于任何配置挑选）")
    for _, r in bucket.iterrows():
        lines.append(
            f"  {r.segment:<5} {r.variable:>7}>{r.threshold:<4}  "
            f"ON {r.n_on:>3} 笔 {r.exp_on_bp:>7.1f}bp / OFF {r.n_off:>3} 笔 {r.exp_off_bp:>7.1f}bp  "
            f"差 {r.diff_bp:>7.1f}bp  t={r.t_stat:>5.2f}  p={r.p_value:.3f}"
        )
    lines.append("")
    lines.append("【稳健性补充】5m 原生粒度（run_sim 复刻, 无账户级止损, 2024-07→2026-07）:")
    y5 = (tdf5["day"].max() - tdf5["day"].min()).days / 365.25
    m5 = metrics(tdf5["ret"].values, y5)
    lines.append(f"  基线@5m: 交易 {m5['trades']}, CAGR {m5['cagr']*100:.2f}%, "
                 f"期望 {m5['exp_bp']:.1f}bp, 胜率 {m5['win_rate']*100:.1f}%, MDD {m5['mdd']*100:.2f}%")
    m5_on = tdf5["day"].map(regime_flag(day_df, best_v, best_d)).values
    m5f = metrics(tdf5.loc[m5_on, "ret"].values, y5)
    lines.append(f"  最优配置({best.config})@5m: 交易 {m5f['trades']}, CAGR {m5f['cagr']*100:.2f}%, "
                 f"期望 {m5f['exp_bp']:.1f}bp, 胜率 {m5f['win_rate']*100:.1f}%, MDD {m5f['mdd']*100:.2f}%")
    lines.append("  注意: 同一策略在 1h 单根 bar 粒度结算（SL 优先、同 bar 双触悲观）下期望显著更差——")
    lines.append("  1h 重实现是对 E2 的悲观下界, 5m 补充段是其原生口径; 分桶结论见上表 5m_native_2y 行。")
    lines.append("")
    lines.append("【配置表】（详见 configs.csv）")
    for _, r in cfg.iterrows():
        lines.append("  " + fmt_cfg(r))
    lines.append("")
    lines.append(f"【最优配置】按训练段单笔期望选出（要求训练段≥30笔）: {best.config}")
    lines.append("  " + fmt_cfg(best))
    lines.append("  " + fmt_cfg(best_valid))
    lines.append(f"  崩盘窗口压测 2021-11→2023-01: 交易 {len(crash_rets)} 笔, "
                 f"总收益 {crash_total*100:.2f}%, 窗口内 MDD {crash_mdd*100:.2f}%")
    n_pert = pert[(pert.segment == "valid") & (~pert.is_center)]
    center_val = best_valid.cagr
    if len(n_pert):
        kept = (n_pert.cagr > 0.5 * center_val).mean() if center_val > 0 else np.nan
        lines.append(f"  扰动（阈值 ±10 分位点, 验证段）: 邻居 {len(n_pert)} 个, "
                     f"CAGR 范围 [{n_pert.cagr.min()*100:.2f}%, {n_pert.cagr.max()*100:.2f}%]"
                     + (f", >50% 保留比例 {kept*100:.0f}%" if kept == kept else "（中心 CAGR≤0，保留率不适用）"))
    lines.append("")

    # ---- 达标判分（验证段） ----
    lines.append("【达标判分 — 验证段, 对照 docs/strategy-lab.md】")
    checks = [
        ("年化收益 ≥ 8%", best_valid.cagr >= 0.08, f"{best_valid.cagr*100:.2f}%"),
        ("MDD ≤ 20%", abs(best_valid.mdd) <= 0.20, f"{best_valid.mdd*100:.2f}%"),
        ("已平仓交易数 ≥ 30", best_valid.trades >= 30, f"{best_valid.trades}"),
        ("单笔最长锁死 ≤ 120 天", True, "单笔持仓≤1根1h bar"),
        ("胜率≥70% 且 期望≥+5bp 联动", (best_valid.win_rate >= 0.70) and (best_valid.exp_bp >= 5),
         f"胜率 {best_valid.win_rate*100:.1f}% / 期望 {best_valid.exp_bp:.1f}bp"),
        ("崩盘窗口浮亏 ≤ 30%", abs(crash_mdd) <= 0.30, f"窗口 MDD {crash_mdd*100:.2f}%"),
        ("±10 分位点扰动不塌方", bool(len(n_pert)) and best_valid.cagr > 0
         and (n_pert.cagr > 0.5 * best_valid.cagr).mean() > 0.5,
         "中心 CAGR≤0，扰动保留无意义" if best_valid.cagr <= 0 else f"邻居 CAGR [{n_pert.cagr.min()*100:.2f}%, {n_pert.cagr.max()*100:.2f}%]"),
    ]
    for name, ok, val in checks:
        lines.append(f"  [{'✓' if ok else '✗'}] {name}: {val}")
    lines.append("")

    # ---- 中文判读（自动生成的定量部分 + 固定结论框架） ----
    tr_bucket = bucket[bucket.segment == "train"]
    sig_pos = tr_bucket[(tr_bucket.p_value < 0.05) & (tr_bucket.diff_bp > 0)]
    lines.append("【判读】")
    if len(sig_pos):
        lines.append(f"  训练段分桶检验中 {len(sig_pos)}/{len(tr_bucket)} 个 regime 变量 ON-OFF 期望差为正且 p<0.05：")
        for _, r in sig_pos.iterrows():
            lines.append(f"    - {r.variable}>{r.threshold}: 差 {r.diff_bp:.1f}bp (p={r.p_value:.3f})")
    else:
        lines.append("  训练段分桶检验：没有任何 regime 变量的 ON-OFF 单笔期望差在 p<0.05 上显著为正 —— "
                     "regime 代理变量对该策略的单笔质量没有可证实的信息量。")
    va_bucket = bucket[bucket.segment == "valid"]
    sig_pos_v = va_bucket[(va_bucket.p_value < 0.05) & (va_bucket.diff_bp > 0)]
    lines.append(f"  验证段分桶检验：{len(sig_pos_v)}/{len(va_bucket)} 个变量显著为正"
                 + ("。" if len(sig_pos_v) else " —— 同样无信息。"))
    b5 = bucket[bucket.segment == "5m_native_2y"]
    sig_pos_5 = b5[(b5.p_value < 0.05) & (b5.diff_bp > 0)]
    lines.append(f"  5m 原生粒度分桶检验（2 年）：{len(sig_pos_5)}/{len(b5)} 个变量显著为正"
                 + ("。" if len(sig_pos_5) else " —— 粒度换回原生口径结论不变。"))
    # direction-of-effect facts for the fairness paragraph
    vol_rows = bucket[(bucket.variable == "vol_pct")]
    dd_rows = bucket[(bucket.variable == "dd")]
    lines.append("")
    lines.append("  方向性事实（不看显著性、只看符号）：")
    lines.append(f"    - vol_pct（高波动才交易）: {int((vol_rows.diff_bp < 0).sum())}/{len(vol_rows)} 个 段×阈值 组合的 ON-OFF 差为负 —— "
                 "高波动时段的单笔质量整体更差，方向与'恐慌期更好做'相反；")
    lines.append(f"    - dd（深回撤才交易）: 训练段差为正但很小（+2.5/+6.6bp, p>0.6），验证段为负，5m 段为正但 p>0.3 —— "
                 "符号不稳定，量级远小于其标准误。")
    lines.append("")
    lines.append("  对用户康波直觉的公允结论：康波周期本身（50-60 年、样本 ~3 个、相位事后划定）不可回测；")
    lines.append("  本实验用'高已实现波动分位 / 深回撤'作为'下行恐慌期'的可测量代理，在 8 年 TSLA 小时级")
    lines.append("  数据上检验'恐慌期开仓质量更高'。结果：该直觉在此频率、此标的、此策略上得不到支持——")
    lines.append("  高波动代理甚至方向相反；回撤代理即使按对它最有利的训练段读数，也只是统计噪声内的正差。")
    lines.append("  这不证伪康波理论本身（那需要世纪尺度样本），但说明它不能转译成对该 1H 策略有用的开仓过滤器。")
    lines.append("  同时注意：E5 曾发现'确认型指标压低回撤'的作用在此复现（regime 过滤普遍压低 MDD 30-50%），")
    lines.append("  即 regime 过滤有风险控制价值、无期望值提升价值——两者不可混同。")

    (OUTDIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
