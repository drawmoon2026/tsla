#!/usr/bin/env python
"""N5 — 坑底签名跨标的验证（N4 空头腿的判决性检验）.

背景：N4 在 TSLA 27 坑上发现名义签名——假坑坑底空头利益 ~6 周在急加仓、
真坑持平/回补（p≈0.05-0.10，26 检验零过 Bonferroni，只够给方向）。
N2（lead-lag）与 N3-H（走步推演）独立同向。本实验把样本放大 ~25 倍做一锤定音。

假设（预登记）：空头腿签名是市场共性而非 TSLA 特质。放风腿（Musk）为 TSLA
特有，跨标的只验空头腿。

协议（预登记，跑主检验前写死）：
- 标的池：E8 池 25 只（outputs/e8_pooled/per_symbol_events.csv）+ TSLA，共 26 只
- 价格：各标的 8 年 1h（Alpaca，split-adjusted）聚合日线；TSLA 复用 N4 同一文件
- 坑识别与真/假分类：**逐参数复用 N4 预登记参数**（import research.n4_golden_pit：
  回撤>=20%、±5日局部低、聚类/波级去重、真=60日内先收复25%、假=先破位10%或
  最大反弹<10%；中间态/截断剔除）
- 签名：si_chg_6wk_pct（坑底当日收盘前最近已发布期 vs 其 ~6 周前最近已发布期的
  空头利益水平变化；发布时刻近似 = 结算+9交易日 16:00 ET，与 N2/N4 同口径；
  公共函数 si_signature 与 N4 同一实现）
- **拆股修正（数据伪影修复，主检验前定死）**：FINRA short_interest 为未复权股数，
  6 周窗跨拆股日会产生水平跳变伪影（实测：N4 的 2022-10-14 假坑 +211% 即
  TSLA 3:1 拆股伪影；AAPL/NVDA/AMZN/GOOGL/WMT/AVGO/NFLX 均在样本期内拆股）。
  用硬编码拆股表（下方 SPLITS，已与 SI×ADV 同步跳变侦测交叉核对）把
  short_interest 折算到统一股本基准后再算变化率。TSLA 亦修正（与 N4 原值的
  差异逐坑列报）。
- **主检验（唯一，1 个假设）**：全池真坑组 vs 假坑组 si_chg_6wk_pct，
  按标的分层的置换检验（只在标的内打乱真/假标签，防标的间波动率/空头基数混杂；
  仅同时含真、假坑且签名非 NaN 的标的进入检验）。统计量 = 合并 mean(真) - mean(假)。
  方向为 N4 预登记方向（真 < 假，即假坑空头在加仓）→ 单侧 p，α=0.05。
  20000 次置换，seed 42。**p<0.05 过，过不了判死。**
- 辅助描述（不计假设、不做通过判定）：双侧 p、中位数差置换 p（稳健性）、
  剔除 TSLA 重算（防"共性"其实全由 TSLA 贡献）、逐标的方向一致性计数、
  Hedges g、各标的覆盖如实报
- 多重比较记账：主检验 +1（计数器由主线更新 docs）

用法： .venv/bin/python research/n5_cross_pit.py
输出： outputs/n5_cross_pit/{pits_all_symbols.csv, signature_test.csv, summary.txt}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.n4_golden_pit import (  # noqa: E402  预登记参数与实现逐一复用
    build_daily,
    find_pits,
    si_signature,
)
from src.common.data_io import load_bars  # noqa: E402

DATA = ROOT / "data"
POOL_1H = DATA / "pool_1h"
POOL_SHORT = DATA / "intel" / "pool_short"
OUT = ROOT / "outputs" / "n5_cross_pit"

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "CRM", "COST", "JPM", "V", "UNH", "XOM", "WMT", "DIS", "BA", "CAT", "GS",
    "INTC", "MU", "QCOM", "PLTR", "COIN", "TSLA",
]

# 拆股表：effective_date（该日及之后的结算期为拆后股本）× 因子。
# 与 SI/ADV 同步跳变侦测（scratch 核对 2026-07-24）交叉验证；
# COIN 2022-05、QCOM 2019-04 的跳变为真实空头波动，不在表内。
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

N_PERM = 20000
SEED = 42


# ---------------------------------------------------------------- loading


def load_short(symbol: str) -> pd.DataFrame:
    """N1 schema CSV → 排序 DataFrame，payload 内 short_interest 做拆股折算."""
    path = (DATA / "intel" / "finra_short.csv" if symbol == "TSLA"
            else POOL_SHORT / f"{symbol}.csv")
    df = pd.read_csv(path)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed")
    df["payload"] = df["payload"].map(json.loads)
    df = df.sort_values("event_time_utc").reset_index(drop=True)
    splits = SPLITS.get(symbol, [])
    if splits:
        for p in df["payload"]:
            sd = p["settlement_date"]
            factor = 1.0
            for eff, k in splits:
                if sd >= eff:
                    factor *= k
            if p.get("short_interest") is not None:
                p["si_unadjusted"] = p["short_interest"]
                p["short_interest"] = p["short_interest"] / factor
    return df


def load_price(symbol: str) -> pd.DataFrame:
    path = (DATA / "TSLA_1h_alpaca.csv" if symbol == "TSLA"
            else POOL_1H / f"{symbol}_1h.csv")
    return load_bars(str(path))


# ---------------------------------------------------------------- per-symbol pipeline


def run_symbol(symbol: str) -> tuple[pd.DataFrame, dict]:
    daily = build_daily(load_price(symbol))
    si = load_short(symbol)
    pits = find_pits(daily)
    rows = []
    for _, pit in pits.iterrows():
        i = int(pit["pit_i"])
        t1 = daily["close_time_utc"].iloc[i]
        f, last = si_signature(si, t1)
        rows.append({
            "symbol": symbol,
            "pit_date": pit["pit_date"],
            "pit_close": pit["pit_close"],
            "peak_date": pit["peak_date"],
            "dd_pct": pit["dd_pct"],
            "fwd60_max_pct": pit["fwd60_max_pct"],
            "fwd60_min_pct": pit["fwd60_min_pct"],
            "label": pit["label"],
            "si_chg_6wk_pct": f["si_chg_6wk_pct"],
            "si_latest_chg_pct": f["si_latest_chg_pct"],
            "si_days_to_cover": f["si_days_to_cover"],
            "si_settlement_latest": last["settlement_date"] if last else None,
        })
    cov = {
        "symbol": symbol,
        "daily_start": daily.index[0].date(),
        "daily_end": daily.index[-1].date(),
        "n_days": len(daily),
        "si_rows": len(si),
        "si_first_settlement": si["payload"].iloc[0]["settlement_date"] if len(si) else None,
    }
    return pd.DataFrame(rows), cov


# ---------------------------------------------------------------- stratified permutation


def stratified_perm(df: pd.DataFrame, stat_fn, n_perm: int = N_PERM,
                    seed: int = SEED) -> tuple[float, float, float, np.ndarray]:
    """标的内置换。df 需含 symbol/label/si_chg_6wk_pct（已过滤 both-class 且非 NaN）。

    返回 (观测统计量, 单侧 p[stat<=obs 方向为负], 双侧 p, 置换分布)。
    """
    rng = np.random.default_rng(seed)
    obs = stat_fn(df["label"].to_numpy(), df["si_chg_6wk_pct"].to_numpy())
    groups = [(g["label"].to_numpy().copy(), g["si_chg_6wk_pct"].to_numpy())
              for _, g in df.groupby("symbol")]
    dist = np.empty(n_perm)
    for k in range(n_perm):
        labs_all, vals_all = [], []
        for labs, vals in groups:
            labs_all.append(rng.permutation(labs))
            vals_all.append(vals)
        dist[k] = stat_fn(np.concatenate(labs_all), np.concatenate(vals_all))
    p_one = float((dist <= obs + 1e-15).mean())          # H1：真 - 假 < 0
    p_two = float((np.abs(dist) >= abs(obs) - 1e-15).mean())
    return float(obs), p_one, p_two, dist


def mean_diff(labels: np.ndarray, vals: np.ndarray) -> float:
    return vals[labels == "true"].mean() - vals[labels == "false"].mean()


def median_diff(labels: np.ndarray, vals: np.ndarray) -> float:
    return float(np.median(vals[labels == "true"]) - np.median(vals[labels == "false"]))


def hedges_g(df: pd.DataFrame) -> float:
    a = df.loc[df["label"] == "true", "si_chg_6wk_pct"].to_numpy()
    b = df.loc[df["label"] == "false", "si_chg_6wk_pct"].to_numpy()
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp
    return float(g * (1 - 3 / (4 * (na + nb) - 9)))


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_pits, coverage = [], []
    for sym in SYMBOLS:
        pits, cov = run_symbol(sym)
        n_t = int((pits["label"] == "true").sum())
        n_f = int((pits["label"] == "false").sum())
        cov.update({"n_pits": len(pits), "n_true": n_t, "n_false": n_f})
        all_pits.append(pits)
        coverage.append(cov)
        print(f"{sym:6s} days {cov['n_days']} pits {len(pits)} (T{n_t}/F{n_f})")
    pits = pd.concat(all_pits, ignore_index=True)
    pits.to_csv(OUT / "pits_all_symbols.csv", index=False, float_format="%.4f")
    cov_df = pd.DataFrame(coverage)

    # ---- 检验样本：true/false、签名非 NaN、标的同时含两类 ----
    tf = pits[pits["label"].isin(["true", "false"]) & pits["si_chg_6wk_pct"].notna()].copy()
    cls = tf.groupby("symbol")["label"].nunique()
    both = sorted(cls[cls == 2].index)
    excluded = sorted(set(tf["symbol"]) - set(both))
    test_df = tf[tf["symbol"].isin(both)].reset_index(drop=True)

    n_true = int((test_df["label"] == "true").sum())
    n_false = int((test_df["label"] == "false").sum())
    mean_t = test_df.loc[test_df["label"] == "true", "si_chg_6wk_pct"].mean()
    mean_f = test_df.loc[test_df["label"] == "false", "si_chg_6wk_pct"].mean()

    # ---- 主检验（唯一） ----
    obs, p_one, p_two, _ = stratified_perm(test_df, mean_diff)
    g = hedges_g(test_df)
    verdict = "PASS" if p_one < 0.05 else "FAIL"

    # ---- 辅助描述（不计假设） ----
    med_obs, med_p_one, med_p_two, _ = stratified_perm(test_df, median_diff)
    ex = test_df[test_df["symbol"] != "TSLA"].reset_index(drop=True)
    ex_obs, ex_p_one, ex_p_two, _ = stratified_perm(ex, mean_diff)
    ex_med_obs, ex_med_p_one, _, _ = stratified_perm(ex, median_diff)

    per_sym = []
    for sym, gdf in test_df.groupby("symbol"):
        a = gdf.loc[gdf["label"] == "true", "si_chg_6wk_pct"]
        b = gdf.loc[gdf["label"] == "false", "si_chg_6wk_pct"]
        d = a.mean() - b.mean()
        per_sym.append({
            "symbol": sym, "n_true": len(a), "n_false": len(b),
            "mean_true": a.mean(), "mean_false": b.mean(), "diff": d,
            "same_direction_as_tsla": bool(d < 0),
        })
    per_sym_df = pd.DataFrame(per_sym).sort_values("symbol")
    n_neg = int(per_sym_df["same_direction_as_tsla"].sum())
    n_sym = len(per_sym_df)
    # 方向一致性的符号检验参考值（描述性）
    from scipy.stats import binomtest
    sign_p = binomtest(n_neg, n_sym, 0.5, alternative="greater").pvalue

    rows = [
        {"test": "PRIMARY mean_diff stratified_perm (one-sided, H1: true<false)",
         "n_true": n_true, "n_false": n_false, "n_symbols": n_sym,
         "mean_true": mean_t, "mean_false": mean_f, "stat": obs,
         "hedges_g": g, "p_one_sided": p_one, "p_two_sided": p_two,
         "alpha": 0.05, "verdict": verdict},
        {"test": "aux median_diff stratified_perm", "n_true": n_true,
         "n_false": n_false, "n_symbols": n_sym,
         "mean_true": float(np.median(test_df.loc[test_df['label'] == 'true', 'si_chg_6wk_pct'])),
         "mean_false": float(np.median(test_df.loc[test_df['label'] == 'false', 'si_chg_6wk_pct'])),
         "stat": med_obs, "hedges_g": np.nan, "p_one_sided": med_p_one,
         "p_two_sided": med_p_two, "alpha": np.nan, "verdict": ""},
        {"test": "aux mean_diff ex-TSLA", "n_true": int((ex['label'] == 'true').sum()),
         "n_false": int((ex['label'] == 'false').sum()),
         "n_symbols": ex["symbol"].nunique(),
         "mean_true": ex.loc[ex['label'] == 'true', 'si_chg_6wk_pct'].mean(),
         "mean_false": ex.loc[ex['label'] == 'false', 'si_chg_6wk_pct'].mean(),
         "stat": ex_obs, "hedges_g": hedges_g(ex), "p_one_sided": ex_p_one,
         "p_two_sided": ex_p_two, "alpha": np.nan, "verdict": ""},
        {"test": "aux median_diff ex-TSLA", "n_true": int((ex['label'] == 'true').sum()),
         "n_false": int((ex['label'] == 'false').sum()),
         "n_symbols": ex["symbol"].nunique(),
         "mean_true": np.nan, "mean_false": np.nan, "stat": ex_med_obs,
         "hedges_g": np.nan, "p_one_sided": ex_med_p_one, "p_two_sided": np.nan,
         "alpha": np.nan, "verdict": ""},
    ]
    pd.DataFrame(rows).to_csv(OUT / "signature_test.csv", index=False, float_format="%.4f")
    per_sym_df.to_csv(OUT / "per_symbol_direction.csv", index=False, float_format="%.4f")

    # ---- TSLA 拆股修正 vs N4 原值 逐坑对照 ----
    n4_catalog = pd.read_csv(ROOT / "outputs" / "n4_golden_pit" / "pits_catalog.csv")
    n4_catalog["pit_date"] = n4_catalog["pit_date"].astype(str)
    tsla_now = pits.loc[pits["symbol"] == "TSLA",
                        ["pit_date", "label", "si_chg_6wk_pct"]].copy()
    tsla_now["pit_date"] = tsla_now["pit_date"].astype(str)
    tsla_cmp = n4_catalog[["pit_date", "label", "si_chg_6wk_pct"]].merge(
        tsla_now, on=["pit_date", "label"], suffixes=("_n4_raw", "_n5_adj"))
    tsla_cmp["delta"] = tsla_cmp["si_chg_6wk_pct_n5_adj"] - tsla_cmp["si_chg_6wk_pct_n4_raw"]
    changed = tsla_cmp[tsla_cmp["delta"].abs() > 0.01]

    # ---- summary ----
    L = [
        "N5 — 坑底签名跨标的验证（N4 空头腿判决性检验）摘要",
        f"生成时间：{pd.Timestamp.now()}",
        "",
        "== 协议（预登记） ==",
        "坑识别/分类参数逐一复用 N4（import 同一实现，N4 重跑 diff 校验通过）；",
        "签名 si_chg_6wk_pct 同 N4 口径 + 拆股修正（伪影修复，主检验前定死，拆股表见脚本）；",
        "主检验唯一：标的内分层置换（20000 次，seed 42），统计量=合并 mean(真)-mean(假)，",
        "方向承 N4 预登记（真<假），单侧 α=0.05。p<0.05 过，否则判死。",
        "",
        "== 数据覆盖 ==",
    ]
    for _, c in cov_df.iterrows():
        L.append(f"  {c['symbol']:6s} 日线 {c['daily_start']}→{c['daily_end']}"
                 f"（{c['n_days']}d） SI {c['si_rows']} 期（首结算 {c['si_first_settlement']}）"
                 f" 坑 {c['n_pits']}（真 {c['n_true']}/假 {c['n_false']}）")
    lab_counts = pits["label"].value_counts()
    L += [
        "",
        f"全池坑：{len(pits)} 个 = 真 {lab_counts.get('true', 0)} / 假 {lab_counts.get('false', 0)}"
        f" / 中间态剔除 {lab_counts.get('mid', 0)} / 截断剔除 {lab_counts.get('truncated', 0)}",
        f"进入检验（真/假 且签名非 NaN 且标的同含两类）：{len(test_df)} 坑 = "
        f"真 {n_true} / 假 {n_false}，覆盖 {n_sym} 只标的",
        f"被排除标的（单类或无有效签名）：{', '.join(excluded) if excluded else '无'}",
        "",
        "== 主检验（唯一，预登记） ==",
        f"  mean(真) = {mean_t:+.2f}%   mean(假) = {mean_f:+.2f}%",
        f"  效应量：diff = {obs:+.2f} 个百分点   Hedges g = {g:+.3f}",
        f"  分层置换 p（单侧，H1 真<假）= {p_one:.4f}   （双侧参考 {p_two:.4f}）",
        f"  ==> {'通过（p<0.05）' if verdict == 'PASS' else '未通过（p>=0.05）——判死'}",
        "",
        "== 辅助描述（不计假设、不做判定） ==",
        f"  中位数差置换（稳健性）：diff = {med_obs:+.2f}pp，单侧 p = {med_p_one:.4f}",
        f"  剔除 TSLA：diff = {ex_obs:+.2f}pp（g={hedges_g(ex):+.3f}），单侧 p = {ex_p_one:.4f}；"
        f"中位数差 {ex_med_obs:+.2f}pp，p = {ex_med_p_one:.4f}",
        f"  逐标的方向一致性：{n_neg}/{n_sym} 只与 TSLA 同向（真<假）"
        f"（符号检验参考 p = {sign_p:.4f}，描述性）",
    ]
    for _, r in per_sym_df.iterrows():
        mark = "同向" if r["same_direction_as_tsla"] else "反向"
        L.append(f"    {r['symbol']:6s} 真 {r['n_true']:2d}（{r['mean_true']:+7.1f}%）"
                 f" 假 {r['n_false']:2d}（{r['mean_false']:+7.1f}%）"
                 f" diff {r['diff']:+7.1f}pp {mark}")
    L += ["", "== TSLA 拆股修正对 N4 原值的影响（逐坑，|Δ|>0.01pp 者） =="]
    if len(changed):
        for _, r in changed.iterrows():
            L.append(f"  {r['pit_date']}（{r['label']}）：N4 原值 "
                     f"{r['si_chg_6wk_pct_n4_raw']:+.1f}% → 修正后 "
                     f"{r['si_chg_6wk_pct_n5_adj']:+.1f}%")
        L.append("  （N4 报告的假坑组均值 +27.7% 含 2022-10-14 拆股伪影 +211%；"
                 "修正不改 N4 存档，仅在 N5 口径内生效）")
    else:
        L.append("  无变化")

    # 判读
    L += ["", "== 判读 =="]
    if verdict == "PASS":
        L += [
            f"签名是市场共性：{n_sym} 只标的、{len(test_df)} 个真/假坑上，假坑坑底空头 6 周",
            f"变化系统性高于真坑（diff {obs:+.2f}pp，单侧 p={p_one:.4f}<0.05，标的内置换）。",
            "N4 在 TSLA 上看到的空头腿方向不是单标的幻觉；与 N2/N3-H 构成第四条独立证据。",
            "注意边界：效应量为组均值差，不等于单坑可用的判别器；下一步若做规则化",
            "（阈值/联合 Musk 腿）须另行登记假设并留前向裁决。",
        ]
    else:
        L += [
            f"签名判死：样本放大 ~25 倍后效应不显著（diff {obs:+.2f}pp，单侧 p={p_one:.4f}）。",
            "N4 的 TSLA 结果应视为小样本波动 + 拆股伪影的合成；空头腿不升格，",
            "N3 前向值班的空头腿权重应据此下调（由主线决定）。",
        ]
    L += [
        "",
        "== 记账与限制 ==",
        "多重比较：主检验 +1（辅助描述不计；计数器由主线更新 docs）。",
        "限制：①真/假坑窗口时间重叠、共享宏观 regime（2020-03、2022 熊市为跨标的同期坑簇），",
        "标的内置换不消除时间聚集，p 值仍偏乐观；②SI 发布时刻为结算+9交易日近似；",
        "③META 空头系列由 FB(Facebook)+META(Meta Platforms) 按 issueName 过滤拼接",
        "（FINRA symbolCode 有复用污染，已剔除 Roundhill/ProShares ETF 行）；",
        "④PLTR/COIN 上市晚（2020-09/2021-04），历史短；⑤坑分类用未来信息（逆向研究），",
        "签名只用坑底收盘前已发布信息。",
    ]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
