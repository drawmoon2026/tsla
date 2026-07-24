#!/usr/bin/env python
"""E8 — GBDT cross-symbol pooled training for the 5m V-rebound filter.

Hypothesis under test (docs/strategy-lab.md E8): single-symbol data volume is
the binding constraint behind the ~0.60 AUC ceiling. If true, training on
~24 other liquid large caps and inferring on TSLA (leave-TSLA-out) should
approach or beat TSLA self-training; pooling (incl. TSLA) should beat it.

Protocol (fixed before results)
-------------------------------
- Events: src/v_stats.py 5m unique tables per symbol (same detector params).
- Features/labels: research/ml_common.py — identical construction to
  ml_filter.py, then return-scale features are vol-normalized per event
  (divided by that symbol's rv20 at confirm) + rv20_rel (rv20 / symbol's
  TRAIN-period median rv20) + `symbol` categorical. Label: same triple
  barrier (+0.5% / -1.0% / 24 bars / day-end), fixed across symbols.
- Split, purged by time: train = ET day < 2025-09-30 (one-day embargo),
  holdout = ET day >= 2025-10-01. Labels resolve intraday (<=2h), so the
  day boundary alone already purges label overlap; the embargo is belt and
  braces.
- (a) leave-TSLA-out: train on the other symbols' train segment -> TSLA
  holdout AUC.  (b) TSLA self-train (same split; reported with the original
  ml_filter feature set AND with the normalized set, to separate feature
  effects from data-volume effects).  (c) pooled incl. TSLA train segment ->
  TSLA holdout.  (d) leave-one-out AUC for every symbol.
- Bootstrap 95% CI on each headline AUC (1000 resamples of test events).
- Only if pooled/LOO AUC >= 0.63 on the TSLA holdout: re-run the E9 family-A
  grid (gate x geometry x timeout) on the TSLA holdout with the new
  probabilities, settled via research/e9_frontier_search.py machinery
  (settle_bracket, pessimistic costs), and diff against E9's stored
  family-A holdout results.

Usage:  .venv/bin/python research/e8_pooled_gbdt.py
Outputs: outputs/e8_pooled/{per_symbol_events.csv, auc_comparison.csv,
         per_symbol_loo_auc.csv, frontier_shift.csv (conditional),
         summary.txt}
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from research.ml_common import FEATURES, build_dataset, load_events, make_lgbm  # noqa: E402
from src.common.data_io import load_bars  # noqa: E402

# ----------------------------------------------------------------- parameters
POOL_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "CRM", "COST", "JPM", "V", "UNH", "XOM", "WMT", "DIS", "BA", "CAT", "GS",
    "INTC", "MU", "QCOM", "PLTR", "COIN",
]
TARGET = "TSLA"

POOL_DIR = ROOT / "data" / "pool"
EVENTS_POOL_DIR = ROOT / "outputs" / "v_events_pool"
TSLA_BARS = ROOT / "data" / "TSLA_5m_3y.csv"
TSLA_EVENTS = ROOT / "outputs" / "v_events_3y" / "v_events_5m_unique.csv"
OUT = ROOT / "outputs" / "e8_pooled"

TRAIN_END = date(2025, 9, 30)      # train: et_day <  TRAIN_END (embargo day dropped)
HOLDOUT_START = date(2025, 10, 1)  # test : et_day >= HOLDOUT_START
AUC_GATE_FOR_FRONTIER = 0.63
N_BOOT = 1000
SEED = 42

# vol-normalized pooled feature set (return-scale features / per-event rv20)
POOLED_FEATURES = [
    "drop_pct_n", "drop_speed_n", "vol_ratio_trough", "et_hour",
    "open_to_now_ret_n", "prev_day_ret_n", "rv20_rel", "bars_since_open",
    "symbol",
]


# ------------------------------------------------------------------ data prep
def build_symbol_dataset(sym: str) -> tuple[pd.DataFrame, dict] | None:
    if sym == TARGET:
        bars_csv, events_csv = TSLA_BARS, TSLA_EVENTS
    else:
        bars_csv = POOL_DIR / f"{sym}_5m_3y.csv"
        events_csv = EVENTS_POOL_DIR / sym / "v_events_5m_unique.csv"
    if not Path(bars_csv).exists() or not Path(events_csv).exists():
        return None
    ev = load_events(Path(events_csv))
    bars = load_bars(str(bars_csv))
    ds, info = build_dataset(ev, bars)
    ds["symbol"] = sym
    info["n_events_raw"] = len(ev)
    info["n_timeout_raw"] = int((ev["outcome"] == "timeout").sum())
    return ds, info


def add_normalized_features(all_ds: pd.DataFrame) -> pd.DataFrame:
    """Vol-normalize return-scale features by per-event rv20; rv20_rel uses
    the symbol's TRAIN-period median rv20 only (no holdout information)."""
    ds = all_ds.copy()
    rv = ds["rv20"].replace(0, np.nan)
    ds["drop_pct_n"] = ds["drop_pct"] / rv
    ds["drop_speed_n"] = ds["drop_speed"] / rv
    ds["open_to_now_ret_n"] = ds["open_to_now_ret"] / rv
    ds["prev_day_ret_n"] = ds["prev_day_ret"] / rv
    med = (
        ds[ds["et_day"] < TRAIN_END]
        .groupby("symbol")["rv20"]
        .median()
        .rename("rv20_med_train")
    )
    ds = ds.merge(med, left_on="symbol", right_index=True, how="left")
    ds["rv20_rel"] = ds["rv20"] / ds["rv20_med_train"]
    ds["symbol"] = pd.Categorical(ds["symbol"], categories=sorted(ds["symbol"].unique()))
    return ds


# ------------------------------------------------------------------ modelling
def fit_score(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int = SEED
) -> np.ndarray:
    model = make_lgbm(seed)
    model.fit(train[features], train["label"])
    return model.predict_proba(test[features])[:, 1]


def auc_ci(labels: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    auc = roc_auc_score(labels, probs)
    n = len(labels)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        lb, pb = labels[pick], probs[pick]
        if len(np.unique(lb)) < 2:
            continue
        boots.append(roc_auc_score(lb, pb))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(auc), float(lo), float(hi)


# ---------------------------------------------------------- frontier re-check
def frontier_recheck(tsla_hold: pd.DataFrame, probs: np.ndarray, tag: str) -> pd.DataFrame:
    """Re-run the E9 family-A grid on the TSLA holdout with new probabilities,
    settled with the E9 machinery, and diff against E9's stored results.

    Gate matching: the pooled model's probabilities are calibrated to the
    pooled base rate (much lower than TSLA's holdout base rate), so E9's
    absolute gates (0.55..0.70) are not comparable — they would keep almost
    no events. Instead each E9 gate is mapped to the NEW-probability quantile
    threshold that keeps the SAME retention as that gate kept on the old
    (ml_filter_3y OOF) probabilities over the holdout signal pool, making the
    gate x geometry grid retention-matched and apples-to-apples."""
    from research.e9_frontier_search import (
        GATES, GEOMS, TIMEOUTS, BarSet, run_combo, seg_stats, signals_A,
    )

    bars = load_bars(str(TSLA_BARS))
    bs = BarSet(bars)
    scores = tsla_hold[["entry_t"]].copy()
    scores["prob"] = probs
    epos = bars.index.get_indexer(pd.DatetimeIndex(scores["entry_t"]))
    scores["epos"] = epos
    scores = scores[scores["epos"] >= 0].sort_values("epos").reset_index(drop=True)

    # old-prob retention per gate on the holdout signal pool
    old = pd.read_csv(ROOT / "outputs" / "ml_filter_3y" / "per_event_scores.csv")
    old = old[old["prob"].notna()].copy()
    old["et_day"] = pd.to_datetime(old["et_day"]).dt.date
    old_hold = old[old["et_day"] >= HOLDOUT_START]
    p_new = scores["prob"].to_numpy()
    gate_map: dict = {None: None}
    for g in GATES:
        if g is None:
            continue
        retention = float((old_hold["prob"] >= g).mean())
        gate_map[g] = float(np.quantile(p_new, 1.0 - retention)) if retention > 0 else np.inf

    e9 = pd.read_csv(ROOT / "outputs" / "e9_frontier" / "all_combos.csv")
    e9a = e9[(e9["family"] == "A") & (~e9["session_gate"])]

    rows = []
    for g in GATES:
        for tp, sl in GEOMS:
            for to in TIMEOUTS:
                sig = signals_A(bs, scores, gate_map[g], session=False)
                trades = run_combo(bs, sig, tp, sl, to)
                st = seg_stats(trades, "new")
                gate_v = np.nan if g is None else g
                old = e9a[
                    (e9a["gate"].isna() if g is None else (e9a["gate"] == g))
                    & (e9a["tp"] == tp) & (e9a["sl"] == sl) & (e9a["timeout"] == to)
                ]
                row = {
                    "prob_source": tag, "gate": gate_v,
                    "gate_thr_new": np.nan if g is None else gate_map[g],
                    "tp": tp, "sl": sl, "timeout": to,
                    "new_n": st["new_n"], "new_wr": st["new_wr"],
                    "new_avg_bp": st["new_avg_bp"],
                }
                if len(old):
                    o = old.iloc[0]
                    row.update({
                        "e9_holdout_n": o["holdout_n"], "e9_holdout_wr": o["holdout_wr"],
                        "e9_holdout_avg_bp": o["holdout_avg_bp"],
                    })
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- assemble per-symbol datasets ---------------------------
    frames, ev_rows, missing = [], [], []
    for sym in POOL_SYMBOLS + [TARGET]:
        got = build_symbol_dataset(sym)
        if got is None:
            missing.append(sym)
            print(f"[{sym}] MISSING data/events — skipped")
            continue
        ds, info = got
        frames.append(ds)
        ev_rows.append({
            "symbol": sym,
            "n_events_raw": info["n_events_raw"],
            "n_timeout_raw": info["n_timeout_raw"],
            "n_usable": len(ds),
            "n_train": int((ds["et_day"] < TRAIN_END).sum()),
            "n_holdout": int((ds["et_day"] >= HOLDOUT_START).sum()),
            "pos_rate": float(ds["label"].mean()),
            "median_rv20": float(ds["rv20"].median()),
            "first_day": str(ds["et_day"].min()),
            "last_day": str(ds["et_day"].max()),
        })
        print(f"[{sym}] usable events: {len(ds)} (pos rate {ds['label'].mean():.3f})")
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(OUT / "per_symbol_events.csv", index=False)

    all_ds = pd.concat(frames, ignore_index=True)
    all_ds = add_normalized_features(all_ds)
    feat_ok = all_ds[POOLED_FEATURES[:-1]].notna().all(axis=1)  # symbol never NaN
    n_dropped_nan = int((~feat_ok).sum())
    all_ds = all_ds[feat_ok].reset_index(drop=True)

    pool_syms = [s for s in POOL_SYMBOLS if s in set(all_ds["symbol"].astype(str))]
    train_mask = all_ds["et_day"] < TRAIN_END
    hold_mask = all_ds["et_day"] >= HOLDOUT_START

    tsla = all_ds["symbol"] == TARGET
    tsla_train = all_ds[tsla & train_mask]
    tsla_hold = all_ds[tsla & hold_mask].reset_index(drop=True)
    y_hold = tsla_hold["label"].to_numpy()
    print(f"\nTSLA train events: {len(tsla_train)}, holdout events: {len(tsla_hold)} "
          f"(holdout pos rate {y_hold.mean():.3f}); dropped NaN-feature rows: {n_dropped_nan}")

    # ---------------- headline comparisons -----------------------------------
    comp_rows = []

    def record(name, desc, train_df, features, probs=None):
        if probs is None:
            probs = fit_score(train_df, tsla_hold, features)
        auc, lo, hi = auc_ci(y_hold, probs, rng)
        comp_rows.append({
            "model": name, "description": desc, "n_train": len(train_df),
            "n_test": len(tsla_hold), "auc": auc, "ci_lo": lo, "ci_hi": hi,
        })
        print(f"  {name:<28s} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]  (train n={len(train_df)})")
        return probs

    print("\n== TSLA holdout (2025-10 .. 2026-07) AUC ==")
    loo_train = all_ds[(all_ds["symbol"] != TARGET) & train_mask]
    probs_loo = record(
        "a_leave_tsla_out",
        f"train {len(pool_syms)} pool symbols only",
        loo_train, POOLED_FEATURES)

    record(
        "b_tsla_self_raw_feats",
        "TSLA train segment, original ml_filter features",
        tsla_train, FEATURES)
    record(
        "b2_tsla_self_norm_feats",
        "TSLA train segment, normalized features",
        tsla_train, POOLED_FEATURES)

    pooled_train = all_ds[train_mask]
    probs_pooled = record(
        "c_pooled_incl_tsla",
        f"train {len(pool_syms)} pool symbols + TSLA train segment",
        pooled_train, POOLED_FEATURES)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "auc_comparison.csv", index=False)

    # ---------------- per-symbol leave-one-out table -------------------------
    print("\n== per-symbol leave-one-out (train on the rest -> own holdout) ==")
    loo_rows = []
    for sym in pool_syms + [TARGET]:
        s_mask = all_ds["symbol"] == sym
        te = all_ds[s_mask & hold_mask]
        if len(te) < 30 or te["label"].nunique() < 2:
            loo_rows.append({"symbol": sym, "n_test": len(te), "note": "too few/one-class"})
            continue
        tr_others = all_ds[(~s_mask) & train_mask]
        p_loo = fit_score(tr_others, te, POOLED_FEATURES)
        auc_loo = roc_auc_score(te["label"], p_loo)
        tr_self = all_ds[s_mask & train_mask]
        if len(tr_self) >= 50 and tr_self["label"].nunique() == 2:
            p_self = fit_score(tr_self, te, POOLED_FEATURES)
            auc_self = roc_auc_score(te["label"], p_self)
        else:
            auc_self = np.nan
        loo_rows.append({
            "symbol": sym, "n_test": len(te), "test_pos_rate": float(te["label"].mean()),
            "auc_loo": float(auc_loo), "auc_self": float(auc_self), "note": "",
        })
        print(f"  {sym:<6s} n_test={len(te):>4d}  LOO {auc_loo:.3f}  self "
              f"{'n/a' if not np.isfinite(auc_self) else f'{auc_self:.3f}'}")
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(OUT / "per_symbol_loo_auc.csv", index=False)

    # ---------------- conditional frontier re-check --------------------------
    best_name, best_probs, best_auc = max(
        [("a_leave_tsla_out", probs_loo,
          float(comp_df.loc[comp_df["model"] == "a_leave_tsla_out", "auc"].iloc[0])),
         ("c_pooled_incl_tsla", probs_pooled,
          float(comp_df.loc[comp_df["model"] == "c_pooled_incl_tsla", "auc"].iloc[0]))],
        key=lambda x: x[2])
    frontier_df = None
    if best_auc >= AUC_GATE_FOR_FRONTIER:
        print(f"\n{best_name} AUC {best_auc:.3f} >= {AUC_GATE_FOR_FRONTIER} — "
              "re-running E9 family-A grid on TSLA holdout with new probabilities")
        frontier_df = frontier_recheck(tsla_hold, best_probs, best_name)
        frontier_df.to_csv(OUT / "frontier_shift.csv", index=False)
    else:
        print(f"\nbest pooled AUC {best_auc:.3f} < {AUC_GATE_FOR_FRONTIER} — frontier step skipped")

    # ---------------- summary -------------------------------------------------
    auc_self_raw = float(comp_df.loc[comp_df["model"] == "b_tsla_self_raw_feats", "auc"].iloc[0])
    auc_loo_t = float(comp_df.loc[comp_df["model"] == "a_leave_tsla_out", "auc"].iloc[0])
    auc_pool = float(comp_df.loc[comp_df["model"] == "c_pooled_incl_tsla", "auc"].iloc[0])
    ok_loo = loo_df[loo_df["note"] == ""] if len(loo_df) else pd.DataFrame()
    n_transfer = int((ok_loo["auc_loo"] > 0.55).sum()) if len(ok_loo) else 0

    L = []
    L.append("E8 — GBDT 跨标的联合训练（leave-TSLA-out 主检验）")
    L.append(f"协议：训练段 et_day < {TRAIN_END}（含 1 天 embargo），留出段 et_day >= {HOLDOUT_START}；"
             "标签 = 三重障碍 +0.5%/-1.0%/24bar/日末强平（全标的一致）；"
             "特征 = ml_common 特征经 rv20 波动率标准化 + rv20_rel + symbol 类别。")
    L.append(f"数据覆盖：{len(pool_syms)} 只池内标的 + TSLA；缺失跳过: {missing if missing else '无'}")
    L.append(f"合计可用事件 {len(all_ds)}（训练段 {int(train_mask.sum())}, 留出段 {int(hold_mask.sum())}）")
    L.append("")
    L.append("== TSLA 留出段 AUC 对比（bootstrap 95% CI）==")
    L.append(comp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    L.append("")
    L.append("== 逐标的 leave-one-out 泛化表 ==")
    L.append(loo_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    L.append("")
    L.append("== 判读 ==")
    ci_a = comp_df.loc[comp_df["model"] == "a_leave_tsla_out", ["ci_lo", "ci_hi"]].iloc[0]
    transfer_holds = auc_loo_t > 0.55
    L.append(f"1. 迁移检验：leave-TSLA-out AUC = {auc_loo_t:.3f} [{ci_a['ci_lo']:.3f},{ci_a['ci_hi']:.3f}] "
             f"vs TSLA 自训 {auc_self_raw:.3f}。"
             + ("跨标的模型对 TSLA 有排序能力（>0.55），V 反弹结构在标的间可迁移。"
                if transfer_holds else "跨标的模型在 TSLA 上接近随机，迁移不成立。"))
    gain = auc_pool - auc_self_raw
    L.append(f"2. 数据量假设：pooled（含 TSLA）AUC = {auc_pool:.3f}，比自训 "
             f"{'高' if gain > 0 else '低'} {abs(gain):.3f}。"
             + (f"未达预登记门槛 0.63 —— 24 倍训练数据没有带来实质提升，"
                "『单标的数据量是 0.60 瓶颈』假设被证伪：瓶颈更可能在特征信息量/信噪比本身。"
                if auc_pool < AUC_GATE_FOR_FRONTIER else
                "达到预登记门槛 0.63 —— 数据量假设获得支持，见 frontier_shift.csv。"))
    if len(ok_loo):
        L.append(f"3. 泛化普遍性：{n_transfer}/{len(ok_loo)} 只标的 LOO AUC > 0.55"
                 f"（LOO 均值 {ok_loo['auc_loo'].mean():.3f}，自训均值 "
                 f"{ok_loo['auc_self'].mean():.3f}）。")
    if frontier_df is not None:
        L.append("4. 前沿复检：见 frontier_shift.csv（新概率 vs E9 家族 A 存档）。")
    else:
        L.append(f"4. 前沿复检：跳过（最好 AUC {best_auc:.3f} < {AUC_GATE_FOR_FRONTIER}，"
                 "按预登记规则不做几何重扫，避免多重比较通胀）。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
