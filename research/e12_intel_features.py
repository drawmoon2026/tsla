#!/usr/bin/env python
"""E12 — intel features (short interest + Musk post density) into the E8 pooled GBDT.

Hypothesis (docs/strategy-lab.md E12): short_interest_change (all symbols) and
Musk post density (TSLA only, NaN elsewhere) added to the E8 feature pool lift
the pooled/leave-TSLA-out AUC (baseline 0.642).

Protocol — identical to E8 (research/e8_pooled_gbdt.py), fixed before results:
- Same events, bars, triple-barrier labels, vol-normalized features, LightGBM
  config, split (train et_day < 2025-09-30 with 1-day embargo, TSLA holdout
  et_day >= 2025-10-01), same NaN row filter (base features only, so the event
  sample is row-identical to E8's).
- New features (research/ml_common.py, INTEL_FEATURES, anti-lookahead: only
  reports PUBLISHED by the confirm-bar close are visible; publication approx =
  settlement + 9 bdays 16:00 ET, same convention as N2):
    si_chg_recent, si_days_since  — all symbols (data/intel/finra_short.csv for
                                    TSLA, data/intel/pool_short/{SYM}.csv else)
    musk_daily_posts              — TSLA events on archive-covered ET days only
                                    (< 2025-05-08); NaN everywhere else,
                                    LightGBM handles NaN natively.
- Models compared on the TSLA holdout (bootstrap 95% CI, N=1000):
    base  : POOLED_FEATURES               (leave-TSLA-out + pooled)
    +si   : + si_chg_recent, si_days_since (leave-TSLA-out + pooled)
    +si+musk : + musk_daily_posts          (leave-TSLA-out + pooled)
- Pre-registered verdict: +si leave-TSLA-out AUC - base leave-TSLA-out AUC
  >= 0.005 AND the two 95% CIs not fully overlapping (operationalised: the +si
  CI is not contained in the base CI nor vice versa; paired-bootstrap delta CI
  reported alongside for transparency). Pass -> features active; fail ->
  features stay in code marked inactive.
- If pass: replay the frozen E8-A config (retention-matched gate 0.70 ~= top
  10% x tp 0.5% / sl 2% / timeout 48) on the holdout with the new
  probabilities and diff against outputs/e8_pooled/frontier_shift.csv.

Usage:  .venv/bin/python research/e12_intel_features.py
Outputs: outputs/e12_intel_features/{auc_comparison.csv,
         feature_importance.txt, summary.txt, e8a_replay.csv (conditional)}
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

from sklearn.metrics import roc_auc_score  # noqa: E402

from research.e8_pooled_gbdt import (  # noqa: E402
    HOLDOUT_START, N_BOOT, POOL_SYMBOLS, POOLED_FEATURES, TARGET, TRAIN_END,
    TSLA_BARS, build_symbol_dataset,  add_normalized_features, auc_ci,
)
from research.ml_common import (  # noqa: E402
    INTEL_FEATURES, SEED, add_intel_features, load_musk_post_times,
    load_short_interest_events, make_lgbm,
)

INTEL_DIR = ROOT / "data" / "intel"
POOL_SHORT_DIR = INTEL_DIR / "pool_short"
OUT = ROOT / "outputs" / "e12_intel_features"

SI_FEATS = ["si_chg_recent", "si_days_since"]
MUSK_FEAT = ["musk_daily_posts"]

FEATURE_SETS = {
    "base": POOLED_FEATURES,
    "si": POOLED_FEATURES + SI_FEATS,
    "si_musk": POOLED_FEATURES + SI_FEATS + MUSK_FEAT,
}

# pre-registered verdict thresholds
MIN_AUC_GAIN = 0.005

# frozen E8-A config (E9 gate 0.70 retention-matched, tp0.5%/sl2%/to48)
E8A_GATE_OLD = 0.70
E8A_TP, E8A_SL, E8A_TO = 0.005, 0.020, 48


def fit_model(train: pd.DataFrame, features: list[str], seed: int = SEED):
    model = make_lgbm(seed)
    model.fit(train[features], train["label"])
    return model


def paired_delta_ci(labels: np.ndarray, p_new: np.ndarray, p_old: np.ndarray,
                    rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap CI of AUC(p_new) - AUC(p_old) on SHARED resamples."""
    delta = roc_auc_score(labels, p_new) - roc_auc_score(labels, p_old)
    n = len(labels)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        lb = labels[pick]
        if len(np.unique(lb)) < 2:
            continue
        boots.append(roc_auc_score(lb, p_new[pick]) - roc_auc_score(lb, p_old[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(delta), float(lo), float(hi)


def e8a_replay(tsla_hold: pd.DataFrame, probs: np.ndarray, tag: str) -> pd.DataFrame:
    """Replay the frozen E8-A config with new probabilities (retention-matched
    gate, same machinery/costs as E8's frontier_recheck) and diff vs the
    stored E8 baseline row in outputs/e8_pooled/frontier_shift.csv."""
    from research.e9_frontier_search import BarSet, run_combo, seg_stats, signals_A
    from src.common.data_io import load_bars

    bars = load_bars(str(TSLA_BARS))
    bs = BarSet(bars)
    scores = tsla_hold[["entry_t"]].copy()
    scores["prob"] = probs
    epos = bars.index.get_indexer(pd.DatetimeIndex(scores["entry_t"]))
    scores["epos"] = epos
    scores = scores[scores["epos"] >= 0].sort_values("epos").reset_index(drop=True)

    old = pd.read_csv(ROOT / "outputs" / "ml_filter_3y" / "per_event_scores.csv")
    old = old[old["prob"].notna()].copy()
    old["et_day"] = pd.to_datetime(old["et_day"]).dt.date
    old_hold = old[old["et_day"] >= HOLDOUT_START]
    retention = float((old_hold["prob"] >= E8A_GATE_OLD).mean())
    gate_thr = float(np.quantile(scores["prob"].to_numpy(), 1.0 - retention))

    sig = signals_A(bs, scores, gate_thr, session=False)
    trades = run_combo(bs, sig, E8A_TP, E8A_SL, E8A_TO)
    st = seg_stats(trades, "new")

    e8 = pd.read_csv(ROOT / "outputs" / "e8_pooled" / "frontier_shift.csv")
    ref = e8[(e8["gate"] == E8A_GATE_OLD) & (e8["tp"] == E8A_TP)
             & (e8["sl"] == E8A_SL) & (e8["timeout"] == E8A_TO)].iloc[0]
    return pd.DataFrame([{
        "prob_source": tag, "gate_old": E8A_GATE_OLD, "gate_thr_new": gate_thr,
        "retention": retention, "tp": E8A_TP, "sl": E8A_SL, "timeout": E8A_TO,
        "new_n": st["new_n"], "new_wr": st["new_wr"],
        "new_avg_bp": st["new_avg_bp"], "new_total": st["new_total"],
        "new_mdd": st["new_mdd"],
        "e8_n": ref["new_n"], "e8_wr": ref["new_wr"], "e8_avg_bp": ref["new_avg_bp"],
    }])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    musk_times = load_musk_post_times(INTEL_DIR / "musk_tweets.csv")

    # ---------------- per-symbol datasets + intel features -------------------
    frames, missing_si = [], []
    for sym in POOL_SYMBOLS + [TARGET]:
        got = build_symbol_dataset(sym)
        if got is None:
            print(f"[{sym}] MISSING bars/events — skipped")
            continue
        ds, _ = got
        si_csv = (INTEL_DIR / "finra_short.csv" if sym == TARGET
                  else POOL_SHORT_DIR / f"{sym}.csv")
        si = load_short_interest_events(si_csv) if si_csv.exists() else None
        if si is None:
            missing_si.append(sym)
        ds = add_intel_features(
            ds, si=si, musk_times=musk_times if sym == TARGET else None)
        frames.append(ds)
        print(f"[{sym}] events {len(ds)}, si rows {0 if si is None else len(si)}, "
              f"si_chg_recent NaN rate {ds['si_chg_recent'].isna().mean():.3f}")

    all_ds = pd.concat(frames, ignore_index=True)
    all_ds = add_normalized_features(all_ds)
    # identical row filter to E8: base features only -> identical event sample
    feat_ok = all_ds[POOLED_FEATURES[:-1]].notna().all(axis=1)
    all_ds = all_ds[feat_ok].reset_index(drop=True)

    train_mask = all_ds["et_day"] < TRAIN_END
    hold_mask = all_ds["et_day"] >= HOLDOUT_START
    tsla = all_ds["symbol"] == TARGET
    tsla_hold = all_ds[tsla & hold_mask].reset_index(drop=True)
    y_hold = tsla_hold["label"].to_numpy()

    loo_train = all_ds[(~tsla) & train_mask]
    pooled_train = all_ds[train_mask]
    tsla_train_intel = all_ds[tsla & train_mask]

    n_musk_train = int(tsla_train_intel["musk_daily_posts"].notna().sum())
    n_musk_hold = int(tsla_hold["musk_daily_posts"].notna().sum())
    print(f"\nTSLA train {len(tsla_train_intel)} / holdout {len(tsla_hold)} events "
          f"(holdout pos rate {y_hold.mean():.3f})")
    print(f"musk_daily_posts non-NaN: train {n_musk_train}, holdout {n_musk_hold} "
          "(archive ends 2025-05-08 -> holdout all-NaN by construction)")

    # ---------------- AUC comparison ----------------------------------------
    comp_rows, probs_store, models_store = [], {}, {}
    for fs_name, feats in FEATURE_SETS.items():
        for scheme, train_df in (("leave_tsla_out", loo_train),
                                 ("pooled_incl_tsla", pooled_train)):
            model = fit_model(train_df, feats)
            probs = model.predict_proba(tsla_hold[feats])[:, 1]
            auc, lo, hi = auc_ci(y_hold, probs, rng)
            key = f"{fs_name}_{scheme}"
            probs_store[key] = probs
            models_store[key] = (model, feats)
            comp_rows.append({
                "features": fs_name, "scheme": scheme, "n_feats": len(feats),
                "n_train": len(train_df), "n_test": len(tsla_hold),
                "auc": auc, "ci_lo": lo, "ci_hi": hi,
            })
            print(f"  {key:<30s} AUC {auc:.4f}  [{lo:.3f}, {hi:.3f}]")
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "auc_comparison.csv", index=False)

    def get(fs, sc, col):
        r = comp_df[(comp_df["features"] == fs) & (comp_df["scheme"] == sc)]
        return float(r[col].iloc[0])

    # ---------------- paired delta bootstrap (supplementary) -----------------
    d_si, d_si_lo, d_si_hi = paired_delta_ci(
        y_hold, probs_store["si_leave_tsla_out"],
        probs_store["base_leave_tsla_out"], rng)
    d_sm, d_sm_lo, d_sm_hi = paired_delta_ci(
        y_hold, probs_store["si_musk_leave_tsla_out"],
        probs_store["base_leave_tsla_out"], rng)

    # ---------------- verdict (pre-registered) --------------------------------
    auc_base, lo_base, hi_base = (get("base", "leave_tsla_out", c)
                                  for c in ("auc", "ci_lo", "ci_hi"))
    auc_si, lo_si, hi_si = (get("si", "leave_tsla_out", c)
                            for c in ("auc", "ci_lo", "ci_hi"))
    gain = auc_si - auc_base
    contained = (lo_si >= lo_base and hi_si <= hi_base) or \
                (lo_base >= lo_si and hi_base <= hi_si)
    ci_ok = not contained
    valid = (gain >= MIN_AUC_GAIN) and ci_ok

    # ---------------- feature importance -------------------------------------
    fi_lines = ["E12 feature importance (LightGBM gain), TSLA-holdout models", ""]
    for key in ("si_leave_tsla_out", "si_musk_leave_tsla_out",
                "si_musk_pooled_incl_tsla"):
        model, feats = models_store[key]
        imp = model.booster_.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1]
        total = imp.sum() or 1.0
        fi_lines.append(f"== {key} ==")
        for rank, j in enumerate(order, 1):
            mark = "  <-- intel" if feats[j] in INTEL_FEATURES else ""
            fi_lines.append(f"  {rank:>2d}. {feats[j]:<20s} {imp[j]:>14.1f} "
                            f"({imp[j] / total:6.1%}){mark}")
        fi_lines.append("")
    (OUT / "feature_importance.txt").write_text("\n".join(fi_lines), encoding="utf-8")

    # ---------------- conditional E8-A replay ---------------------------------
    replay_df = None
    if valid:
        best_key = max(("si_leave_tsla_out", "si_pooled_incl_tsla"),
                       key=lambda k: roc_auc_score(y_hold, probs_store[k]))
        replay_df = e8a_replay(tsla_hold, probs_store[best_key], best_key)
        replay_df.to_csv(OUT / "e8a_replay.csv", index=False)

    # ---------------- summary -------------------------------------------------
    L = []
    L.append("E12 — 情报特征并入 E8 pooled GBDT（判读）")
    L.append(f"协议：与 E8 完全一致（训练 et_day < {TRAIN_END} 含 1 天 embargo / "
             f"TSLA 留出 et_day >= {HOLDOUT_START}；同事件样本、同标签、同模型配置）。")
    L.append("新特征（ml_common.INTEL_FEATURES，防前视：只用确认 bar 收盘前『已发布』的报告；"
             "发布时刻近似 = 结算日 + 9 交易日 16:00 ET，同 N2 口径）：")
    L.append("  si_chg_recent / si_days_since —— 全标的（TSLA: data/intel/finra_short.csv；"
             "池标的: data/intel/pool_short/{SYM}.csv，FINRA API 现拉）")
    L.append("  musk_daily_posts —— 仅 TSLA 且归档覆盖日（< 2025-05-08）；其余 NaN")
    if missing_si:
        L.append(f"  ！缺空头数据的标的（si 特征 NaN）：{missing_si}")
    L.append(f"TSLA 留出段 {len(tsla_hold)} 事件；musk_daily_posts 非 NaN：训练段 "
             f"{n_musk_train}，留出段 {n_musk_hold}（归档止于 2025-05-08，留出段全 NaN——"
             "该特征只影响训练拟合，不给留出推断提供区分信息，其真正检验在前向）")
    L.append("")
    L.append("== TSLA 留出段 AUC 对比（bootstrap 95% CI, N=1000）==")
    L.append(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    L.append("")
    L.append("== 配对 bootstrap ΔAUC（leave-TSLA-out，同一重采样）==")
    L.append(f"  +si    - base : {d_si:+.4f} [{d_si_lo:+.4f}, {d_si_hi:+.4f}]")
    L.append(f"  +si+musk - base : {d_sm:+.4f} [{d_sm_lo:+.4f}, {d_sm_hi:+.4f}]")
    L.append("")
    L.append("== 预登记判决 ==")
    L.append(f"条件：+si leave-TSLA-out AUC 提升 >= {MIN_AUC_GAIN} 且 95% CI 不完全重叠"
             "（操作化：两 CI 不互相包含；配对 ΔCI 附上供参考）")
    L.append(f"  基线 AUC {auc_base:.4f} [{lo_base:.3f},{hi_base:.3f}]，"
             f"+si AUC {auc_si:.4f} [{lo_si:.3f},{hi_si:.3f}]，提升 {gain:+.4f}")
    L.append(f"  提升 >= {MIN_AUC_GAIN}: {'是' if gain >= MIN_AUC_GAIN else '否'}；"
             f"CI 不互相包含: {'是' if ci_ok else '否'}")
    if valid:
        L.append("  判决：✅ 有效 —— si 特征并入 E8 特征池（active）。")
    else:
        L.append("  判决：❌ 无效 —— si 特征保留在 ml_common（INTEL_FEATURES）但标注 "
                 "inactive，不并入生产特征集。")
    L.append("")
    if replay_df is not None:
        r = replay_df.iloc[0]
        L.append("== E8-A 配置重放（gate top10% 保留率匹配 × tp0.5%/sl2%/to48，留出段）==")
        L.append(f"  概率来源 {r['prob_source']}（新门槛 {r['gate_thr_new']:.4f}，"
                 f"保留率 {r['retention']:.3f}）")
        L.append(f"  新概率: n={int(r['new_n'])}, WR {r['new_wr']:.3f}, "
                 f"期望 {r['new_avg_bp']:+.1f}bp, total {r['new_total']:+.3%}, "
                 f"MDD {r['new_mdd']:.3%}")
        L.append(f"  E8 基线: n={int(r['e8_n'])}, WR {r['e8_wr']:.3f}, "
                 f"期望 {r['e8_avg_bp']:+.1f}bp")
        L.append("  注意：留出段事后配置，与 E8 同样只作方向参考，不构成上钱依据。")
    else:
        L.append("== E8-A 配置重放：跳过（判决无效，按预登记规则不做，避免多重比较通胀）==")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    text = "\n".join(L)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
