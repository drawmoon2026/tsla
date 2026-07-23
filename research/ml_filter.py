#!/usr/bin/env python
"""GBDT rebound-probability filter for 5m V-reversal signals.

Pipeline-validation only (~247 physical events from 60 days of 5m bars).
See docs/review-2026-07-23/ai-algo-survey.md section on GBDT filters +
purged walk-forward for the methodology rationale.

Anti-lookahead contract
-----------------------
- Bars are stamped by bar-START time (yfinance convention, UTC tz-aware).
- `trough_confirm_t` is the start time of the bar `lf` bars after the trough
  (src/v_stats.py). The trough is identifiable only once that bar CLOSES,
  i.e. at confirm_t + 5m. Therefore:
    * actionable entry = Open of the FIRST bar AFTER the confirm bar;
    * features may use anything up to and including the confirm bar's close;
    * anything after the confirm bar's close feeds the LABEL only.
- Validation: purged walk-forward grouped by ET trading day, expanding
  window, 5 test folds, 1-trading-day embargo between train end and test
  start (label horizon is <= 2h intraday, so 1 day fully purges overlap).

Run from project root:  .venv/bin/python research/ml_filter.py
    [--events_csv PATH] [--bars_csv PATH] [--out_dir DIR] [--n_folds N]
Outputs: {out_dir}/{metrics.txt, per_event_scores.csv,
         feature_importance.txt}

Survivorship-bias note
----------------------
src/v_stats.py now also emits triggers whose rebound timed out
(outcome="timeout"). They are included in the training set — their label is
still computed by the SAME triple-barrier logic (never taken from `outcome`).
`outcome` itself is resolved after entry, i.e. future information, and MUST
NEVER be used as a feature; it is excluded explicitly below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, intraday_returns, load_bars  # noqa: E402

# ---------------------------------------------------------------- parameters
DEFAULT_EVENTS_CSV = ROOT / "v_events_5m_unique.csv"
DEFAULT_BARS_CSV = ROOT / "data" / "TSLA_5m_60d.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "ml_filter"

LABEL_TP = 0.005      # +y: take-profit touch (via High) -> label 1
LABEL_SL = 0.010      # -x: stop touch (via Low) -> label 0
LABEL_HORIZON = 24    # bars (24 x 5m = 2h); timeout -> label 0
DEFAULT_N_TEST_FOLDS = 5   # day-grouped expanding walk-forward
EMBARGO_DAYS = 1      # trading days purged between train end and test start
THRESHOLDS = (0.5, 0.6, 0.7)
SEED = 42

FEATURES = [
    "drop_pct",          # peak->trough drawdown (known at confirm)
    "drop_speed",        # drop_pct / bars peak->trough
    "vol_ratio_trough",  # trough-bar volume / mean volume of prior 20 bars
    "et_hour",           # ET hour of confirm time (fractional)
    "open_to_now_ret",   # day open -> confirm-bar close return
    "prev_day_ret",      # previous trading day close-to-close return
    "rv20",              # std of intraday 5m returns, 20 bars ending at confirm
    "bars_since_open",   # confirm bar's position within the trading day
]

# `outcome` (rebounded/timeout) is resolved AFTER entry -> future info.
# It must never leak into the model input.
FORBIDDEN_FEATURES = {"outcome", "rebound_t", "rebound_p", "rebound_pct", "bars_to_rebound"}
assert not (set(FEATURES) & FORBIDDEN_FEATURES), "future-information column in FEATURES"


def warning_banner(n_events: int, has_outcome: bool) -> str:
    surv = (
        "2. 事件表已包含反弹超时 (outcome=timeout) 的触发（幸存者偏差已修复）。\n"
        "   标签仍由三重障碍现算，outcome 不作标签、不作特征（未来信息）。"
        if has_outcome
        else "2. 事件表本身存在幸存者偏差：该版本 v_stats.py 只记录最终在 T 根 bar\n"
        "   内反弹 >= Y 的事件（未反弹的触发被丢弃），标签正例率系统性偏高，\n"
        "   precision 数字不可外推。"
    )
    return f"""\
================================================================================
!!  样本量警告 / RESEARCH ONLY  !!
--------------------------------------------------------------------------------
1. 当前共 {n_events} 个 5m 物理 V 形触发。本管线是研究性验证，
   不是可交易信号。
{surv}
3. 单折 AUC 的抽样噪声可能很大，逐折方差本身是主要读数之一。
================================================================================
"""


# ---------------------------------------------------------------- data prep
def load_events(events_csv: Path) -> pd.DataFrame:
    ev = pd.read_csv(events_csv)
    for c in ("peak_t", "trough_t", "trough_confirm_t", "rebound_t"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    if "outcome" not in ev.columns:
        # legacy table (pre-survivorship-fix): every recorded event rebounded
        ev["outcome"] = "rebounded"
    return ev.sort_values("trough_confirm_t").reset_index(drop=True)


def build_dataset(ev: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return one row per usable event: features + label + bookkeeping."""
    idx = bars.index
    n = len(bars)
    opens = bars["Open"].to_numpy()
    highs = bars["High"].to_numpy()
    lows = bars["Low"].to_numpy()
    closes = bars["Close"].to_numpy()

    et_index = idx.tz_convert(ET)
    et_dates = np.asarray(et_index.date)
    # position of each day's first bar, per bar
    day_change = np.r_[True, et_dates[1:] != et_dates[:-1]]
    day_first_pos = np.maximum.accumulate(np.where(day_change, np.arange(n), -1))

    # per-day close series for prev-day return (close-to-close)
    daily_close = bars["Close"].groupby(et_dates).last()
    daily_ret = daily_close.pct_change()  # ret of day d vs d-1, known at d's close
    days_sorted = list(daily_close.index)
    day_pos = {d: i for i, d in enumerate(days_sorted)}

    # volume ratio denominator: mean volume of the 20 bars BEFORE each bar
    vol_ma20_prior = (
        bars["Volume"].shift(1).rolling(20, min_periods=10).mean().to_numpy()
    )
    # realized vol: std of overnight-gap-free 5m returns, 20 bars ending at pos
    iret = intraday_returns(bars["Close"])
    rv20 = iret.rolling(20, min_periods=10).std().to_numpy()

    confirm_pos = idx.get_indexer(ev["trough_confirm_t"])
    trough_pos = idx.get_indexer(ev["trough_t"])
    if (confirm_pos < 0).any() or (trough_pos < 0).any():
        bad = int((confirm_pos < 0).sum() + (trough_pos < 0).sum())
        raise RuntimeError(f"{bad} event timestamps not found in the bar index")

    rows: list[dict] = []
    drops = {"entry_beyond_data": 0, "entry_next_day": 0, "label_unresolved": 0}
    n_ambiguous = 0

    for i, e in ev.iterrows():
        cpos = int(confirm_pos[i])
        tpos = int(trough_pos[i])
        epos = cpos + 1
        if epos >= n:
            drops["entry_beyond_data"] += 1
            continue
        if et_dates[epos] != et_dates[cpos]:
            # confirm bar was the day's last bar; "next bar open" would be an
            # overnight-gap entry on the following day -> different trade, skip
            drops["entry_next_day"] += 1
            continue

        # ---------------- label (uses ONLY post-confirm information) --------
        entry = opens[epos]
        tp_px = entry * (1.0 + LABEL_TP)
        sl_px = entry * (1.0 - LABEL_SL)
        label: int | None = None
        last = min(epos + LABEL_HORIZON - 1, n - 1)
        for k in range(epos, last + 1):
            if et_dates[k] != et_dates[epos]:
                label = 0  # day ended first -> forced flat -> timeout
                break
            hit_tp = highs[k] >= tp_px
            hit_sl = lows[k] <= sl_px
            if hit_tp and hit_sl:
                # both barriers inside one 5m bar: order unknowable ->
                # conservative label 0 (assume stop first)
                n_ambiguous += 1
                label = 0
                break
            if hit_sl:
                label = 0
                break
            if hit_tp:
                label = 1
                break
        if label is None:
            if last == n - 1 and et_dates[last] == et_dates[epos]:
                # data (not the day) ran out mid-horizon: outcome unknown
                drops["label_unresolved"] += 1
                continue
            label = 0  # full horizon elapsed without a touch -> timeout

        # ---------------- features (info <= confirm-bar close only) ---------
        d = day_pos[et_dates[cpos]]
        prev_day_ret = float(daily_ret.iloc[d - 1]) if d >= 1 else np.nan

        rows.append(
            {
                "peak_t": e["peak_t"],
                "trough_t": e["trough_t"],
                "trough_confirm_t": e["trough_confirm_t"],
                "entry_t": idx[epos],
                "entry_price": float(entry),
                "et_day": et_dates[cpos],
                "outcome": e["outcome"],  # bookkeeping ONLY — never a feature
                "label": int(label),
                "drop_pct": float(e["drop_pct"]),
                "drop_speed": float(e["drop_pct"]) / max(int(e["bars_to_trough"]), 1),
                "vol_ratio_trough": float(
                    bars["Volume"].iloc[tpos] / vol_ma20_prior[tpos]
                )
                if np.isfinite(vol_ma20_prior[tpos]) and vol_ma20_prior[tpos] > 0
                else np.nan,
                "et_hour": et_index[cpos].hour + et_index[cpos].minute / 60.0,
                "open_to_now_ret": float(
                    closes[cpos] / opens[day_first_pos[cpos]] - 1.0
                ),
                "prev_day_ret": prev_day_ret,
                "rv20": float(rv20[cpos]),
                "bars_since_open": int(cpos - day_first_pos[cpos]),
            }
        )

    info = {"drops": drops, "n_ambiguous_bars": n_ambiguous}
    return pd.DataFrame(rows), info


# ---------------------------------------------------------- walk-forward CV
def day_blocks(days: list, n_folds: int) -> list[np.ndarray]:
    """Split the sorted unique trading days into n_folds+1 contiguous blocks.
    Block 0 is train-only seed; blocks 1..n_folds are the test folds."""
    return np.array_split(np.asarray(days, dtype=object), n_folds + 1)


def run_walk_forward(ds: pd.DataFrame, n_folds: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    days = sorted(ds["et_day"].unique())
    blocks = day_blocks(days, n_folds)

    ds = ds.copy()
    ds["fold"] = -1  # -1 = train-only seed block, never scored out-of-fold
    ds["prob"] = np.nan

    fold_rows = []
    importances = []
    fold_notes = []

    for f in range(1, n_folds + 1):
        test_days = set(blocks[f])
        train_days_sorted = [d for b in blocks[:f] for d in b]
        embargoed = train_days_sorted[-EMBARGO_DAYS:] if EMBARGO_DAYS else []
        train_days = set(train_days_sorted) - set(embargoed)

        tr = ds[ds["et_day"].isin(train_days)]
        te_mask = ds["et_day"].isin(test_days)
        te = ds[te_mask]
        ds.loc[te_mask, "fold"] = f

        note = ""
        if len(te) == 0:
            fold_notes.append(f"fold {f}: no test events, skipped")
            continue
        if tr["label"].nunique() < 2:
            fold_notes.append(
                f"fold {f}: train set has a single class "
                f"({len(tr)} events), skipped"
            )
            continue

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            deterministic=True,
            verbosity=-1,
        )
        model.fit(tr[FEATURES], tr["label"])
        prob = model.predict_proba(te[FEATURES])[:, 1]
        ds.loc[te_mask, "prob"] = prob

        if te["label"].nunique() < 2:
            auc = np.nan
            note = "test set single-class, AUC undefined"
        else:
            auc = roc_auc_score(te["label"], prob)

        imp = model.booster_.feature_importance(importance_type="gain")
        importances.append(imp)
        fold_rows.append(
            {
                "fold": f,
                "train_n": len(tr),
                "train_pos_rate": tr["label"].mean(),
                "test_n": len(te),
                "test_pos_rate": te["label"].mean(),
                "test_days": f"{min(test_days)}..{max(test_days)}",
                "embargoed_days": ", ".join(str(d) for d in embargoed),
                "auc": auc,
                "note": note,
            }
        )

    oof = ds[ds["prob"].notna()]
    pooled_auc = np.nan
    if len(oof) and oof["label"].nunique() == 2:
        pooled_auc = roc_auc_score(oof["label"], oof["prob"])

    imp_avg = np.mean(importances, axis=0) if importances else np.zeros(len(FEATURES))
    return ds, fold_rows, fold_notes, pooled_auc, imp_avg


# ------------------------------------------------------------------ reports
def threshold_table(oof: pd.DataFrame) -> list[dict]:
    base = oof["label"].mean()
    out = [
        {
            "threshold": "none (baseline: pass all signals)",
            "kept": len(oof),
            "retention": 1.0,
            "precision": base,
        }
    ]
    for thr in THRESHOLDS:
        kept = oof[oof["prob"] >= thr]
        out.append(
            {
                "threshold": f"prob >= {thr}",
                "kept": len(kept),
                "retention": len(kept) / len(oof) if len(oof) else np.nan,
                "precision": kept["label"].mean() if len(kept) else np.nan,
            }
        )
    return out


def verdict(pooled_auc: float, fold_rows: list[dict]) -> str:
    aucs = [r["auc"] for r in fold_rows if np.isfinite(r["auc"])]
    spread = (max(aucs) - min(aucs)) if len(aucs) >= 2 else np.nan
    if not np.isfinite(pooled_auc):
        return "汇总 AUC 无法计算（OOF 集单一类别）——无任何可用结论。"
    if pooled_auc < 0.55:
        return (
            f"汇总 OOF AUC = {pooled_auc:.3f}，接近 0.5：在当前样本上模型没有"
            "学到可辨别的信号。这不意外——样本太少，且逐折方差"
            + (f"高达 {spread:.2f}" if np.isfinite(spread) else "无法评估")
            + "。结论：流程可用，信号证据不足，等多年数据重跑。"
        )
    if pooled_auc < 0.65:
        return (
            f"汇总 OOF AUC = {pooled_auc:.3f}：有微弱的信号迹象，但在当前"
            "样本量、逐折方差"
            + (f" {spread:.2f}" if np.isfinite(spread) else "未知")
            + " 的条件下完全可能是噪声。不构成任何交易依据，等多年数据重跑。"
        )
    return (
        f"汇总 OOF AUC = {pooled_auc:.3f}：数字偏高，但在这么小的样本上高 AUC"
        "更应引起怀疑（偶然性/残余泄漏），而不是兴奋。务必用多年数据重跑验证。"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GBDT rebound-probability filter for 5m V-reversal signals.")
    p.add_argument("--events_csv", default=str(DEFAULT_EVENTS_CSV), help="v_events_5m_unique.csv produced by src/v_stats.py")
    p.add_argument("--bars_csv", default=str(DEFAULT_BARS_CSV), help="5m OHLCV bars CSV covering the events")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory for metrics/scores/importance")
    p.add_argument("--n_folds", type=int, default=DEFAULT_N_TEST_FOLDS, help="Number of walk-forward test folds (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events_csv = Path(args.events_csv)
    bars_csv = Path(args.bars_csv)
    out_dir = Path(args.out_dir)

    ev = load_events(events_csv)
    bars = load_bars(str(bars_csv))
    ds, info = build_dataset(ev, bars)
    ds, fold_rows, fold_notes, pooled_auc, imp_avg = run_walk_forward(ds, args.n_folds)
    oof = ds[ds["prob"].notna()]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per_event_scores.csv
    score_cols = [
        "peak_t", "trough_t", "trough_confirm_t", "entry_t", "entry_price",
        "et_day", "outcome", "label", "prob", "fold",
    ] + FEATURES
    ds[score_cols].to_csv(out_dir / "per_event_scores.csv", index=False)

    # ---- feature_importance.txt
    order = np.argsort(imp_avg)[::-1]
    total = imp_avg.sum() or 1.0
    fi_lines = [
        "Feature importance (LightGBM gain, averaged over walk-forward folds)",
        "NOTE: with small event counts these rankings are unstable across folds;",
        "treat them as descriptive, not as validated predictive structure.",
        "",
    ]
    for j in order:
        fi_lines.append(
            f"{FEATURES[j]:<20s} {imp_avg[j]:>12.1f}  ({imp_avg[j] / total:6.1%})"
        )
    (out_dir / "feature_importance.txt").write_text("\n".join(fi_lines) + "\n")

    # ---- metrics.txt
    has_outcome = (ev["outcome"] == "timeout").any()
    banner = warning_banner(len(ev), has_outcome)
    n_ev_ok = int((ev["outcome"] == "rebounded").sum())
    n_ev_to = int((ev["outcome"] == "timeout").sum())
    L: list[str] = [banner]
    L.append("== Setup ==")
    L.append(
        f"events file        : {events_csv} ({len(ev)} physical triggers: "
        f"{n_ev_ok} rebounded, {n_ev_to} timeout)"
    )
    L.append(f"bars file          : {bars_csv} ({len(bars)} 5m bars)")
    L.append(
        f"label              : entry = Open of first bar after confirm bar; "
        f"+{LABEL_TP:.1%} via High -> 1; -{LABEL_SL:.1%} via Low, timeout "
        f"({LABEL_HORIZON} bars) or day end -> 0; both barriers in one bar -> 0"
    )
    L.append(
        f"validation         : day-grouped expanding walk-forward, "
        f"{args.n_folds} test folds, embargo = {EMBARGO_DAYS} trading day"
    )
    L.append("")
    L.append("== Sample ==")
    n_ds_to = int((ds["outcome"] == "timeout").sum())
    L.append(
        f"usable events      : {len(ds)} "
        f"({len(ds) - n_ds_to} rebounded, {n_ds_to} timeout)"
    )
    for k, v in info["drops"].items():
        L.append(f"dropped ({k}): {v}")
    L.append(f"ambiguous both-barrier bars labelled 0: {info['n_ambiguous_bars']}")
    L.append(f"positive rate (all usable) : {ds['label'].mean():.3f}")
    if n_ds_to:
        L.append(
            f"positive rate by outcome   : rebounded "
            f"{ds.loc[ds['outcome'] == 'rebounded', 'label'].mean():.3f}, "
            f"timeout {ds.loc[ds['outcome'] == 'timeout', 'label'].mean():.3f} "
            f"(label is triple-barrier, independent of outcome)"
        )
    L.append(
        f"out-of-fold scored events  : {len(oof)} "
        f"(fold=-1 seed-block events are never scored)"
    )
    if len(oof):
        L.append(f"positive rate (OOF)        : {oof['label'].mean():.3f}")
    L.append("")
    L.append("== Per-fold walk-forward ==")
    hdr = (
        f"{'fold':<5}{'train_n':>8}{'tr_pos':>8}{'test_n':>8}{'te_pos':>8}"
        f"{'AUC':>8}  test days / notes"
    )
    L.append(hdr)
    for r in fold_rows:
        auc_s = f"{r['auc']:.3f}" if np.isfinite(r["auc"]) else "n/a"
        L.append(
            f"{r['fold']:<5}{r['train_n']:>8}{r['train_pos_rate']:>8.3f}"
            f"{r['test_n']:>8}{r['test_pos_rate']:>8.3f}{auc_s:>8}  "
            f"{r['test_days']}"
            + (f"  [{r['note']}]" if r["note"] else "")
        )
    for note in fold_notes:
        L.append(f"  ! {note}")
    aucs = [r["auc"] for r in fold_rows if np.isfinite(r["auc"])]
    if aucs:
        L.append(
            f"fold AUC mean = {np.mean(aucs):.3f}, std = {np.std(aucs):.3f}, "
            f"min = {min(aucs):.3f}, max = {max(aucs):.3f}"
        )
    pooled_s = f"{pooled_auc:.3f}" if np.isfinite(pooled_auc) else "n/a"
    L.append(f"pooled out-of-fold AUC = {pooled_s}")
    L.append("")
    L.append("== Threshold table (pooled out-of-fold) ==")
    L.append(
        f"{'rule':<36}{'kept':>6}{'retention':>11}{'precision':>11}"
    )
    for row in threshold_table(oof):
        prec = f"{row['precision']:.3f}" if np.isfinite(row["precision"]) else "n/a"
        L.append(
            f"{row['threshold']:<36}{row['kept']:>6}{row['retention']:>11.1%}"
            f"{prec:>11}"
        )
    L.append("")
    L.append("== Verdict ==")
    L.append(verdict(pooled_auc, fold_rows))
    L.append("")
    L.append(banner)

    text = "\n".join(L)
    (out_dir / "metrics.txt").write_text(text)
    print(text)
    print(f"written: {out_dir / 'metrics.txt'}")
    print(f"written: {out_dir / 'per_event_scores.csv'}")
    print(f"written: {out_dir / 'feature_importance.txt'}")


if __name__ == "__main__":
    main()
