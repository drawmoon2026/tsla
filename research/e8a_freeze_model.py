#!/usr/bin/env python
"""E8-A model freeze: retrain the leave-TSLA-out pooled GBDT once and persist
the artifact for the trading/ real-time layer (docs/strategy-lab.md E11 待办).

This is NOT a new experiment — it replays the exact e8_pooled_gbdt.py
a_leave_tsla_out configuration (same data, same features, same seed,
deterministic LightGBM) and asserts the result against the stored archives
before persisting:

- AUC on the TSLA holdout must match outputs/e8_pooled/auc_comparison.csv
- retention-matched gate threshold (E9 gate 0.70 -> top ~10% of new probs)
  must match frontier_shift.csv's 0.43250235336744486
- replayed E8-A holdout trades must match the stored aggregates
  (n=62, WR 80.65%, +15.97bp) — the same assertion e11_bear_switch.py uses.

Artifacts written to models/e8a/:
    model.joblib      sklearn LGBMClassifier (used by the live strategy)
    model.txt         native LightGBM booster dump (archival / portability)
    meta.json         frozen params: features, gate, geometry, S2 rule,
                      symbol categories, TSLA train-period median rv20,
                      training cutoff 2025-09-30, freeze statement
    holdout_ref.csv   per-event reference for the shadow smoke check:
                      entry_t / confirm_t / prob / gate_pass / s2_off /
                      in_trade (member of the 62-trade E8-A stream)

Frozen parameters (E11 freeze declaration 2026-07-24) — DO NOT EDIT:
    gate      prob >= 0.43250235336744486  (top10% retention mapping)
    geometry  tp +0.5% / sl -2.0% / timeout 48 x 5m bars / day-flat, long only
    S2        disable entries when yesterday's close is >20% below the max of
              the last 252 daily closes (incl. yesterday); rolling(252)
              min_periods=252, shift(1) — e11_bear_switch.switch_states

Usage:  .venv/bin/python research/e8a_freeze_model.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import research.e8_pooled_gbdt as e8  # noqa: E402
from research.e11_bear_switch import daily_closes, switch_states  # noqa: E402
from research.e9_frontier_search import BarSet, run_combo, seg_stats, signals_A  # noqa: E402
from research.ml_common import (  # noqa: E402
    LABEL_HORIZON, LABEL_SL, LABEL_TP, make_lgbm,
)
from src.common.data_io import load_bars  # noqa: E402

OUT = ROOT / "models" / "e8a"

# frozen E8-A + S2 (strategy-lab E11 freeze declaration, 2026-07-24)
GATE_ARCHIVED = 0.43250235336744486   # frontier_shift.csv gate=0.70 row
E9_GATE = 0.70
TP, SL, TIMEOUT = 0.005, 0.020, 48
EXP_N, EXP_WR, EXP_AVG_BP = 62, 0.8064516129032258, 15.965470720673224
EXP_AUC = 0.642                        # auc_comparison.csv a_leave_tsla_out
S2_LOOKBACK, S2_DD = 252, -0.20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- dataset (exact e8_pooled_gbdt pipeline) ----------------
    frames = []
    for sym in e8.POOL_SYMBOLS + [e8.TARGET]:
        got = e8.build_symbol_dataset(sym)
        if got is None:
            raise SystemExit(f"[{sym}] bars/events missing — cannot freeze E8-A")
        frames.append(got[0])
        print(f"[{sym}] dataset built ({len(got[0])} events)")
    all_ds = e8.add_normalized_features(pd.concat(frames, ignore_index=True))
    feat_ok = all_ds[e8.POOLED_FEATURES[:-1]].notna().all(axis=1)
    all_ds = all_ds[feat_ok].reset_index(drop=True)

    is_tsla = all_ds["symbol"].astype(str) == e8.TARGET
    loo_train = all_ds[(~is_tsla) & (all_ds["et_day"] < e8.TRAIN_END)]
    tsla_hold = all_ds[is_tsla & (all_ds["et_day"] >= e8.HOLDOUT_START)].reset_index(drop=True)
    print(f"leave-TSLA-out train n={len(loo_train)}, TSLA holdout n={len(tsla_hold)}")

    tsla_rv20_med_train = float(
        all_ds.loc[is_tsla, "rv20_med_train"].iloc[0]
    )  # per-symbol constant merged by add_normalized_features
    symbol_categories = list(all_ds["symbol"].cat.categories)

    # ---------------- train (deterministic, seed 42) -------------------------
    model = make_lgbm(e8.SEED)
    model.fit(loo_train[e8.POOLED_FEATURES], loo_train["label"])
    probs = model.predict_proba(tsla_hold[e8.POOLED_FEATURES])[:, 1]
    auc = roc_auc_score(tsla_hold["label"], probs)
    print(f"TSLA holdout AUC = {auc:.4f} (archived {EXP_AUC})")
    assert abs(auc - EXP_AUC) < 0.005, "AUC does not reproduce the archive"

    # ---------------- retention-matched gate threshold -----------------------
    bars = load_bars(str(e8.TSLA_BARS))
    bs = BarSet(bars)
    scores = tsla_hold[["entry_t", "trough_confirm_t", "et_day"]].copy()
    scores["prob"] = probs
    scores["epos"] = bars.index.get_indexer(pd.DatetimeIndex(scores["entry_t"]))
    scores = scores[scores["epos"] >= 0].sort_values("epos").reset_index(drop=True)

    old = pd.read_csv(ROOT / "outputs" / "ml_filter_3y" / "per_event_scores.csv")
    old = old[old["prob"].notna()].copy()
    old["et_day"] = pd.to_datetime(old["et_day"]).dt.date
    retention = float((old[old["et_day"] >= e8.HOLDOUT_START]["prob"] >= E9_GATE).mean())
    thr = float(np.quantile(scores["prob"].to_numpy(), 1.0 - retention))
    print(f"gate threshold (retention {retention:.4f}) = {thr!r} "
          f"(archived {GATE_ARCHIVED!r})")
    assert abs(thr - GATE_ARCHIVED) < 1e-9, "gate threshold does not reproduce the archive"

    # ---------------- replay the 62 E8-A holdout trades ----------------------
    trades = run_combo(bs, signals_A(bs, scores, thr, session=False), TP, SL, TIMEOUT)
    st = seg_stats(trades, "h")
    print(f"replayed E8-A trades: n={st['h_n']}, WR={st['h_wr']:.2%}, "
          f"avg={st['h_avg_bp']:+.2f}bp")
    assert st["h_n"] == EXP_N and abs(st["h_wr"] - EXP_WR) < 1e-6 \
        and abs(st["h_avg_bp"] - EXP_AVG_BP) < 0.05, "62-trade replay mismatch"

    # ---------------- S2 states over the holdout (reference only) ------------
    st_h = switch_states(daily_closes(bars))["S2"]
    trade_epos = set(trades["epos"].astype(int))

    ref = scores.copy()
    ref["gate_pass"] = ref["prob"] >= thr
    ref["s2_off"] = ref["et_day"].map(st_h).fillna(False).astype(bool)
    ref["in_trade"] = ref["epos"].astype(int).isin(trade_epos) & ref["gate_pass"]
    ref.to_csv(OUT / "holdout_ref.csv", index=False,
               date_format="%Y-%m-%dT%H:%M:%S%z")
    n_s2_kept = int((~trades["et_day"].map(st_h).fillna(False).astype(bool)).sum())
    print(f"E8-A+S2 reference: {n_s2_kept}/{EXP_N} trades survive S2 "
          f"(e11 archive: 54/62)")
    assert n_s2_kept == 54, "S2-filtered trade count does not reproduce e11"

    # ---------------- persist ------------------------------------------------
    joblib.dump(model, OUT / "model.joblib")
    model.booster_.save_model(str(OUT / "model.txt"))

    meta = {
        "strategy": "E8-A+S2",
        "frozen_at": "2026-07-24",
        "freeze_statement": (
            "候选 = E8-A + S2 绑定（docs/strategy-lab.md E11 冻结声明 2026-07-24）。"
            "参数与开关规则自此不得再调；裁决权移交 shadow 前向 >= 8 周。"
        ),
        "persisted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "kind": "lightgbm.LGBMClassifier",
            "training": "a_leave_tsla_out (25 pool symbols, no TSLA)",
            "train_end": str(e8.TRAIN_END),
            "holdout_start": str(e8.HOLDOUT_START),
            "seed": e8.SEED,
            "n_train": int(len(loo_train)),
            "holdout_auc": float(auc),
            "lgbm_params": model.get_params(),
        },
        "features": e8.POOLED_FEATURES,
        "symbol_categories": symbol_categories,
        "tsla_rv20_med_train": tsla_rv20_med_train,
        "label_spec": {"tp": LABEL_TP, "sl": LABEL_SL, "horizon_bars": LABEL_HORIZON,
                       "note": "training label only; trading geometry differs"},
        "gate": {
            "threshold": thr,
            "archived_threshold": GATE_ARCHIVED,
            "provenance": ("E9 gate 0.70 retention-matched quantile on the TSLA "
                           "holdout signal pool (top ~10% kept), "
                           "outputs/e8_pooled/frontier_shift.csv"),
        },
        "geometry": {"direction": "long_only", "tp_pct": TP, "sl_pct": SL,
                     "timeout_bars": TIMEOUT, "bar_seconds": 300,
                     "day_flat": True},
        "s2_switch": {
            "rule": ("disable entries when dd = close/rolling_max(close,252) - 1 "
                     "< -0.20, evaluated on YESTERDAY's daily close "
                     "(shift(1)); min_periods=252 (inactive before warm-up)"),
            "lookback_days": S2_LOOKBACK, "dd_threshold": S2_DD,
        },
        "v_detection": {
            "source": "src/v_stats.py 5m unique tables (deduped grid union)",
            "grid": {"x": [0.005, 0.01, 0.02], "y": [0.003, 0.005, 0.01],
                     "t": [12, 24, 48, 78]},
            "lookback": 3, "lookforward": 3, "max_drop_bars": 78,
        },
        "holdout_reference": {
            "n_trades": EXP_N, "wr": EXP_WR, "avg_bp": EXP_AVG_BP,
            "n_trades_after_s2": n_s2_kept,
        },
        "data_files": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [e8.TSLA_BARS, e8.TSLA_EVENTS]
        },
    }
    (OUT / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"artifacts -> {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
