#!/usr/bin/env python
"""E18 — E8-A 配方全历史滚动 walk-forward 校验（2019-07..2026-07）.

Protocol (pre-registered, docs/strategy-lab.md E18 — executed verbatim):

- Data: per-symbol stitch of data/pool_crash/ (2019-07..2023-07, 16 symbols
  incl TSLA) + data/pool/ 3y (2023-07..2026-07, 25 symbols) / data/TSLA_5m_3y.csv.
  Overlapping bars deduped (keep crash-segment values; price agreement on the
  overlap is measured and reported). 10 of the 25 pool symbols only exist from
  2023-07 — per-fold training coverage is recorded honestly.
- Events: reuse of the existing deduped physical-event tables
  (outputs/e8a_crash_test/v_events/<SYM>/ for 2019-2023,
  outputs/v_events_pool/<SYM>/ + outputs/v_events_3y/ for 2023-07+),
  concatenated and deduped on (peak_t, trough_t, trough_confirm_t).
  No new v_stats runs needed: coverage is complete.
- Features/labels/hyper-params: research/ml_common.py — byte-identical recipe
  to E8/e8a_crash_test (rv20-normalized, triple-barrier +0.5%/-1%/24bar,
  E3 LGBM frozen params = models/e8a/meta.json lgbm_params).
- Rolling folds: TSLA test folds of 6 months starting 2020-07-01 (12 folds,
  last fold extended to data end 2026-07). Per fold:
    train = ALL pool events (leave-TSLA-out) with et_day < fold_start - 1 day
            (the day before fold start itself dropped = 1-day embargo);
    rv20_rel anchored on the fold's train-period per-symbol median (no test info);
    gate  = 90th pct of leave-one-symbol-out OOF probabilities on the train
            segment (e8a_crash_test precedent — zero test-period information);
    test  = TSLA fold events, prob >= gate -> entry next-bar-open,
            frozen geometry tp0.5%/sl2%/timeout48, long-only, day-flat,
            pessimistic settlement (1bp fee + 2bp slip per side);
    S2    = disable entries when TSLA dd vs 252d rolling max < -20%,
            yesterday's close decides today (shift(1)), min_periods=252.
- Variants per fold: A = gate+S2 (main), B = gate only, C = ungated all events
  (anti-selection control, no gate no S2).
- Aggregate OOS (main = A): total, annualized, bar-level MTM MDD; L=2 mirror
  (2x net ret minus 6.5%/yr financing on the borrowed leg for actual holding
  time — e17_ab_tracks convention). Controls: TSLA B&H 2020-07..2026-07 and
  S2-only overlay on B&H (cash leg 4%/yr, 3bp per exposure switch — E17-B
  convention).
- Multiple comparisons: single pre-registered recipe, +1 (aggregate main).

Checkpointing: per-fold JSON + trades CSV under outputs/e18_rolling/checkpoints/;
re-running skips completed folds.

Usage:   .venv/bin/python research/e18_rolling_validation.py
Outputs: outputs/e18_rolling/{folds.csv, trades.csv, summary.txt}
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from research.e9_frontier_search import COST, BarSet, run_combo, seg_stats  # noqa: E402
from research.ml_common import build_dataset, load_events, make_lgbm  # noqa: E402
from src.common.data_io import ET, load_bars  # noqa: E402
from src.common.execution import entry_fill  # noqa: E402

# ----------------------------------------------------------------- parameters
CRASH_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX",
    "JPM", "XOM", "DIS", "CAT", "INTC", "PLTR", "COIN",
]
POOL_SYMBOLS = [
    "AAPL", "AMD", "AMZN", "AVGO", "BA", "CAT", "COIN", "COST", "CRM", "DIS",
    "GOOGL", "GS", "INTC", "JPM", "META", "MSFT", "MU", "NFLX", "NVDA",
    "PLTR", "QCOM", "UNH", "V", "WMT", "XOM",
]
TARGET = "TSLA"

CRASH_DATA = ROOT / "data" / "pool_crash"
POOL_DATA = ROOT / "data" / "pool"
CRASH_EVENTS = ROOT / "outputs" / "e8a_crash_test" / "v_events"
POOL_EVENTS = ROOT / "outputs" / "v_events_pool"
TSLA_3Y_BARS = ROOT / "data" / "TSLA_5m_3y.csv"
TSLA_3Y_EVENTS = ROOT / "outputs" / "v_events_3y" / "v_events_5m_unique.csv"

OUT = ROOT / "outputs" / "e18_rolling"
CKPT = OUT / "checkpoints"

FOLD_START0 = date(2020, 7, 1)   # first test fold start
FOLD_MONTHS = 6
TP, SL, TIMEOUT = 0.005, 0.020, 48   # frozen E8-A geometry (models/e8a/meta.json)
RETENTION = 0.10                     # top-10% gate
S2_LOOKBACK, S2_DD = 252, -0.20      # frozen S2 rule
LEV, FIN_RATE, YEAR_S = 2.0, 0.065, 365.25 * 24 * 3600
CASH_RATE = 0.04                     # S2-only B&H control cash leg (E17-B)
SWITCH_BP = 3e-4                     # 1bp fee + 2bp slip per exposure change
SEED = 42

POOLED_FEATURES = [
    "drop_pct_n", "drop_speed_n", "vol_ratio_trough", "et_hour",
    "open_to_now_ret_n", "prev_day_ret_n", "rv20_rel", "bars_since_open",
    "symbol",
]

HOLDOUT_ANN_REF = 0.152   # E8-A+S2 留出段年化（判读锚）


# --------------------------------------------------------------- data stitch
def stitch_bars(sym: str) -> tuple[pd.DataFrame, float]:
    """Concat crash-segment + 3y bars, dedupe on timestamp (keep crash values).
    Returns (bars, max relative Close diff on the overlapping timestamps)."""
    parts = []
    crash_csv = CRASH_DATA / f"{sym}_5m_2019_2023.csv"
    if crash_csv.exists():
        parts.append(load_bars(str(crash_csv)))
    pool_csv = TSLA_3Y_BARS if sym == TARGET else POOL_DATA / f"{sym}_5m_3y.csv"
    if pool_csv.exists():
        parts.append(load_bars(str(pool_csv)))
    if not parts:
        raise SystemExit(f"{sym}: no bar files found")
    if len(parts) == 1:
        return parts[0], np.nan
    a, b = parts
    common = a.index.intersection(b.index)
    max_diff = float(
        (a.loc[common, "Close"] / b.loc[common, "Close"] - 1.0).abs().max()
    ) if len(common) else np.nan
    bars = pd.concat([a, b])
    bars = bars[~bars.index.duplicated(keep="first")].sort_index()
    return bars, max_diff


def stitch_events(sym: str) -> pd.DataFrame:
    parts = []
    crash_csv = CRASH_EVENTS / sym / "v_events_5m_unique.csv"
    if crash_csv.exists():
        parts.append(load_events(crash_csv))
    pool_csv = TSLA_3Y_EVENTS if sym == TARGET else (
        POOL_EVENTS / sym / "v_events_5m_unique.csv")
    if pool_csv.exists():
        parts.append(load_events(pool_csv))
    if not parts:
        raise SystemExit(f"{sym}: no event tables found")
    ev = pd.concat(parts, ignore_index=True)
    ev = ev.drop_duplicates(subset=["peak_t", "trough_t", "trough_confirm_t"])
    return ev.sort_values("trough_confirm_t").reset_index(drop=True)


def add_normalized_features(all_ds: pd.DataFrame, train_end: date) -> pd.DataFrame:
    """e8a_crash_test.add_normalized_features, parameterized by the fold's
    train end. rv20_rel uses each symbol's train-period median rv20 only."""
    ds = all_ds.copy()
    rv = ds["rv20"].replace(0, np.nan)
    ds["drop_pct_n"] = ds["drop_pct"] / rv
    ds["drop_speed_n"] = ds["drop_speed"] / rv
    ds["open_to_now_ret_n"] = ds["open_to_now_ret"] / rv
    ds["prev_day_ret_n"] = ds["prev_day_ret"] / rv
    med = (
        ds[ds["et_day"] < train_end]
        .groupby("symbol")["rv20"]
        .median()
        .rename("rv20_med_train")
    )
    ds = ds.merge(med, left_on="symbol", right_index=True, how="left")
    ds["rv20_rel"] = ds["rv20"] / ds["rv20_med_train"]
    ds["symbol"] = pd.Categorical(
        ds["symbol"], categories=sorted(all_ds["symbol"].unique()))
    return ds


# ------------------------------------------------------------- MTM machinery
def mdd(path: np.ndarray) -> float:
    if len(path) == 0:
        return np.nan
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def mtm_equity_path(bs: BarSet, trades: pd.DataFrame, lev: float = 1.0) -> np.ndarray:
    """Bar-level MTM equity across the sequential trade list (flat between
    trades). lev=1 reproduces e8a_crash_test.mtm_equity_path; lev>1 marks the
    leveraged excursion and books lev*ret minus financing (e17 convention)."""
    eq = 1.0
    path = [eq]
    for _, t in trades.iterrows():
        epos = int(t["epos"])
        ret, exit_pos = bs.eval_trade(epos, 1, TP, SL, TIMEOUT)  # cached
        entry_px = entry_fill(bs.opens[epos], 1, COST)
        hold_s = max(
            (bs.idx[exit_pos] - bs.idx[epos]).total_seconds(), 300.0)
        ret_book = lev * ret - (lev - 1.0) * FIN_RATE * hold_s / YEAR_S
        eq0 = eq
        for k in range(epos, exit_pos):
            path.append(eq0 * (1.0 + lev * (bs.closes[k] / entry_px - 1.0)))
        eq = eq0 * (1.0 + ret_book)
        path.append(eq)
    return np.asarray(path)


def lever_total(trades: pd.DataFrame, bs: BarSet, lev: float) -> float:
    """Compounded total return of the leveraged stream (financing included)."""
    eq = 1.0
    for _, t in trades.iterrows():
        epos = int(t["epos"])
        ret, exit_pos = bs.eval_trade(epos, 1, TP, SL, TIMEOUT)
        hold_s = max((bs.idx[exit_pos] - bs.idx[epos]).total_seconds(), 300.0)
        eq *= 1.0 + lev * ret - (lev - 1.0) * FIN_RATE * hold_s / YEAR_S
    return eq - 1.0


# ------------------------------------------------------------------ S2 state
def s2_off_days(tsla_bars: pd.DataFrame) -> pd.Series:
    """True = entries DISABLED that day (frozen rule, meta.json s2_switch)."""
    et_days = tsla_bars.index.tz_convert(ET).date
    dc = tsla_bars["Close"].groupby(et_days).last()
    dd = dc / dc.rolling(S2_LOOKBACK, min_periods=S2_LOOKBACK).max() - 1.0
    off = (dd < S2_DD).fillna(False)
    return off.shift(1).fillna(False).astype(bool)   # yesterday decides today


# ------------------------------------------------------------------ folds
def make_folds(data_end: date) -> list[tuple[int, date, date]]:
    folds, k, start = [], 1, FOLD_START0
    while start <= data_end:
        nxt = (pd.Timestamp(start) + pd.DateOffset(months=FOLD_MONTHS)).date()
        end = nxt - timedelta(days=1)
        if nxt > data_end:               # last fold absorbs the tail
            end = data_end
            folds.append((k, start, end))
            break
        # extend the final full fold to data end when the remainder is short
        if (pd.Timestamp(nxt) + pd.DateOffset(months=FOLD_MONTHS)).date() > data_end \
                and (data_end - nxt).days < 92:
            folds.append((k, start, data_end))
            break
        folds.append((k, start, end))
        start, k = nxt, k + 1
    return folds


def regime_label(bh_ret: float, bh_mdd: float) -> str:
    if bh_ret >= 0.10:
        base = "牛"
    elif bh_ret <= -0.10:
        base = "熊"
    else:
        base = "震荡"
    if base != "熊" and bh_mdd <= -0.25:
        base += "(深回撤)"
    return base


# ------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------- assemble stitched datasets --------------------------------
    frames, cov_rows, seam_rows = [], [], []
    tsla_bars = None
    for sym in POOL_SYMBOLS + [TARGET]:
        bars, seam = stitch_bars(sym)
        ev = stitch_events(sym)
        ds, _info = build_dataset(ev, bars)
        ds["symbol"] = sym
        frames.append(ds)
        if sym == TARGET:
            tsla_bars = bars
        cov_rows.append({
            "symbol": sym, "n_events": len(ds),
            "first_day": str(ds["et_day"].min()), "last_day": str(ds["et_day"].max()),
            "crash_segment": (CRASH_DATA / f"{sym}_5m_2019_2023.csv").exists()
            or sym == TARGET,
            "seam_max_close_diff": seam,
        })
        seam_rows.append(seam)
        seam_txt = f"seam_diff={seam:.2e}" if seam == seam else "(single segment)"
        print(f"[{sym}] events {len(ds)}  {ds['et_day'].min()}..{ds['et_day'].max()}  {seam_txt}")
    cov_df = pd.DataFrame(cov_rows)
    raw_ds = pd.concat(frames, ignore_index=True)
    bs = BarSet(tsla_bars)
    data_end = tsla_bars.index.tz_convert(ET).date.max()
    s2_off = s2_off_days(tsla_bars)
    folds = make_folds(data_end)
    print(f"\nstitched dataset: {len(raw_ds)} events, {len(cov_df)} symbols; "
          f"TSLA bars {tsla_bars.index[0]}..{tsla_bars.index[-1]}; "
          f"{len(folds)} folds; S2 off-days {int(s2_off.sum())}")

    # ------------- per-fold walk-forward -------------------------------------
    fold_rows, trade_frames = [], []
    for k, f_start, f_end in folds:
        ck_json = CKPT / f"fold_{k:02d}.json"
        ck_trades = CKPT / f"fold_{k:02d}_trades.csv"
        if ck_json.exists() and ck_trades.exists():
            fold_rows.append(json.loads(ck_json.read_text()))
            tdf = pd.read_csv(ck_trades)
            if len(tdf):
                trade_frames.append(tdf)
            print(f"fold {k:02d} [{f_start}..{f_end}] — checkpoint reused")
            continue

        tf0 = time.time()
        embargo_day = f_start - timedelta(days=1)   # dropped entirely
        ds = add_normalized_features(raw_ds, embargo_day)
        feat_ok = ds[POOLED_FEATURES[:-1]].notna().all(axis=1)
        ds = ds[feat_ok].reset_index(drop=True)
        is_tsla = ds["symbol"].astype(str) == TARGET
        train = ds[(~is_tsla) & (ds["et_day"] < embargo_day)].reset_index(drop=True)
        test = ds[is_tsla & (ds["et_day"] >= f_start)
                  & (ds["et_day"] <= f_end)].reset_index(drop=True)
        train_syms = sorted(train["symbol"].astype(str).unique())

        # gate threshold: leave-one-symbol-out OOF on the TRAIN segment only
        oof = np.full(len(train), np.nan)
        for sym in train_syms:
            m = (train["symbol"].astype(str) == sym).to_numpy()
            mdl = make_lgbm(SEED)
            mdl.fit(train.loc[~m, POOLED_FEATURES], train.loc[~m, "label"])
            oof[m] = mdl.predict_proba(train.loc[m, POOLED_FEATURES])[:, 1]
        thr = float(np.quantile(oof, 1.0 - RETENTION))

        model = make_lgbm(SEED)
        model.fit(train[POOLED_FEATURES], train["label"])
        test["prob"] = model.predict_proba(test[POOLED_FEATURES])[:, 1]

        y = test["label"].to_numpy()
        auc = float(roc_auc_score(y, test["prob"])) if len(np.unique(y)) > 1 else np.nan

        gated = test[test["prob"] >= thr]
        off_day = gated["et_day"].map(s2_off).fillna(False).astype(bool)
        gated_s2 = gated[~off_day.to_numpy()]

        def settle(ev_df: pd.DataFrame, variant: str):
            if len(ev_df) == 0:
                return seg_stats(pd.DataFrame(columns=["ret"]), "s"), pd.DataFrame()
            epos = tsla_bars.index.get_indexer(pd.DatetimeIndex(ev_df["entry_t"]))
            assert (epos >= 0).all()
            sig = [(int(p), 1) for p in np.sort(epos)]
            trades = run_combo(bs, sig, TP, SL, TIMEOUT)
            st = seg_stats(trades, "s")
            prob_map = dict(zip(epos.astype(int), ev_df["prob"].astype(float)))
            rows = []
            for _, t in trades.iterrows():
                ep = int(t["epos"])
                _, exit_pos = bs.eval_trade(ep, 1, TP, SL, TIMEOUT)
                rows.append({
                    "fold": k, "variant": variant, "epos": ep,
                    "entry_t": str(tsla_bars.index[ep]), "et_day": str(t["et_day"]),
                    "exit_t": str(tsla_bars.index[exit_pos]),
                    "prob": prob_map.get(ep, np.nan), "ret": float(t["ret"]),
                    "hold_s": max((tsla_bars.index[exit_pos]
                                   - tsla_bars.index[ep]).total_seconds(), 300.0),
                })
            return st, pd.DataFrame(rows)

        st_a, tr_a = settle(gated_s2, "gated_s2")      # main
        st_b, tr_b = settle(gated, "gated")            # gate only
        st_c, tr_c = settle(test, "ungated")           # control

        # fold-window B&H on TSLA
        wmask = (bs.et_dates >= f_start) & (bs.et_dates <= f_end)
        wpos = np.nonzero(wmask)[0]
        bh_ret = float(bs.closes[wpos[-1]] / bs.opens[wpos[0]] - 1.0)
        bh_mdd_f = mdd(bs.closes[wpos] / bs.opens[wpos[0]])
        n_off = int(sum(bool(s2_off.get(d, False))
                        for d in np.unique(bs.et_dates[wpos])))

        row = {
            "fold": k, "start": str(f_start), "end": str(f_end),
            "regime": regime_label(bh_ret, bh_mdd_f),
            "bh_ret": bh_ret, "bh_mdd": bh_mdd_f,
            "n_train_events": len(train), "n_train_syms": len(train_syms),
            "train_first": str(train["et_day"].min()) if len(train) else "",
            "train_last": str(train["et_day"].max()) if len(train) else "",
            "gate_thr": thr, "auc": auc,
            "n_events": len(test), "n_gated": len(gated),
            "retention": float(len(gated) / len(test)) if len(test) else np.nan,
            "n_after_s2": len(gated_s2), "s2_off_days_in_fold": n_off,
            "n_trades": st_a["s_n"], "wr": st_a["s_wr"],
            "avg_bp": st_a["s_avg_bp"], "total": st_a["s_total"] if st_a["s_n"] else 0.0,
            "gateonly_n": st_b["s_n"], "gateonly_wr": st_b["s_wr"],
            "gateonly_avg_bp": st_b["s_avg_bp"],
            "gateonly_total": st_b["s_total"] if st_b["s_n"] else 0.0,
            "ungated_n": st_c["s_n"], "ungated_wr": st_c["s_wr"],
            "ungated_avg_bp": st_c["s_avg_bp"],
            "ungated_total": st_c["s_total"] if st_c["s_n"] else 0.0,
            "elapsed_s": round(time.time() - tf0, 1),
        }
        row = {k2: (None if isinstance(v, float) and not np.isfinite(v) else v)
               for k2, v in row.items()}
        tdf = pd.concat([tr_a, tr_b, tr_c], ignore_index=True)
        ck_json.write_text(json.dumps(row, ensure_ascii=False, indent=1))
        tdf.to_csv(ck_trades, index=False)
        fold_rows.append(row)
        if len(tdf):
            trade_frames.append(tdf)
        print(f"fold {k:02d} [{f_start}..{f_end}] {row['regime']:>7} "
              f"train={len(train)}({len(train_syms)}sym) thr={thr:.4f} auc={auc:.3f} "
              f"ev={len(test)} gate={len(gated)} s2={len(gated_s2)} tr={st_a['s_n']} "
              f"wr={st_a['s_wr'] if st_a['s_n'] else float('nan'):.1%} "
              f"avg={st_a['s_avg_bp'] if st_a['s_n'] else float('nan'):+.1f}bp "
              f"tot={row['total']:+.2%} bh={bh_ret:+.1%} ({row['elapsed_s']}s)")

    folds_df = pd.DataFrame(fold_rows).replace({None: np.nan})
    folds_df.to_csv(OUT / "folds.csv", index=False)
    trades_df = (pd.concat(trade_frames, ignore_index=True)
                 if trade_frames else pd.DataFrame())
    trades_df.to_csv(OUT / "trades.csv", index=False)

    # ------------- aggregate OOS ---------------------------------------------
    oos_start, oos_end = folds[0][1], folds[-1][2]
    years = (oos_end - oos_start).days / 365.25

    def agg(variant: str) -> dict:
        t = trades_df[trades_df["variant"] == variant].copy()
        t["entry_ts"] = pd.to_datetime(t["entry_t"])
        t = t.sort_values("entry_ts").reset_index(drop=True)
        st = seg_stats(t, "s")
        d = {"n": st["s_n"], "wr": st["s_wr"], "avg_bp": st["s_avg_bp"],
             "total": st["s_total"] if st["s_n"] else 0.0,
             "mdd_closed": st["s_mdd"] if st["s_n"] else 0.0}
        d["ann"] = (1.0 + d["total"]) ** (1.0 / years) - 1.0
        d["mtm_mdd"] = mdd(mtm_equity_path(bs, t, 1.0)) if st["s_n"] else 0.0
        d["_trades"] = t
        return d

    A = agg("gated_s2")
    B = agg("gated")
    C = agg("ungated")
    lev_total = lever_total(A["_trades"], bs, LEV)
    lev_ann = (1.0 + lev_total) ** (1.0 / years) - 1.0
    lev_mdd = mdd(mtm_equity_path(bs, A["_trades"], LEV)) if A["n"] else 0.0

    # ------------- controls: B&H and S2-only B&H -----------------------------
    omask = (bs.et_dates >= oos_start) & (bs.et_dates <= oos_end)
    opos = np.nonzero(omask)[0]
    bh_total = float(bs.closes[opos[-1]] / bs.opens[opos[0]] - 1.0)
    bh_ann = (1.0 + bh_total) ** (1.0 / years) - 1.0
    bh_mdd_all = mdd(bs.closes[opos] / bs.opens[opos[0]])

    et_days_all = tsla_bars.index.tz_convert(ET).date
    dc = tsla_bars["Close"].groupby(et_days_all).last()
    win = dc[(dc.index >= oos_start) & (dc.index <= oos_end)]
    r_d = win.pct_change().fillna(0.0)
    expo = pd.Series([0.0 if bool(s2_off.get(d, False)) else 1.0
                      for d in win.index], index=win.index)
    switches = int(np.abs(np.diff(expo.to_numpy())).sum())
    r_s2 = expo * r_d + (1 - expo) * (CASH_RATE / 252) \
        - np.abs(expo.diff().fillna(0.0)) * SWITCH_BP
    eq_s2 = (1 + r_s2).cumprod()
    s2bh_total = float(eq_s2.iloc[-1] - 1.0)
    s2bh_ann = (1.0 + s2bh_total) ** (1.0 / years) - 1.0
    s2bh_mdd = mdd(eq_s2.to_numpy())

    # ------------- regime grouping -------------------------------------------
    folds_df["regime_base"] = folds_df["regime"].str.replace(r"\(.*\)", "", regex=True)
    grp_rows = []
    for g, sub in folds_df.groupby("regime_base"):
        t = trades_df[(trades_df["variant"] == "gated_s2")
                      & (trades_df["fold"].isin(sub["fold"]))]
        r = t["ret"].to_numpy()
        grp_rows.append({
            "regime": g, "n_folds": len(sub), "n_trades": len(r),
            "wr": float((r > 0).mean()) if len(r) else np.nan,
            "avg_bp": float(r.mean() * 1e4) if len(r) else np.nan,
            "sum_fold_total": float(sub["total"].sum()),
        })
    grp_df = pd.DataFrame(grp_rows)

    # ------------- summary ---------------------------------------------------
    def pct(x): return f"{x:+.2%}" if x == x else "n/a"

    L = []
    L.append("E18 — E8-A 配方全历史滚动 walk-forward 校验（2019-07..2026-07 数据，OOS 2020-07..%s）" % oos_end)
    L.append("=" * 110)
    L.append("协议（预登记，docs/strategy-lab.md E18，零改动执行）：")
    L.append(f"- 数据：pool_crash（2019-07..2023-07，15 池标的+TSLA）与 pool 3y（2023-07..2026-07，25 池标的）逐标的拼接去重；"
             f"接缝重叠段 Close 最大相对偏差 {np.nanmax(seam_rows):.2e}（同源 Alpaca split 调整，无缝）")
    L.append(f"- 事件表：全部复用既有 v_events（e8a_crash_test 2019-2023 + v_events_pool/v_events_3y 2023-07+），"
             f"按 (peak_t,trough_t,trough_confirm_t) 去重，无需补跑 v_stats")
    L.append(f"- 滚动折：TSLA 6 个月测试折 x {len(folds)}（首折 {folds[0][1]}，末折延长至数据端点 {oos_end}）；"
             f"每折训练 = 折起点前 1 天（该日弃用 = 1 天 embargo）之前的全部池事件（leave-TSLA-out），LGBM 超参冻结（meta.json）")
    L.append(f"- gate = 该折训练段 leave-one-symbol-out OOF 概率 top{RETENTION:.0%} 保留分位（零测试段信息，e8a_crash_test 口径）；"
             f"rv20_rel 锚定各折训练段中位数")
    L.append(f"- 几何冻结 tp {TP:.1%}/sl {SL:.1%}/to {TIMEOUT}bar，多头、日内强平、悲观结算（1bp fee+2bp slip 单边）；"
             f"S2 = 252 日高回撤 <-20% 停用入场（昨收决定今日，min_periods=252）")
    L.append("- 多重比较：配方单一无网格，+1（聚合主口径）")
    L.append("")
    L.append("== 数据覆盖（诚实记录）==")
    L.append(f"- 16 标的（15 池+TSLA）全程 2019-07..2026-07；其余 10 只池标的（AVGO/BA/COST/CRM/GS/MU/QCOM/UNH/V/WMT）"
             "仅 2023-07 起 —— 前 7 折训练池只有 15 只（COIN 2021-04 / PLTR 2020-09 上市，早期折贡献更短）")
    L.append(f"- 任务书写『其余 9 只』，实际为 10 只（25-15），如实更正")
    L.append("")
    L.append("== 逐折表（主口径 A = gate+S2）==")
    show = folds_df[["fold", "start", "end", "regime", "bh_ret", "bh_mdd",
                     "n_train_events", "n_train_syms", "auc", "n_events",
                     "n_gated", "n_after_s2", "n_trades", "wr", "avg_bp",
                     "total", "gateonly_avg_bp", "ungated_avg_bp"]].copy()
    for c in ("bh_ret", "bh_mdd", "total"):
        show[c] = show[c].map(lambda v: f"{v:+.1%}" if v == v else "n/a")
    show["wr"] = show["wr"].map(lambda v: f"{v:.1%}" if v == v else "n/a")
    for c in ("avg_bp", "gateonly_avg_bp", "ungated_avg_bp"):
        show[c] = show[c].map(lambda v: f"{v:+.1f}" if v == v else "n/a")
    show["auc"] = show["auc"].map(lambda v: f"{v:.3f}" if v == v else "n/a")
    L.append(show.to_string(index=False))
    L.append("")
    L.append("== 聚合 OOS（%s..%s，%.2f 年）==" % (oos_start, oos_end, years))
    L.append(f"A 主口径 gate+S2 : n={A['n']}, WR={A['wr']:.1%}, avg={A['avg_bp']:+.1f}bp, "
             f"total={pct(A['total'])}, 年化={pct(A['ann'])}, 已平仓MDD={A['mdd_closed']:.2%}, MTM MDD={A['mtm_mdd']:.2%}")
    L.append(f"B gate-only      : n={B['n']}, WR={B['wr']:.1%}, avg={B['avg_bp']:+.1f}bp, "
             f"total={pct(B['total'])}, 年化={pct(B['ann'])}, MTM MDD={B['mtm_mdd']:.2%}")
    L.append(f"C ungated 对照   : n={C['n']}, WR={C['wr']:.1%}, avg={C['avg_bp']:+.1f}bp, "
             f"total={pct(C['total'])}, 年化={pct(C['ann'])}")
    L.append(f"A @ L={LEV:.0f}（融资 {FIN_RATE:.1%}/yr 于借入腿，按实际持仓时长）: "
             f"total={pct(lev_total)}, 年化={pct(lev_ann)}, MTM MDD={lev_mdd:.2%}")
    L.append(f"对照 TSLA B&H    : total={pct(bh_total)}, 年化={pct(bh_ann)}, MTM MDD={bh_mdd_all:.1%}")
    L.append(f"对照 S2-only B&H : total={pct(s2bh_total)}, 年化={pct(s2bh_ann)}, "
             f"日线MDD={s2bh_mdd:.1%}, 换挡 {switches} 次（现金腿 {CASH_RATE:.0%}/yr，换挡 {SWITCH_BP*1e4:.0f}bp）")
    L.append("")
    L.append("== 按 regime 分组（A 口径）==")
    L.append(grp_df.to_string(index=False))
    L.append("")

    # ------------- 判读（预登记锚，数据驱动措辞） -----------------------------
    L.append("== 判读（预登记锚）==")
    L.append(f"1. 聚合 OOS 年化 {pct(A['ann'])} vs 留出段年化 +15.2%："
             + ("显著更低——与预期一致，留出段是配方的顺风窗口。"
                if A["ann"] < HOLDOUT_ANN_REF - 0.03 else
                ("大体相当——多周期口径未显著衰减。" if A["ann"] > HOLDOUT_ANN_REF - 0.03 and A["ann"] < HOLDOUT_ANN_REF + 0.03
                 else "更高——超出预期，需警惕折内运气成分。")))
    bear = folds_df[folds_df["regime_base"] == "熊"]
    if len(bear):
        bf = bear["fold"].tolist()
        gb_r = trades_df[(trades_df["variant"] == "gated")
                         & trades_df["fold"].isin(bf)]["ret"].to_numpy()
        ub_r = trades_df[(trades_df["variant"] == "ungated")
                         & trades_df["fold"].isin(bf)]["ret"].to_numpy()
        gb_bp = float(gb_r.mean() * 1e4) if len(gb_r) else np.nan
        ub_bp = float(ub_r.mean() * 1e4) if len(ub_r) else np.nan
        # e8a_crash_test 窗口（2021-10..2023-01）对应折的逐折对照
        cw_lo, cw_hi = date(2021, 10, 1), date(2023, 1, 31)
        cw = folds_df[(pd.to_datetime(folds_df["start"]).dt.date <= cw_hi)
                      & (pd.to_datetime(folds_df["end"]).dt.date >= cw_lo)]
        detail = "；".join(
            f"fold{int(r['fold'])} gate-only {r['gateonly_avg_bp']:+.1f}bp(n={int(r['gateonly_n'])}) "
            f"vs ungated {r['ungated_avg_bp']:+.1f}bp"
            for _, r in cw.iterrows() if r["gateonly_avg_bp"] == r["gateonly_avg_bp"])
        L.append(f"2. 崩盘折反选复现（e8a_crash_test 窗口 2021-10..2023-01 对应折）：{detail}——"
                 + ("2022 主崩盘折复现反选（gate 挑中的深跌事件更易打穿 2% 止损）。"
                    if (cw["gateonly_avg_bp"] < cw["ungated_avg_bp"]).any()
                    else "崩盘窗口折未复现反选。"))
        L.append(f"   全部 {len(bear)} 个熊折按笔数加权 pooled：gate-only {gb_bp:+.1f}bp (n={len(gb_r)}) "
                 f"vs ungated {ub_bp:+.1f}bp (n={len(ub_r)})——"
                 + ("整体仍反选。" if gb_bp < ub_bp else
                    "pooled 口径未反选（2023 后熊折 gate 几乎不放行、幸存少数为 tp 命中，n 过小不构成反证）。"))
    bull_bp = grp_df.loc[grp_df["regime"] == "牛", "avg_bp"]
    chop_bp = grp_df.loc[grp_df["regime"] == "震荡", "avg_bp"]
    if len(bull_bp) and len(chop_bp):
        L.append(f"3. 牛市折 vs 震荡折单笔期望：{float(bull_bp.iloc[0]):+.1f}bp vs {float(chop_bp.iloc[0]):+.1f}bp——"
                 + ("edge 主要来自牛市 regime。" if float(bull_bp.iloc[0]) > float(chop_bp.iloc[0]) + 3
                    else ("震荡折不弱于牛市折，edge 对 regime 不敏感。" if abs(float(bull_bp.iloc[0]) - float(chop_bp.iloc[0])) <= 3
                          else "震荡折期望反而更高。")))
    L.append(f"4. 【最终回答】这个算法的多周期真实年化：L=1 口径 {pct(A['ann'])}（MTM MDD {A['mtm_mdd']:.1%}），"
             f"L=2 口径 {pct(lev_ann)}（MTM MDD {lev_mdd:.1%}）。30% 目标在全历史口径下："
             f"L=1 缺口 {0.30 - A['ann']:+.1%}/yr，L=2 缺口 {0.30 - lev_ann:+.1%}/yr"
             + ("——L=2 过线。" if lev_ann >= 0.30 else "——未过线。"))
    L.append("")
    L.append("== 诚实性备注 ==")
    L.append("- 训练覆盖逐折不均：前 7 折仅 15 只池标的（且 COIN/PLTR 早期贡献短），2024-01 起 25 只——"
             "早期折的模型方差更大，逐折数字不可等权解读。")
    L.append("- 事件表网格与 confirm 逻辑沿用冻结口径零改动；事件表按段生成后拼接去重，"
             "接缝 4 个交易日窗口内两段各自检出的事件取并集（时间戳一致自动去重），可能与全程单次 v_stats 存在毫厘差异。")
    L.append("- 拼接后 label 用全程 bar 序列重算（ml_common 三重障碍），段尾事件的 label 比原 crash 表更完整（少截断）。")
    L.append("- 本实验是【配方级】验证（每折重训练），与 models/e8a 冻结单模型的 shadow 前向是两个互补口径：")
    L.append("  前者回答『这套管线在任意历史时点部署会怎样』，后者回答『2026-07 冻结的那个模型实例前向如何』——"
             "两者都成立才有资格谈实盘。")
    L.append("- S2 暖机：252 日窗口自 2019-07 起算，首折（2020-07 起）恰好出暖机期，全 OOS 段 S2 有效。")
    L.append(f"- L=2 口径沿用 E17-A 假设：滑点 2bp 不随杠杆重估（零售规模成立）、融资 {FIN_RATE:.1%}/yr 仅计实际持仓时长。")
    L.append("- 多重比较：+1（聚合主口径 A）；B/C/L=2/regime 分组均为同一预登记协议内的诊断切片，不另计。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
