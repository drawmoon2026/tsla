#!/usr/bin/env python
"""E9 — 70% win-rate + positive-expectation frontier search (2023-07 .. 2026-07, 5m).

Protocol (docs/strategy-lab.md E9, fixed BEFORE looking at results):
- Split: train = entries with ET date <  2025-10-01
         holdout = entries with ET date >= 2025-10-01  (~10 months)
- Train screen  : WR >= 70%  AND  avg >= +5bp (net)  AND  n >= 50  -> candidate set
- Holdout pass  : WR >= 70%  AND  avg >= +5bp (net)  AND  n >= 30
- Bootstrap p on holdout mean (1000 resamples) for every candidate;
  multiple-comparison ledger: report N candidates and Bonferroni p*N.
- Qualifiers get +-10% perturbation (tp / sl / gate-or-signal-param).
- Crash stress on data/TSLA_1h_alpaca.csv 2021-11 .. 2023-01 (1h-granularity
  approximation, see CRASH_NOTE below for the exact mapping).
- Random-entry baseline: for each qualifying / frontier geometry, 200
  simulations with the same (direction, tp, sl, timeout) at uniformly random
  holdout bars -> the real combo must beat the baseline to carry information.

Execution model: src.common.execution.settle_bracket EVERYWHERE (pessimistic:
gap-through stops, strict-cross TP, SL priority, 1bp fee + 2bp slip per side),
entry = next-bar open with adverse slippage, one position at a time, always
flat by day end.

Families:
  A  GBDT-gated V-rebound long (OOF probabilities from outputs/ml_filter_3y).
     Gate uses ONLY out-of-fold probs; gate=None also restricted to the
     OOF-scored set so gates are comparable.
  B  1H momentum follow/fade: signal = previous COMPLETE 1h bucket's
     intraday return (close/open-1); direction = sign (follow) or -sign
     (fade); settled on the 5m path.
  C  Intraday dip-buy long: Close crosses below (running day high * (1-d)).
  Overlay (top train combos per family): entries restricted to 10:30-15:00 ET
  (exclude first and last hour).

Usage:  .venv/bin/python research/e9_frontier_search.py
Outputs: outputs/e9_frontier/{all_combos.csv, candidates_train.csv,
         holdout_results.csv, pareto.csv, random_baseline.csv, summary.txt}
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

from src.common.data_io import ET, load_bars, resample_bars  # noqa: E402
from src.common.execution import CostModel, entry_fill, settle_bracket  # noqa: E402

# ----------------------------------------------------------------- parameters
BARS_5M = ROOT / "data" / "TSLA_5m_3y.csv"
BARS_1H = ROOT / "data" / "TSLA_1h_alpaca.csv"
SCORES = ROOT / "outputs" / "ml_filter_3y" / "per_event_scores.csv"
OUT = ROOT / "outputs" / "e9_frontier"

HOLDOUT_START = date(2025, 10, 1)
CRASH_START, CRASH_END = date(2021, 11, 1), date(2023, 1, 31)

TRAIN_WR, TRAIN_BP, TRAIN_N = 0.70, 5.0, 50
HOLD_WR, HOLD_BP, HOLD_N = 0.70, 5.0, 30

GATES = [None, 0.55, 0.60, 0.65, 0.70]
TPS = [0.005, 0.0075, 0.010, 0.015]
SLS = [0.010, 0.015, 0.020, 0.030]
GEOMS = [(tp, sl) for tp in TPS for sl in SLS if tp <= sl]   # 15 pairs
TIMEOUTS = [12, 24, 48]
B_TRIGGERS = [0.0, 0.003, 0.005, 0.010]
C_DIPS = [0.01, 0.02, 0.03]
N_BOOT = 1000
N_RANDOM = 200
SEED = 42

COST = CostModel(fee_bp=1.0, slippage_bp=2.0)

CRASH_NOTE = (
    "崩盘压测口径（1h 近似）：5m 规则按如下映射移植到 1h 条上——"
    "timeout 换算 max(1, round(timeout/12)) 根 1h 条；结算同 settle_bracket（悲观），日内强平；"
    "A 族的 5m V 检测与 GBDT 门在 1h/2021 年无法复现，用『上一根 1h 日内收益 <= -0.5% 后做多』近似"
    "（对应事件表 X=0.5% 的跌幅触发，无门控，故为几何压力的下界口径）；"
    "B 族用上一根完整 1h 条日内收益做信号（与 5m 版同义）；C 族用 1h Close 对当日 High 累计最高的回撤触发。"
)


# ------------------------------------------------------------- bar containers
class BarSet:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars
        self.idx = bars.index
        self.n = len(bars)
        self.opens = bars["Open"].to_numpy()
        self.highs = bars["High"].to_numpy()
        self.lows = bars["Low"].to_numpy()
        self.closes = bars["Close"].to_numpy()
        et_idx = self.idx.tz_convert(ET)
        self.et_dates = np.asarray(et_idx.date)
        self.et_min = et_idx.hour * 60 + et_idx.minute
        self.cache: dict = {}

    def eval_trade(self, epos: int, direction: int, tp: float, sl: float,
                   timeout: int):
        """Enter at open of bar `epos`; settle pessimistically; day-flat.
        Returns (net_ret, exit_pos) or None if not tradable."""
        if epos >= self.n:
            return None
        key = (epos, direction, tp, sl, timeout)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        end = min(epos + timeout, self.n)
        day = self.et_dates[epos]
        off = np.nonzero(self.et_dates[epos:end] != day)[0]  # day-end cut
        if len(off):
            end = epos + int(off[0])
        window = self.bars.iloc[epos:end]
        if window.empty:
            self.cache[key] = None
            return None
        entry_px = entry_fill(self.opens[epos], direction, COST)
        if direction == 1:
            tp_px, sl_px = entry_px * (1 + tp), entry_px * (1 - sl)
        else:
            tp_px, sl_px = entry_px * (1 - tp), entry_px * (1 + sl)
        res = settle_bracket(window, direction, entry_px, tp_px, sl_px, COST)
        exit_pos = epos + window.index.get_loc(res.exit_time)
        out = (res.ret, exit_pos)
        self.cache[key] = out
        return out


def run_combo(bs: BarSet, signals: list[tuple[int, int]], tp: float, sl: float,
              timeout: int) -> pd.DataFrame:
    """signals = sorted [(entry_pos, direction)]; one position at a time."""
    rows = []
    busy_until = -1
    for epos, d in signals:
        if epos <= busy_until:
            continue
        tr = bs.eval_trade(epos, d, tp, sl, timeout)
        if tr is None:
            continue
        ret, exit_pos = tr
        busy_until = exit_pos
        rows.append((epos, bs.et_dates[epos], d, ret))
    return pd.DataFrame(rows, columns=["epos", "et_day", "dir", "ret"])


def seg_stats(trades: pd.DataFrame, prefix: str) -> dict:
    n = len(trades)
    d = {f"{prefix}_n": n}
    if n == 0:
        d.update({f"{prefix}_wr": np.nan, f"{prefix}_avg_bp": np.nan,
                  f"{prefix}_mdd": np.nan, f"{prefix}_total": np.nan})
        return d
    r = trades["ret"].to_numpy()
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    d[f"{prefix}_wr"] = float((r > 0).mean())
    d[f"{prefix}_avg_bp"] = float(r.mean() * 1e4)
    d[f"{prefix}_mdd"] = float((eq / peak - 1).min())
    d[f"{prefix}_total"] = float(eq[-1] - 1)
    return d


# ------------------------------------------------------------ signal builders
def signals_A(bs: BarSet, scores: pd.DataFrame, gate: float | None,
              session: bool) -> list[tuple[int, int]]:
    sub = scores if gate is None else scores[scores["prob"] >= gate]
    pos = sub["epos"].to_numpy()
    if session:
        pos = pos[(bs.et_min[pos] >= 630) & (bs.et_min[pos] < 900)]
    return [(int(p), 1) for p in np.sort(pos)]


def build_hourly_signals(bs: BarSet) -> pd.DataFrame:
    """One row per complete 1h bucket: entry_pos (first 5m bar after bucket
    end, same ET day) + bucket intraday return."""
    hourly = resample_bars(bs.bars, 60)
    rows = []
    for t, row in hourly.iterrows():
        end_t = t + pd.Timedelta(minutes=60)
        p = int(np.searchsorted(bs.idx.values, np.datetime64(end_t)))
        if p >= bs.n:
            continue
        if bs.et_dates[p] != t.tz_convert(ET).date():
            continue
        rows.append((p, float(row["Close"] / row["Open"] - 1.0)))
    return pd.DataFrame(rows, columns=["epos", "sig_ret"])


def signals_B(bs: BarSet, hourly_sig: pd.DataFrame, trigger: float,
              mode: str, session: bool) -> list[tuple[int, int]]:
    sub = hourly_sig[hourly_sig["sig_ret"].abs() >= trigger]
    sub = sub[sub["sig_ret"] != 0.0]
    out = []
    for _, r in sub.iterrows():
        d = 1 if r["sig_ret"] > 0 else -1
        if mode == "fade":
            d = -d
        p = int(r["epos"])
        if session and not (630 <= bs.et_min[p] < 900):
            continue
        out.append((p, d))
    return sorted(out)


def dip_entry_positions(bs: BarSet, d: float) -> np.ndarray:
    """Bars whose Close crosses below runmax(High)*(1-d); entry = next bar,
    same ET day. Recomputed per day; re-triggers allowed on new crossings."""
    entries = []
    n = bs.n
    i = 0
    while i < n:
        day = bs.et_dates[i]
        j = i
        runmax = -np.inf
        below_prev = False
        while j < n and bs.et_dates[j] == day:
            runmax = max(runmax, bs.highs[j])
            below = bs.closes[j] <= runmax * (1 - d)
            if below and not below_prev and j + 1 < n and bs.et_dates[j + 1] == day:
                entries.append(j + 1)
            below_prev = below
            j += 1
        i = j
    return np.asarray(entries, dtype=int)


def signals_C(bs: BarSet, dip_pos: np.ndarray, session: bool) -> list[tuple[int, int]]:
    pos = dip_pos
    if session:
        pos = pos[(bs.et_min[pos] >= 630) & (bs.et_min[pos] < 900)]
    return [(int(p), 1) for p in pos]


# ---------------------------------------------------------------- evaluation
def evaluate(bs: BarSet, combo: dict, signal_fn) -> dict:
    trades = run_combo(bs, signal_fn(combo), combo["tp"], combo["sl"], combo["timeout"])
    tr = trades[trades["et_day"] < HOLDOUT_START]
    ho = trades[trades["et_day"] >= HOLDOUT_START]
    out = dict(combo)
    out.update(seg_stats(tr, "train"))
    out.update(seg_stats(ho, "holdout"))
    out["_holdout_rets"] = ho["ret"].to_numpy()
    return out


def bootstrap_p(rets: np.ndarray, rng: np.random.Generator) -> float:
    """One-sided p that the holdout expectation <= 0."""
    if len(rets) == 0:
        return np.nan
    means = rng.choice(rets, size=(N_BOOT, len(rets)), replace=True).mean(axis=1)
    return float((np.sum(means <= 0) + 1) / (N_BOOT + 1))


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    bars = load_bars(str(BARS_5M))
    bs = BarSet(bars)

    scores = pd.read_csv(SCORES, parse_dates=["entry_t"])
    scores = scores[scores["prob"].notna()].copy()
    epos = bars.index.get_indexer(pd.DatetimeIndex(scores["entry_t"]))
    scores["epos"] = epos
    scores = scores[scores["epos"] >= 0].sort_values("epos").reset_index(drop=True)
    print(f"family A signal pool (OOF-scored): {len(scores)}")

    hourly_sig = build_hourly_signals(bs)
    print(f"family B hourly buckets usable: {len(hourly_sig)}")
    dip_pos = {d: dip_entry_positions(bs, d) for d in C_DIPS}
    for d, p in dip_pos.items():
        print(f"family C dip {d:.0%} crossings: {len(p)}")

    # ---------------- build the base combo grid (registered up front) --------
    combos: list[dict] = []

    def reg(family, session=False, **params):
        combos.append({"family": family, "session_gate": session, **params})

    for g in GATES:
        for tp, sl in GEOMS:
            for to in TIMEOUTS:
                reg("A", gate=np.nan if g is None else g, tp=tp, sl=sl, timeout=to)
    for mode in ("follow", "fade"):
        for trig in B_TRIGGERS:
            for tp, sl in GEOMS:
                for to in TIMEOUTS:
                    reg("B", mode=mode, trigger=trig, tp=tp, sl=sl, timeout=to)
    for d in C_DIPS:
        for tp, sl in GEOMS:
            for to in TIMEOUTS:
                reg("C", dip=d, tp=tp, sl=sl, timeout=to)
    n_base = len(combos)
    print(f"base combos registered: {n_base}")

    def make_signal_fn(c):
        fam = c["family"]
        if fam == "A":
            g = None if (isinstance(c.get("gate"), float) and np.isnan(c["gate"])) or c.get("gate") is None else c["gate"]
            return lambda cc, g=g: signals_A(bs, scores, g, cc["session_gate"])
        if fam == "B":
            return lambda cc: signals_B(bs, hourly_sig, cc["trigger"], cc["mode"], cc["session_gate"])
        pos = dip_pos.get(c.get("dip"))
        if pos is None:
            pos = dip_entry_positions(bs, c["dip"])
        return lambda cc, pos=pos: signals_C(bs, pos, cc["session_gate"])

    results: list[dict] = []
    for k, c in enumerate(combos):
        results.append(evaluate(bs, c, make_signal_fn(c)))
        if (k + 1) % 100 == 0:
            print(f"  evaluated {k + 1}/{len(combos)} base combos "
                  f"({time.time() - t0:.0f}s, cache={len(bs.cache)})")

    # ---------------- session-gate overlay on top train combos per family ----
    df0 = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in results])
    overlay_src = []
    for fam in ("A", "B", "C"):
        sub = df0[(df0["family"] == fam) & (df0["train_n"] >= TRAIN_N)]
        top = set(sub.nlargest(10, "train_avg_bp").index) | set(sub.nlargest(10, "train_wr").index)
        overlay_src.extend(sorted(top))
    print(f"overlay variants: {len(overlay_src)}")
    for i in overlay_src:
        c = dict(combos[i])
        c["session_gate"] = True
        combos.append(c)
        results.append(evaluate(bs, c, make_signal_fn(c)))

    n_total = len(combos)
    print(f"total combos registered: {n_total} ({time.time() - t0:.0f}s)")

    all_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                           for r in results])
    all_df.insert(0, "combo_id", range(len(all_df)))
    for r, cid in zip(results, all_df["combo_id"]):
        r["combo_id"] = cid

    # ---------------- protocol: train screen -> candidates -------------------
    cand_mask = ((all_df["train_wr"] >= TRAIN_WR)
                 & (all_df["train_avg_bp"] >= TRAIN_BP)
                 & (all_df["train_n"] >= TRAIN_N))
    all_df["is_candidate"] = cand_mask
    cand = all_df[cand_mask].copy()
    N_cand = len(cand)
    print(f"candidate set (train screen): N = {N_cand}")

    # ---------------- candidates on holdout + bootstrap ----------------------
    hold_rows = []
    for _, row in cand.iterrows():
        r = results[int(row["combo_id"])]
        p = bootstrap_p(r["_holdout_rets"], rng)
        passed = (row["holdout_n"] >= HOLD_N
                  and row["holdout_wr"] >= HOLD_WR
                  and row["holdout_avg_bp"] >= HOLD_BP)
        d = row.to_dict()
        d["boot_p"] = p
        d["bonferroni_pN"] = p * N_cand if np.isfinite(p) else np.nan
        d["holdout_pass"] = bool(passed)
        hold_rows.append(d)
    hold_df = pd.DataFrame(hold_rows)
    qualifiers = hold_df[hold_df["holdout_pass"]] if len(hold_df) else pd.DataFrame()
    all_df["holdout_pass"] = False
    if len(hold_df):
        all_df.loc[all_df["combo_id"].isin(
            hold_df.loc[hold_df["holdout_pass"], "combo_id"]), "holdout_pass"] = True

    # ---------------- Pareto frontier on holdout (descriptive) ---------------
    par_pool = all_df[all_df["holdout_n"] >= HOLD_N].copy()
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
               (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
    par_rows = []
    for lo, hi in buckets:
        b = par_pool[(par_pool["holdout_wr"] >= lo) & (par_pool["holdout_wr"] < hi)]
        if len(b):
            best = b.loc[b["holdout_avg_bp"].idxmax()].to_dict()
            best["wr_bucket"] = f"[{lo:.0%},{hi:.0%})"
            par_rows.append(best)
    pareto = pd.DataFrame(par_rows)

    # ---------------- perturbation (+-10%) for qualifiers --------------------
    perturb_lines = []
    for _, q in qualifiers.iterrows():
        c0 = dict(combos[int(q["combo_id"])])
        variants = []
        for f in (0.9, 1.1):
            variants.append({**c0, "tp": c0["tp"] * f, "_lab": f"tp*{f}"})
            variants.append({**c0, "sl": c0["sl"] * f, "_lab": f"sl*{f}"})
            if c0["family"] == "A" and np.isfinite(c0.get("gate", np.nan)):
                variants.append({**c0, "gate": c0["gate"] * f, "_lab": f"gate*{f}"})
            elif c0["family"] == "B" and c0["trigger"] > 0:
                variants.append({**c0, "trigger": c0["trigger"] * f, "_lab": f"trigger*{f}"})
            elif c0["family"] == "C":
                variants.append({**c0, "dip": c0["dip"] * f, "_lab": f"dip*{f}"})
        ok = 0
        details = []
        for v in variants:
            lab = v.pop("_lab")
            r = evaluate(bs, v, make_signal_fn(v))
            still = (r["holdout_n"] >= HOLD_N and r["holdout_wr"] >= HOLD_WR
                     and r["holdout_avg_bp"] >= HOLD_BP)
            ok += still
            details.append(f"    {lab}: n={r['holdout_n']} wr={r['holdout_wr']:.1%} "
                           f"avg={r['holdout_avg_bp']:+.1f}bp {'PASS' if still else 'fail'}")
        perturb_lines.append(
            f"combo {int(q['combo_id'])}: {ok}/{len(variants)} 扰动仍达标\n" + "\n".join(details))

    # ---------------- random-entry baselines ---------------------------------
    focus = pd.concat([qualifiers, pareto], ignore_index=True) if len(pareto) or len(qualifiers) else pd.DataFrame()
    rb_rows = []
    if len(focus):
        hold_pool = np.nonzero(bs.et_dates >= HOLDOUT_START)[0]
        seen = set()
        for _, f in focus.iterrows():
            direction = 1
            if f.get("family") == "B":
                direction = 0  # mixed long/short; baseline runs both, see below
            key = (f.get("family"), direction, f["tp"], f["sl"], int(f["timeout"]),
                   int(f["holdout_n"]))
            if key in seen or f["holdout_n"] < 1:
                continue
            seen.add(key)
            n_tr = int(f["holdout_n"])
            dirs = [1] if f.get("family") != "B" else [1, -1]
            for d in dirs:
                wrs, avgs = [], []
                for _ in range(N_RANDOM):
                    picks = np.sort(rng.choice(hold_pool, size=min(n_tr, len(hold_pool)), replace=False))
                    rets = []
                    for p in picks:
                        tr = bs.eval_trade(int(p), d, f["tp"], f["sl"], int(f["timeout"]))
                        if tr is not None:
                            rets.append(tr[0])
                    rets = np.asarray(rets)
                    if len(rets):
                        wrs.append((rets > 0).mean())
                        avgs.append(rets.mean() * 1e4)
                wrs, avgs = np.asarray(wrs), np.asarray(avgs)
                rb_rows.append({
                    "family": f.get("family"), "dir": d, "tp": f["tp"], "sl": f["sl"],
                    "timeout": int(f["timeout"]), "n_trades": n_tr,
                    "real_holdout_wr": f["holdout_wr"], "real_holdout_avg_bp": f["holdout_avg_bp"],
                    "rand_wr_mean": wrs.mean(), "rand_wr_p95": np.percentile(wrs, 95),
                    "rand_avg_bp_mean": avgs.mean(), "rand_avg_bp_p95": np.percentile(avgs, 95),
                    "pct_rand_wr_ge_real": float((wrs >= f["holdout_wr"]).mean()),
                    "pct_rand_avg_ge_real": float((avgs >= f["holdout_avg_bp"]).mean()),
                })
    rb_df = pd.DataFrame(rb_rows)

    # ---------------- crash stress (1h approximation) ------------------------
    bars1h = load_bars(str(BARS_1H))
    b1 = BarSet(bars1h)
    crash_mask = (b1.et_dates >= CRASH_START) & (b1.et_dates <= CRASH_END)
    crash_pos = set(np.nonzero(crash_mask)[0].tolist())

    def hourly_prev_ret_signals(thresh_neg=None, trigger=None, mode=None):
        out = []
        for p in range(1, b1.n):
            if p not in crash_pos or (p - 1) not in crash_pos:
                continue
            if b1.et_dates[p] != b1.et_dates[p - 1]:
                continue
            r = b1.closes[p - 1] / b1.opens[p - 1] - 1.0
            if thresh_neg is not None:            # family A approx
                if r <= thresh_neg:
                    out.append((p, 1))
            else:                                  # family B
                if abs(r) >= trigger and r != 0:
                    d = 1 if r > 0 else -1
                    out.append((p, d if mode == "follow" else -d))
        return out

    def crash_dip_signals(dpc):
        pos = dip_entry_positions(b1, dpc)
        return [(int(p), 1) for p in pos if p in crash_pos]

    crash_lines = []
    stress_pool = qualifiers if len(qualifiers) else pareto.nlargest(min(3, len(pareto)), "holdout_avg_bp") if len(pareto) else pd.DataFrame()
    stress_tag = "达标者" if len(qualifiers) else "Pareto top（信息性，无人达标）"
    for _, q in stress_pool.iterrows():
        to1 = max(1, round(int(q["timeout"]) / 12))
        fam = q["family"]
        if fam == "A":
            sig = hourly_prev_ret_signals(thresh_neg=-0.005)
        elif fam == "B":
            sig = hourly_prev_ret_signals(trigger=q["trigger"], mode=q["mode"])
        else:
            sig = crash_dip_signals(q["dip"])
        if q.get("session_gate", False):
            sig = [(p, d) for p, d in sig if 630 <= b1.et_min[p] < 900]
        trades = run_combo(b1, sig, q["tp"], q["sl"], to1)
        st = seg_stats(trades, "crash")
        crash_lines.append(
            f"combo {int(q['combo_id'])} [{fam}] tp={q['tp']:.2%}/sl={q['sl']:.2%}/to={int(q['timeout'])}bar"
            f" -> 1h近似: n={st['crash_n']}, wr={st['crash_wr']:.1%}" if st['crash_n'] else
            f"combo {int(q['combo_id'])} [{fam}]: 崩盘窗口无信号")
        if st["crash_n"]:
            crash_lines[-1] += (f", avg={st['crash_avg_bp']:+.1f}bp, "
                                f"MDD={st['crash_mdd']:.1%}, total={st['crash_total']:+.1%}")

    # ---------------- outputs -------------------------------------------------
    all_df.to_csv(OUT / "all_combos.csv", index=False)
    cand.to_csv(OUT / "candidates_train.csv", index=False)
    hold_df.to_csv(OUT / "holdout_results.csv", index=False)
    pareto.to_csv(OUT / "pareto.csv", index=False)
    rb_df.to_csv(OUT / "random_baseline.csv", index=False)

    # gap analysis
    gap_lines = []
    g1 = par_pool[(par_pool["holdout_wr"] >= HOLD_WR)]
    if len(g1):
        b = g1.loc[g1["holdout_avg_bp"].idxmax()]
        gap_lines.append(
            f"留出段 WR>=70% & n>=30 中期望最高: combo {int(b['combo_id'])} [{b['family']}] "
            f"tp={b['tp']:.2%}/sl={b['sl']:.2%}/to={int(b['timeout'])} -> "
            f"n={int(b['holdout_n'])}, wr={b['holdout_wr']:.1%}, avg={b['holdout_avg_bp']:+.1f}bp "
            f"(距 +5bp 差 {5 - b['holdout_avg_bp']:.1f}bp)")
    else:
        gap_lines.append("留出段没有任何 WR>=70% 且 n>=30 的组合。")
    g2 = par_pool[(par_pool["holdout_avg_bp"] >= HOLD_BP)]
    if len(g2):
        b = g2.loc[g2["holdout_wr"].idxmax()]
        gap_lines.append(
            f"留出段 avg>=+5bp & n>=30 中胜率最高: combo {int(b['combo_id'])} [{b['family']}] "
            f"tp={b['tp']:.2%}/sl={b['sl']:.2%}/to={int(b['timeout'])} -> "
            f"n={int(b['holdout_n'])}, wr={b['holdout_wr']:.1%}, avg={b['holdout_avg_bp']:+.1f}bp "
            f"(距 70% 胜率差 {0.70 - b['holdout_wr']:.1%})")
    else:
        gap_lines.append("留出段没有任何 avg>=+5bp 且 n>=30 的组合。")

    L = []
    L.append("E9 前沿搜索 — 摘要（协议先行，悲观结算，全程 settle_bracket + 1bp fee/2bp slip 单边）")
    L.append(f"数据: {BARS_5M.name} ({len(bars)} 根 5m, "
             f"{bs.et_dates[0]} .. {bs.et_dates[-1]}); 切分: 训练 < {HOLDOUT_START} <= 留出")
    L.append(f"登记组合总数: {n_total}（基础 {n_base} + 时段叠加 {n_total - n_base}）")
    L.append(f"训练段初筛（WR>={TRAIN_WR:.0%} & avg>=+{TRAIN_BP:.0f}bp & n>={TRAIN_N}）: 候选 N = {N_cand}")
    L.append("")
    if N_cand:
        L.append("== 候选集留出段结果 ==")
        cols = ["combo_id", "family", "gate", "mode", "trigger", "dip", "tp", "sl",
                "timeout", "session_gate", "train_n", "train_wr", "train_avg_bp",
                "holdout_n", "holdout_wr", "holdout_avg_bp", "boot_p", "bonferroni_pN",
                "holdout_pass"]
        cols = [c for c in cols if c in hold_df.columns]
        L.append(hold_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    L.append("")
    nq = int(len(qualifiers))
    L.append(f"== 达标判定 == 留出段达标（WR>=70% & avg>=+5bp & n>=30）: {nq} 个")
    if nq:
        L.append("多重比较：p 值须按 Bonferroni p*N 解读（N=候选集大小，上表已列）。")
        L.append("")
        L.append("== ±10% 扰动 ==")
        L.extend(perturb_lines)
    else:
        L.append("无人达标。以下为描述性 Pareto 前沿（留出段，n>=30，注意 700+ 组合的多重比较背景，")
        L.append("这些数字是『事后挑最好』，只用于差距分析，不构成任何 edge 证据）。")
        if len(pareto):
            pc = ["wr_bucket", "combo_id", "family", "gate", "mode", "trigger", "dip",
                  "tp", "sl", "timeout", "session_gate", "holdout_n", "holdout_wr",
                  "holdout_avg_bp", "train_n", "train_wr", "train_avg_bp"]
            pc = [c for c in pc if c in pareto.columns]
            L.append(pareto[pc].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        L.append("")
        L.append("== 差距分析 ==")
        L.extend(gap_lines)
    L.append("")
    L.append("== 随机入场基线（同几何，200 次，留出段） ==")
    if len(rb_df):
        L.append(rb_df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        L.append("判读：若 pct_rand_wr_ge_real / pct_rand_avg_ge_real 不小（>5-10%），")
        L.append("说明该组合的胜率/期望大体是几何（tp<<sl + 日内漂移）的产物，不含选时信息。")
    else:
        L.append("（无达标者且 Pareto 为空，未跑基线）")
    L.append("")
    L.append(f"== 崩盘压测（{stress_tag}，2021-11..2023-01） ==")
    L.append(CRASH_NOTE)
    L.extend(crash_lines if crash_lines else ["（无对象）"])
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s；结算缓存 {len(bs.cache)} 条。")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
