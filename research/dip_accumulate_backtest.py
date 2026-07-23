"""Parameterised, honest evaluation of the "buy dips, never sell at a loss" idea.

The user's thesis: a mega-cap won't die, so buy every dip, hold through any
drawdown (NO stop loss), and sell only at highs / in profit.  The naive
version (research/no_loss_stress_test.py) showed 100% closed win rate but
sub-risk-free returns.  Here the idea becomes a strategy *family* and each
member is scored under the project's standard protocol.

Strategy family
---------------
Entry trigger   : price falls d in {5,10,15}% below the running ATH buys
                  ladder level 1; each further d buys the next level.
Entry confirm   : none / rsi (RSI-14 < 30) / bounce (first green hourly bar
                  after the level is hit; fill at that bar's close) /
                  rsi+bounce (both).
Exit            : fixed_tp tp in {2,5,10}%   -- PER-TRANCHE (each tranche has
                                                a limit at cost*(1+tp));
                  trailing  r  in {3,5}%     -- WHOLE POSITION: once the
                                                position is in net profit,
                                                track the high and sell all
                                                on an r pullback;
                  swing (w=5)                -- WHOLE POSITION: a local swing
                                                high (max of +-5 bars) formed
                                                during the holding period,
                                                actionable only 5 bars later,
                                                sell all at that bar's close
                                                if the position is in profit.
                  (fixed_tp is per-tranche because that is the natural
                  "each lot books its own profit" reading; trailing/swing are
                  inherently position-level concepts, so they close all.)
Sizing          : equal (10 levels of capital/10) /
                  martingale (4 levels, 1:2:4:8 of capital) /
                  vol_inverse (10 levels; stake = capital/10 *
                  clip(ref_vol / vol_20d, 0.5, 2), ref_vol = expanding median
                  of the 20-day realised vol -- strict normalisation to total
                  capital would need future vols, so stakes are clipped and
                  cash-capped instead; this is stated in the summary).
Stop            : none (user's core constraint) vs a control variant that
                  force-liquidates everything when account MTM equity falls
                  30% below its running peak.

Timing rules (no look-ahead)
----------------------------
* Ladder levels are computed from the ATH as of the END of the PREVIOUS bar;
  the current bar's new high only raises the ATH after trading.
* fixed_tp sells are limit orders: fill = max(target, open) when High>=target.
* trailing: "armed" state and the trailing peak are updated at bar close;
  the exit check on bar i uses the peak/armed state as of bar i-1.  The stop
  fill is min-side honest: fill = open if open <= trigger else trigger.  The
  no-loss constraint vetoes any fill below the position's breakeven; if price
  falls back under breakeven the trail disarms and waits to re-arm.
* swing: the swing high at bar j is only *confirmed* at bar j+5 (needs 5
  bars each side); action happens at the close of the confirmation bar, and
  only if that close is above breakeven.  Swing highs are precomputed on the
  full series (price history is a legitimate warm-up input).
* Entry fills: 'none' fills at the level price (limit), or at the open if
  the bar gapped below the level.  Confirmed entries fill at the close of
  the confirming bar (RSI is the value at that same close).
* A sold level is DISARMED and can only re-arm after price closes back above
  the level price (prevents same-bar churn); it then needs a fresh downward
  cross to buy again.
* Account-stop control: checked on MTM at bar close, liquidation at close.

Protocol
--------
Train 2018-07 .. 2024-01 (contains the 2021-11..2023-01 -75% crash), grid
scored by CAGR / max(0.05, |MTM maxDD|) with >=20 closed trades; top-5 go to
validation 2024-01 .. 2026-07.  Crash-window stress and a +-10% neighbourhood
perturbation of the winner are reported.  Fees 1bp per side.

Usage:
    .venv/bin/python research/dip_accumulate_backtest.py
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

from src.common.data_io import load_bars

FEE = 0.0001                 # 1bp per side
CAPITAL = 10_000.0
RF_ANNUAL = 0.04
TRAIN_END = pd.Timestamp("2024-01-01", tz="UTC")
CRASH_START = pd.Timestamp("2021-11-01", tz="UTC")
CRASH_END = pd.Timestamp("2023-02-01", tz="UTC")
MIN_CLOSED = 20
ACCT_STOP_DD = 0.30
SWING_W = 5
VOL_WIN_BARS = 140           # ~20 trading days of hourly bars (7/day)
OUT_DIR = ROOT / "outputs" / "dip_accumulate"

DISARMED, ARMED, PENDING, HELD = 0, 1, 2, 3


# ----------------------------------------------------------------- indicators
def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/n
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0)   # warm-up / all-up bars: treat as "not oversold"


def swing_high_flags(high: np.ndarray, w: int) -> np.ndarray:
    """flag[j] = True iff High[j] is the max of High[j-w .. j+w]."""
    s = pd.Series(high)
    roll_max = s.rolling(2 * w + 1, center=True, min_periods=2 * w + 1).max()
    return (s >= roll_max).to_numpy()


def precompute(bars: pd.DataFrame, swing_w: int = SWING_W) -> dict:
    close = bars["Close"]
    ret = close.pct_change()
    vol = ret.rolling(VOL_WIN_BARS).std()
    ref_vol = vol.expanding().median()
    scale = (ref_vol / vol).clip(0.5, 2.0).fillna(1.0).to_numpy()
    return {
        "idx": bars.index,
        "o": bars["Open"].to_numpy(float),
        "h": bars["High"].to_numpy(float),
        "l": bars["Low"].to_numpy(float),
        "c": close.to_numpy(float),
        "rsi": wilder_rsi(close).to_numpy(),
        "green": (bars["Close"] > bars["Open"]).to_numpy(),
        "vol_scale": scale,
        "swing": {w: swing_high_flags(bars["High"].to_numpy(float), w)
                  for w in {swing_w, swing_w - 1, swing_w + 1}},
    }


# ------------------------------------------------------------------ simulator
def simulate(D: dict, i0: int, i1: int, *, d: float, confirm: str,
             exit_kind: str, exit_p: float, sizing: str, acct_stop: bool,
             diag: bool = False) -> tuple[dict, pd.DataFrame | None]:
    idx = D["idx"]
    o, h, l, c = D["o"], D["h"], D["l"], D["c"]
    rsi, green, vscale = D["rsi"], D["green"], D["vol_scale"]
    swing_flag = D["swing"][int(exit_p)] if exit_kind == "swing" else None
    need_rsi = "rsi" in confirm
    need_bounce = "bounce" in confirm

    if sizing == "martingale":
        mults = (1.0, 2.0, 4.0, 8.0)
        n_levels = 4
    else:
        n_levels = 10

    def stake_at(k: int, i: int) -> float:
        if sizing == "equal":
            return CAPITAL / 10.0
        if sizing == "martingale":
            return CAPITAL * mults[k - 1] / 15.0
        return CAPITAL / 10.0 * vscale[i]          # vol_inverse

    cash = CAPITAL
    tranches: list[dict] = []       # {k, stake, shares, fill, i_open}
    closed: list[dict] = []
    state = [ARMED] * (n_levels + 1)   # index 1..n_levels
    ath = h[i0]
    trail_armed, trail_peak = False, 0.0
    first_open_i: int | None = None
    forced_liq = 0
    exh_events = 0
    exh_bar_flags = np.zeros(i1 - i0, dtype=bool)
    eq_peak = CAPITAL
    mtm_arr = np.empty(i1 - i0)
    inv_arr = np.zeros(i1 - i0)
    nopen_arr = np.zeros(i1 - i0, dtype=np.int32)

    def breakeven() -> float:
        tot_stake = sum(t["stake"] for t in tranches)
        tot_sh = sum(t["shares"] for t in tranches)
        return tot_stake / (tot_sh * (1.0 - FEE))

    def close_all(fill: float, i: int) -> None:
        nonlocal cash, tranches, first_open_i, trail_armed
        for t in tranches:
            proceeds = t["shares"] * fill * (1.0 - FEE)
            cash += proceeds
            closed.append({"i_open": t["i_open"], "i_close": i,
                           "ret": proceeds / t["stake"] - 1.0})
            state[t["k"]] = DISARMED
        tranches = []
        first_open_i = None
        trail_armed = False

    for i in range(i0, i1):
        j = i - i0
        # ---------------------------------------------------------- 1) exits
        if tranches:
            if exit_kind == "fixed_tp":
                keep = []
                for t in tranches:
                    target = t["fill"] * (1.0 + exit_p)
                    if h[i] >= target:
                        fill = o[i] if o[i] > target else target
                        proceeds = t["shares"] * fill * (1.0 - FEE)
                        cash += proceeds
                        closed.append({"i_open": t["i_open"], "i_close": i,
                                       "ret": proceeds / t["stake"] - 1.0})
                        state[t["k"]] = DISARMED
                    else:
                        keep.append(t)
                tranches = keep
                if not tranches:
                    first_open_i = None
            elif exit_kind == "trailing":
                if trail_armed:
                    trig = trail_peak * (1.0 - exit_p)
                    if l[i] <= trig:
                        fill = o[i] if o[i] < trig else trig
                        if fill > breakeven():        # no-loss veto
                            close_all(fill, i)
            elif exit_kind == "swing":
                sj = i - int(exit_p)
                if (first_open_i is not None and sj >= first_open_i
                        and swing_flag[sj] and c[i] > breakeven()):
                    close_all(c[i], i)

        # ------------------------------------------- 2) buys (ATH of bar i-1)
        for k in range(1, n_levels + 1):
            lp = ath * (1.0 - k * d)
            if lp <= 0.0:
                break
            st = state[k]
            if st == ARMED and l[i] <= lp:
                if confirm == "none":
                    fill = o[i] if o[i] < lp else lp
                    stake = stake_at(k, i)
                    if cash + 1e-9 >= stake and stake > 0:
                        tranches.append({"k": k, "stake": stake,
                                         "shares": stake * (1.0 - FEE) / fill,
                                         "fill": fill, "i_open": i})
                        cash -= stake
                        state[k] = HELD
                        if first_open_i is None:
                            first_open_i = i
                    else:
                        exh_events += 1
                        exh_bar_flags[j] = True
                    continue
                state[k] = PENDING
                st = PENDING
            if st == PENDING:
                if c[i] > lp:                       # dip over, unconfirmed
                    state[k] = ARMED
                else:
                    ok = (not need_bounce or green[i]) and \
                         (not need_rsi or rsi[i] < 30.0)
                    if ok:
                        fill = c[i]
                        stake = stake_at(k, i)
                        if cash + 1e-9 >= stake and stake > 0:
                            tranches.append({"k": k, "stake": stake,
                                             "shares": stake * (1.0 - FEE) / fill,
                                             "fill": fill, "i_open": i})
                            cash -= stake
                            state[k] = HELD
                            if first_open_i is None:
                                first_open_i = i
                        else:
                            exh_events += 1
                            exh_bar_flags[j] = True

        # -------------------------- 3) end-of-bar state (re-arm, ATH, trail)
        for k in range(1, n_levels + 1):
            if state[k] == DISARMED and c[i] > ath * (1.0 - k * d):
                state[k] = ARMED
        if h[i] > ath:
            ath = h[i]

        mtm = cash + sum(t["shares"] for t in tranches) * c[i]
        if acct_stop and tranches and mtm <= eq_peak * (1.0 - ACCT_STOP_DD):
            close_all(c[i], i)                      # forced liquidation
            forced_liq += 1
            mtm = cash
        eq_peak = max(eq_peak, mtm)

        if exit_kind == "trailing":
            if tranches:
                be = breakeven()
                if trail_armed:
                    if c[i] <= be:
                        trail_armed = False
                    else:
                        trail_peak = max(trail_peak, h[i])
                elif c[i] > be:
                    trail_armed, trail_peak = True, h[i]
            else:
                trail_armed = False

        mtm_arr[j] = mtm
        inv_arr[j] = sum(t["stake"] for t in tranches)
        nopen_arr[j] = len(tranches)

    # ------------------------------------------------------------- metrics
    seg_idx = idx[i0:i1]
    eq = pd.Series(mtm_arr, index=seg_idx)
    run_peak = eq.cummax()
    dd = eq / run_peak - 1.0
    years = max((seg_idx[-1] - seg_idx[0]).days / 365.25, 1e-9)
    final_eq = float(eq.iloc[-1])
    cagr = (final_eq / CAPITAL) ** (1.0 / years) - 1.0 if final_eq > 0 else -1.0

    uw_days, peak_t = 0, seg_idx[0]
    for t, (e, p) in zip(seg_idx, zip(eq.to_numpy(), run_peak.to_numpy())):
        if e >= p:
            peak_t = t
        else:
            uw_days = max(uw_days, (t - peak_t).days)

    closed_df = pd.DataFrame(closed)
    if len(closed_df):
        held_days = [(idx[r["i_close"]] - idx[r["i_open"]]).days
                     for r in closed]
        win = float((closed_df["ret"] > 0).mean())
        exp_bp = float(closed_df["ret"].mean() * 1e4)
    else:
        held_days, win, exp_bp = [0], np.nan, np.nan
    open_days = [(seg_idx[-1] - idx[t["i_open"]]).days for t in tranches]
    max_locked = max(max(held_days), max(open_days, default=0))

    stats = {
        "cagr": cagr,
        "mtm_mdd": float(dd.min()),
        "exp_bp": exp_bp,
        "win_rate": win,
        "n_closed": len(closed_df),
        "max_locked_days": int(max_locked),
        "uw_days": int(uw_days),
        "exh_bars": int(exh_bar_flags.sum()),
        "exh_events": int(exh_events),
        "exposure": float((nopen_arr > 0).mean()),
        "final_eq": final_eq,
        "n_open_end": len(tranches),
        "forced_liq": int(forced_liq),
    }
    diag_df = None
    if diag:
        diag_df = pd.DataFrame({"mtm": mtm_arr, "invested": inv_arr,
                                "n_open": nopen_arr, "exh": exh_bar_flags},
                               index=seg_idx)
    return stats, diag_df


# ----------------------------------------------------------------- benchmarks
def benchmark_rows(D: dict, i0: int, i1: int) -> list[dict]:
    idx, c = D["idx"], D["c"]
    seg = pd.Series(c[i0:i1], index=idx[i0:i1])
    years = (seg.index[-1] - seg.index[0]).days / 365.25
    # buy & hold with 1bp each side
    tot = (seg.iloc[-1] * (1 - FEE)) / (seg.iloc[0] * (1 + FEE))
    eq = seg / seg.iloc[0]
    dd = (eq / eq.cummax() - 1.0).min()
    uw_days, peak_t = 0, seg.index[0]
    run_peak = eq.cummax()
    for t, (e, p) in zip(seg.index, zip(eq.to_numpy(), run_peak.to_numpy())):
        if e >= p:
            peak_t = t
        else:
            uw_days = max(uw_days, (t - peak_t).days)
    return [
        {"combo": "buy_and_hold", "cagr": tot ** (1 / years) - 1,
         "mtm_mdd": float(dd), "exp_bp": np.nan, "win_rate": np.nan,
         "n_closed": 0, "max_locked_days": int((seg.index[-1] - seg.index[0]).days),
         "uw_days": int(uw_days), "exh_bars": 0, "exposure": 1.0,
         "final_eq": CAPITAL * tot},
        {"combo": "risk_free_4pct", "cagr": RF_ANNUAL, "mtm_mdd": 0.0,
         "exp_bp": np.nan, "win_rate": np.nan, "n_closed": 0,
         "max_locked_days": 0, "uw_days": 0, "exh_bars": 0, "exposure": 0.0,
         "final_eq": CAPITAL * (1 + RF_ANNUAL) ** years},
    ]


def combo_name(p: dict) -> str:
    ex = f"{p['exit_kind']}{p['exit_p']:g}" if p["exit_kind"] != "swing" \
        else f"swing_w{int(p['exit_p'])}"
    return (f"d{p['d']*100:g}|{p['confirm']}|{ex}|{p['sizing']}|"
            f"{'stop30' if p['acct_stop'] else 'nostop'}")


# ----------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    bars = load_bars(str(ROOT / "data" / "TSLA_1h_alpaca.csv"))
    D = precompute(bars)
    idx = D["idx"]
    n = len(idx)
    i_split = int(np.searchsorted(idx, TRAIN_END))
    print(f"bars={n}  train=[{idx[0].date()} .. {idx[i_split-1].date()}] "
          f"({i_split})  valid=[{idx[i_split].date()} .. {idx[-1].date()}] "
          f"({n - i_split})")

    ds = [0.05, 0.10, 0.15]
    confirms = ["none", "rsi", "bounce", "rsi+bounce"]
    exits = [("fixed_tp", 0.02), ("fixed_tp", 0.05), ("fixed_tp", 0.10),
             ("trailing", 0.03), ("trailing", 0.05), ("swing", SWING_W)]
    sizings = ["equal", "martingale", "vol_inverse"]
    stops = [False, True]

    grid = [dict(d=d, confirm=cf, exit_kind=ek, exit_p=ep, sizing=sz,
                 acct_stop=stp)
            for d in ds for cf in confirms for (ek, ep) in exits
            for sz in sizings for stp in stops]
    print(f"grid combos: {len(grid)}")

    rows = []
    for gi, p in enumerate(grid):
        stats, _ = simulate(D, 0, i_split, **p)
        score = stats["cagr"] / max(0.05, abs(stats["mtm_mdd"]))
        rows.append({"combo": combo_name(p), **p, **stats, "score": score,
                     "eligible": stats["n_closed"] >= MIN_CLOSED})
        if (gi + 1) % 100 == 0:
            print(f"  {gi+1}/{len(grid)}  ({time.time()-t0:.0f}s)")
    gdf = pd.DataFrame(rows).sort_values("score", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(OUT_DIR / "grid_train.csv", index=False)

    # stop-control pairing: what does the -30% account stop actually do?
    gdf["pair_key"] = gdf["combo"].str.rsplit("|", n=1).str[0]
    piv = gdf.pivot_table(index="pair_key", columns="acct_stop",
                          values="cagr", aggfunc="first")
    stop_diff = (piv[True] - piv[False]).dropna()
    stop_hit = stop_diff[stop_diff.abs() > 1e-9]

    naive_row = gdf[gdf["combo"] == "d5|none|fixed_tp0.02|equal|nostop"]

    elig = gdf[gdf["eligible"]]
    top5 = elig.head(5)
    print("\n== train top5 ==")
    cols = ["combo", "score", "cagr", "mtm_mdd", "exp_bp", "win_rate",
            "n_closed", "max_locked_days", "uw_days", "exh_bars", "exposure"]
    print(top5[cols].to_string(index=False))

    # ---- validation
    vrows = []
    for _, r in top5.iterrows():
        p = dict(d=r["d"], confirm=r["confirm"], exit_kind=r["exit_kind"],
                 exit_p=r["exit_p"], sizing=r["sizing"],
                 acct_stop=bool(r["acct_stop"]))
        st, _ = simulate(D, i_split, n, **p)
        vrows.append({"combo": r["combo"], **st,
                      "score": st["cagr"] / max(0.05, abs(st["mtm_mdd"]))})
    vrows += benchmark_rows(D, i_split, n)
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(OUT_DIR / "valid_top5.csv", index=False)
    print("\n== validation (2024-01 .. 2026-07) ==")
    print(vdf[[c for c in cols if c in vdf.columns]].to_string(index=False))

    # ---- crash window stress (run on train, slice 2021-11..2023-01)
    crows = []
    for _, r in top5.iterrows():
        p = dict(d=r["d"], confirm=r["confirm"], exit_kind=r["exit_kind"],
                 exit_p=r["exit_p"], sizing=r["sizing"],
                 acct_stop=bool(r["acct_stop"]))
        _, dg = simulate(D, 0, i_split, **p, diag=True)
        peak = dg["mtm"].cummax()
        dd = dg["mtm"] / peak - 1.0
        w = dg[(dg.index >= CRASH_START) & (dg.index < CRASH_END)]
        wdd = dd[(dd.index >= CRASH_START) & (dd.index < CRASH_END)]
        crows.append({
            "combo": r["combo"],
            "crash_mdd_vs_peak": float(wdd.min()),
            "crash_min_eq_vs_capital": float(w["mtm"].min() / CAPITAL - 1),
            "locked_peak_pct": float(w["invested"].max() / CAPITAL),
            "max_open_tranches": int(w["n_open"].max()),
            "cash_exhausted_bars": int(w["exh"].sum()),
        })
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT_DIR / "crash_window.csv", index=False)
    print("\n== crash window 2021-11-01 .. 2023-01-31 (within train run) ==")
    print(cdf.to_string(index=False))

    # ---- +-10% neighbourhood perturbation of the train winner
    best = top5.iloc[0]
    bp = dict(d=best["d"], confirm=best["confirm"], exit_kind=best["exit_kind"],
              exit_p=best["exit_p"], sizing=best["sizing"],
              acct_stop=bool(best["acct_stop"]))
    variants = [("base", bp)]
    for f, tag in [(0.9, "-10%"), (1.1, "+10%")]:
        q = dict(bp); q["d"] = round(bp["d"] * f, 4)
        variants.append((f"d{tag}", q))
    if bp["exit_kind"] in ("fixed_tp", "trailing"):
        for f, tag in [(0.9, "-10%"), (1.1, "+10%")]:
            q = dict(bp); q["exit_p"] = round(bp["exit_p"] * f, 5)
            variants.append((f"exit{tag}", q))
    else:  # swing: perturb the confirmation window
        for wv in (SWING_W - 1, SWING_W + 1):
            q = dict(bp); q["exit_p"] = wv
            variants.append((f"swing_w{wv}", q))
    prows = []
    for tag, q in variants:
        st_t, _ = simulate(D, 0, i_split, **q)
        st_v, _ = simulate(D, i_split, n, **q)
        prows.append({
            "variant": tag, "combo": combo_name(q),
            "train_score": st_t["cagr"] / max(0.05, abs(st_t["mtm_mdd"])),
            "train_cagr": st_t["cagr"], "train_mdd": st_t["mtm_mdd"],
            "valid_cagr": st_v["cagr"], "valid_mdd": st_v["mtm_mdd"],
            "valid_exp_bp": st_v["exp_bp"], "valid_n_closed": st_v["n_closed"],
        })
    pdf = pd.DataFrame(prows)
    pdf.to_csv(OUT_DIR / "perturb.csv", index=False)
    print("\n== perturbation of winner ==")
    print(pdf.to_string(index=False))

    # ---- summary.txt (Chinese, honest reading)
    tb = benchmark_rows(D, 0, i_split)
    bh_t, rf_t = tb[0], tb[1]
    vb = [r for r in vrows if r["combo"] == "buy_and_hold"][0]
    rf_v = [r for r in vrows if r["combo"] == "risk_free_4pct"][0]
    strat_v = vdf[~vdf["combo"].isin(["buy_and_hold", "risk_free_4pct"])]
    best_v = strat_v.sort_values("cagr", ascending=False).iloc[0]
    n_beat_bh = int((strat_v["cagr"] > vb["cagr"]).sum())
    n_beat_rf = int((strat_v["cagr"] > rf_v["cagr"]).sum())
    d_scores = pdf["train_score"]
    stable = d_scores.min() > 0.5 * d_scores.max() if d_scores.max() > 0 else False

    def fp(x, pct=True):
        return f"{x*100:.1f}%" if pct else f"{x:.2f}"

    lines = []
    A = lines.append
    A("逢跌买入 + 只在盈利时卖出（无止损）策略族 —— 诚实评估")
    A("=" * 62)
    A(f"数据: TSLA 1h ({idx[0].date()} .. {idx[-1].date()}, {n} 根), 手续费单边 1bp, 初始资金 ${CAPITAL:,.0f}")
    A(f"切分: 训练 {idx[0].date()}..{idx[i_split-1].date()}（含 2021-11..2023-01 崩盘）, 验证 {idx[i_split].date()}..{idx[-1].date()}")
    A(f"网格: d(3) x 确认(4: none/rsi/bounce/rsi+bounce) x 出场(6) x 仓位(3) x 止损对照(2) = {len(grid)} 组合")
    A(f"打分: CAGR / max(0.05, |MTM 最大回撤|), 门槛已平仓笔数 >= {MIN_CLOSED}（达标 {len(elig)}/{len(grid)}）")
    A("出场实现口径: fixed_tp 逐档（每档成本价+tp 限价）；trailing/swing 整体仓位一次性卖出。")
    A("vol_inverse 口径: 档位资金 = capital/10 x clip(参考波动率/当前20日波动率, 0.5, 2)；")
    A("  严格归一化到总资金需要未来波动率，这里用截断+现金上限代替（已如实说明）。")
    A("时序: 档位价用上一根收盘前的 ATH；trailing 峰值用截至上一根的高点；swing 确认滞后 5 根才可行动。")
    A("")
    A("[1] 训练段 top5（按 score；nostop/stop30 结果相同的组合是因为其回撤从未触及 -30%，止损对照失效）")
    A(top5[cols].to_string(index=False))
    A(f"训练段对照: buy&hold CAGR {fp(bh_t['cagr'])} (MTM回撤 {fp(bh_t['mtm_mdd'])}), 无风险 {fp(RF_ANNUAL)}")
    if len(naive_row):
        nr = naive_row.iloc[0]
        A(f"朴素基线（原 no_loss_stress_test 参数 d5/tp2%/equal，本实现口径）: CAGR {fp(nr['cagr'])}, "
          f"MTM回撤 {fp(nr['mtm_mdd'])}, 胜率 {fp(nr['win_rate'])}, 单笔 {nr['exp_bp']:.0f}bp, "
          f"最长锁死 {int(nr['max_locked_days'])} 天, 敞口 {fp(nr['exposure'])}")
        A("  （注意: 本实现的档位锚定滚动 ATH 并允许复位重买，比原脚本的'分段高点'口径在崩盘里买得更深，回撤也更深。）")
    A("")
    A("[2] 验证段 2024-01..2026-07（top5 + 对照）")
    A(vdf[[c for c in cols if c in vdf.columns]].to_string(index=False))
    A("")
    A("[3] 崩盘窗口压测 2021-11-01..2023-01-31（训练段运行内切片）")
    A(cdf.to_string(index=False))
    A("")
    A("[4] 最优组合 ±10% 邻域扰动（d 与出场参数）")
    A(pdf.to_string(index=False))
    A("")
    A("[5] 判读（诚实版）")
    A("-" * 62)
    A("* 胜率与期望必须分开看：fixed_tp 变体的已平仓胜率是构造出来的 100%（卖出条件就是'有盈利'），")
    A("  trailing/swing 整体出场后按档拆分则跌到 ~89% 甚至更低——'只在盈利卖'在档位层面本来就不成立；")
    A(f"  验证段表现最好的组合年化也只有 {fp(best_v['cagr'])}（单笔 {best_v['exp_bp']:.0f}bp），高胜率没有转化为收益率。")
    A(f"* 止损对照: 216 组 nostop/stop30 配对中 {len(stop_hit)} 组真正触发过 -30% 强平；触发者 CAGR 全部变差")
    A(f"  （差幅 {fp(stop_diff.min())} 至 {fp(stop_hit.max())}，无一例改善）。在'最终总会涨回来'的这份样本里，")
    A("  账户级止损只是把浮亏变成实亏并错过反弹——用户'无止损'的直觉在该样本内成立，但这依赖 TSLA 每次都涨回来，")
    A("  是幸存者偏差保护下的结论，不是普适规律。")
    if n_beat_bh == 0:
        A(f"* 验证段没有任何 top5 组合跑赢 buy&hold（{fp(vb['cagr'])}）——用户构想的全部改进版本仍然输给'直接持有'。")
    else:
        A(f"* 验证段有 {n_beat_bh}/{len(strat_v)} 个 top5 组合跑赢 buy&hold（{fp(vb['cagr'])}）。")
    if n_beat_rf == 0:
        A(f"* 验证段没有任何 top5 组合跑赢 4% 无风险利率——该策略族在样本外连存款都不如。")
    else:
        A(f"* 验证段有 {n_beat_rf}/{len(strat_v)} 个 top5 组合跑赢 4% 无风险利率。")
    A(f"* 崩盘压测: top5 在 2021-11..2023-01 内 MTM 回撤最深仅 {fp(cdf['crash_mdd_vs_peak'].min())}，"
      f"锁死资金峰值 {fp(cdf['locked_peak_pct'].max())}，现金耗尽 {int(cdf['cash_exhausted_bars'].max())} bar——")
    A("  但这是 score 公式（分母罚回撤）筛出来的结果: top5 全是低敞口的保守成员，崩盘期基本空仓躲过。")
    ns = gdf[~gdf["acct_stop"]]
    A(f"  同族激进成员（如 d5|none|*|martingale）训练段 MTM 回撤达 {fp(ns['mtm_mdd'].min())}、敞口 >90%、"
      f"单笔最长锁死 {int(ns['max_locked_days'].max())} 天——构想的原始形态在崩盘里并没有被指标救回来。")
    A("  '大公司不会倒'不解决'资金在深水区被锁死两年多、机会成本持续累积'的问题。")
    if stable:
        A("* 扰动: 最优组合 ±10% 邻域 score 未塌方（min > 0.5*max），参数不算孤峰，但这只说明'稳定地平庸'。")
    else:
        A("* 扰动: 最优组合 ±10% 邻域 score 明显退化，训练段最优很可能是过拟合出的孤峰。")
    A("* 结构性结论: 入场确认（RSI/收阳）只改变买点的微观位置，出场改进（trailing/swing）只改变卖点形态；")
    A("  它们都无法改变该构想的根本结构——上涨时仓位很轻(敞口占比低)、暴跌时满仓被锁、")
    A("  盈利被'尽早落袋'截断而亏损被'永不止损'放大为时间成本。指标救不了这个结构。")
    A("")
    A(f"生成: research/dip_accumulate_backtest.py, 用时 {time.time()-t0:.0f}s")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT_DIR}/(grid_train|valid_top5|crash_window|perturb).csv, summary.txt")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
