"""E13 — salary-stream ladder DCA with per-tranche take-profit (user design).

Design under test (as registered in docs/strategy-lab.md E13, with the
2026-07-24 semantics corrections from the user):

- Monthly salary injection: $4,167 (~=CNY 30k @ 7.2) on the first trading day
  of each month, accumulated in a cash pool. Sale proceeds return to the pool.
- First position at the window's first open. Ladder: next buy price =
  last fill * (1 - step); an intraday Low touching it fills at the touched
  price + 2bp slippage (gap-downs fill at the open). Tranche stake =
  min(cash available [+ bridge loan], fixed unit). If the level cannot be
  funded, the order stays standing (ladder does NOT advance).
- Per-tranche take profit: each tranche independently sells when the day's
  High STRICTLY crosses its all-in cost * (1 + tp). 1bp fee per side.
  No stop loss, ever.
- Anchor reset: when flat, a close above the prior 60-day high re-anchors
  the ladder; a fully-closed ladder re-anchors at that day's close.

Financing semantics (user clarification, 2026-07-24 — bridge loan, NOT
steady-state leverage):
- A loan is taken ONLY when a ladder buy triggers and the cash pool cannot
  fund it; loan tops up that buy. Cap = {0%, 50%, 100%} of current own
  equity. Deposits and sale proceeds repay the loan FIRST (single signed
  cash balance: negative = outstanding loan). Interest 6.5%/yr daily on the
  outstanding balance.
- Final clarification (supersedes the intermediate "mark-only" version): the
  loan cap is an optimization variable — grid {0, 1 month salary, 3 months
  salary, 50% of equity, 100% of equity}. The mechanical simulation runs in
  full. No forced-liquidation cascade is simulated, but the 30% maintenance
  line is monitored daily: min(equity/market value) over the run and every
  crossing below 30% is a "danger event". Combos with any danger event are
  excluded from the recommendation list (kept in a separate high-risk group).
  Every borrow is still logged as a manual intervention point, now with the
  post-borrow buffer to the liquidation line attached.

Headline metric: XIRR on the actual cash-flow sequence (deposits out,
final equity in). Controls per window: DCA-hold (same deposits, buy at that
day's open, never sell) and cash at 4%/yr.

Windows: W1 2023-07 -> 2026-07 (user's "ignore the extremes" window),
W2 2018-07 -> 2026-07 (includes the -75% crash). Inside W1: train
2023-07 -> 2025-06, validation 2025-07 -> 2026-07 (grid picked on train,
re-checked on validation, against hindsight-fitted "gut prices").

Outputs -> outputs/e13_salary_ladder/
    grid_results.csv, best_equity_curves.csv, intervention_points.csv,
    summary.txt (written by the runner, verdict composed separately)

Usage: .venv/bin/python research/e13_salary_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402

DEPOSIT = 4167.0          # monthly, ~= CNY 30k / 7.2
FEE = 0.0001              # 1bp per side
SLIP = 0.0002             # +2bp pessimistic slippage on buys
LOAN_RATE = 0.065         # bridge-loan APR, daily accrual on outstanding balance
HIGH_WIN = 60             # flat-period re-anchor: close above prior 60-day high
CASH_RATE = 0.04          # risk-free control

STEPS = [0.05, 0.10, 0.15]
TPS = [0.08, 0.12, 0.20, 0.30]
UNITS = [4167.0, 12500.0, 25000.0]          # 1 / 3 / 6 months of salary
UNIT_LABEL = {4167.0: "1mo", 12500.0: "3mo", 25000.0: "6mo"}
# emergency credit line: (label, kind, value); "abs" = fixed $, "frac" = frac of equity
LOAN_CAPS = [
    ("L0", "abs", 0.0),
    ("L1mo", "abs", 4167.0),
    ("L3mo", "abs", 12500.0),
    ("L50", "frac", 0.5),
    ("L100", "frac", 1.0),
]
MAINT = 0.30                                 # maintenance line monitored, not enforced

OUT = ROOT / "outputs" / "e13_salary_ladder"


# ---------------------------------------------------------------- data


def daily_bars() -> pd.DataFrame:
    bars = load_bars(str(ROOT / "data" / "TSLA_1h_alpaca.csv"))
    et = bars.index.tz_convert(ET)
    d = bars.groupby(pd.Series(et.date, index=bars.index)).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    d.index = pd.to_datetime(d.index)
    d.index.name = "Date"
    return d


def deposit_days(index: pd.DatetimeIndex) -> set:
    """First trading day of each calendar month inside the window."""
    s = pd.Series(index, index=index)
    ym = s.dt.to_period("M")
    return set(s.groupby(ym).first())


def xirr(dates: list, amounts: list) -> float:
    """Annualized money-weighted return from a dated cash-flow sequence."""
    if len(dates) < 2 or amounts[-1] <= 0:
        return float("nan")
    t0 = dates[0]
    yrs = np.array([(d - t0).days / 365.25 for d in dates])
    cf = np.array(amounts, dtype=float)

    def npv(r):
        return float(np.sum(cf / (1.0 + r) ** yrs))

    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def nav_drawdown(dates, equity, flows) -> float:
    """Max drawdown of the flow-adjusted unit-value (TWR) curve — removes the
    cushioning that fresh deposits give a raw equity curve."""
    eq = np.asarray(equity, dtype=float)
    fl = np.asarray(flows, dtype=float)
    nav = np.empty(len(eq))
    nav[0] = 1.0
    for i in range(1, len(eq)):
        base = eq[i - 1] + fl[i]          # deposit arrives at start of day i
        nav[i] = nav[i - 1] * (eq[i] / base) if base > 0 else nav[i - 1]
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())


# ---------------------------------------------------------------- engine


def simulate_ladder(daily: pd.DataFrame, step: float, tp: float, unit: float,
                    cap_kind: str, cap_value: float) -> dict:
    idx = daily.index
    dep_days = deposit_days(idx)
    roll_high = daily["Close"].shift(1).rolling(HIGH_WIN).max()

    cash = 0.0                       # signed: negative = outstanding bridge loan
    tranches: list[dict] = []
    closed: list[dict] = []
    interventions: list[dict] = []
    eq_curve, flow_curve, mv_curve = [], [], []
    cf_dates, cf_amts = [], []

    anchor = float(daily["Open"].iloc[0])
    next_buy = None                  # None => first-day open buy pending
    interest_paid = 0.0
    total_borrowed = 0.0
    max_debt = 0.0
    days_with_debt = 0
    debt_episodes = 0
    in_debt_prev = False
    skipped_buys = 0
    prev_date = idx[0]
    peak_debt_equity = 0.0
    min_margin_ratio = np.inf            # min daily equity/MV (only binds when in debt)
    danger_events = 0                    # crossings below the 30% maintenance line
    days_below_maint = 0
    below_prev = False

    def buy(ts, fill_raw, close_px):
        """Try to fund one tranche at fill_raw. Returns True if bought."""
        nonlocal cash, total_borrowed, max_debt, skipped_buys
        mv = sum(t["shares"] for t in tranches) * fill_raw
        equity = cash + mv
        avail_cash = max(0.0, cash)
        debt = max(0.0, -cash)
        cap = cap_value if cap_kind == "abs" else cap_value * equity
        borrow_room = max(0.0, cap - debt)
        stake = min(unit, avail_cash + borrow_room)
        if stake < 1.0:
            skipped_buys += 1
            return False
        borrow = max(0.0, stake - avail_cash)
        fill_px = fill_raw * (1 + SLIP)
        shares = stake * (1 - FEE) / fill_px
        cost_ps = stake / shares
        cash -= stake
        tranches.append({"cost": cost_ps, "shares": shares, "t_open": ts})
        if borrow > 0:
            total_borrowed += borrow
            max_debt = max(max_debt, -cash)
            mv_now = sum(t["shares"] for t in tranches) * fill_raw
            eq_now = cash + mv_now
            interventions.append({
                "date": ts, "borrow": round(borrow, 2), "stake": round(stake, 2),
                "fill_price": round(fill_raw, 2),
                "pct_below_anchor": round(fill_raw / anchor - 1, 4),
                "cash_before": round(avail_cash, 2),
                "debt_after": round(-cash, 2),
                "equity": round(eq_now, 2),
                "margin_ratio_after": round(eq_now / mv_now, 4) if mv_now > 0 else np.nan,
                "n_open_tranches": len(tranches),
            })
        return True

    for i, (ts, row) in enumerate(daily.iterrows()):
        o, h, lo, c = (float(row["Open"]), float(row["High"]),
                       float(row["Low"]), float(row["Close"]))

        # 1) salary deposit (repays loan first: signed cash balance)
        flow = 0.0
        if ts in dep_days:
            cash += DEPOSIT
            flow = DEPOSIT
            cf_dates.append(ts)
            cf_amts.append(-DEPOSIT)

        # 2) first-day opening position
        if next_buy is None:
            anchor = o
            buy(ts, o, c)
            last_fill = o
            next_buy = last_fill * (1 - step)

        # 3) per-tranche take profits (High strictly crosses cost*(1+tp))
        still = []
        for tr in tranches:
            target = tr["cost"] * (1 + tp)
            if h > target:
                cash += tr["shares"] * target * (1 - FEE)
                closed.append({
                    "t_open": tr["t_open"], "t_close": ts,
                    "days": (ts - tr["t_open"]).days,
                    "ret": target * (1 - FEE) / tr["cost"] - 1,   # ~= tp - 2bp
                })
            else:
                still.append(tr)
        ladder_closed = bool(tranches) and not still
        tranches = still
        if ladder_closed:
            anchor = c
            next_buy = anchor * (1 - step)

        # 4) flat-period re-anchor on 60-day high
        if not tranches and not np.isnan(roll_high.iloc[i]) and c > roll_high.iloc[i]:
            anchor = c
            next_buy = anchor * (1 - step)

        # 5) ladder buys — first fill of the day may be a gap-down open fill
        first_today = True
        guard = 0
        while lo <= next_buy * (1 + 1e-12) and guard < 60:
            guard += 1
            fill_raw = min(next_buy, o) if first_today else next_buy
            first_today = False
            if not buy(ts, fill_raw, c):
                break                          # unfunded: order stays standing
            next_buy = fill_raw * (1 - step)

        # 6) daily interest on outstanding loan
        delta_days = max(1, (ts - prev_date).days) if i else 0
        debt = max(0.0, -cash)
        if debt > 0 and delta_days:
            intr = debt * LOAN_RATE / 365.0 * delta_days
            cash -= intr
            interest_paid += intr
        debt = max(0.0, -cash)
        if debt > 0:
            days_with_debt += 1
            if not in_debt_prev:
                debt_episodes += 1
            in_debt_prev = True
        else:
            in_debt_prev = False
        max_debt = max(max_debt, debt)
        prev_date = ts

        mv = sum(t["shares"] for t in tranches) * c
        equity_close = cash + mv
        eq_curve.append(equity_close)
        flow_curve.append(flow)
        mv_curve.append(mv)

        # 7) daily risk monitors (maintenance line watched, never enforced)
        if equity_close > 0:
            peak_debt_equity = max(peak_debt_equity, debt / equity_close)
        if mv > 0:
            ratio = equity_close / mv
            min_margin_ratio = min(min_margin_ratio, ratio)
            below = ratio < MAINT
            if below:
                days_below_maint += 1
                if not below_prev:
                    danger_events += 1
            below_prev = below
        else:
            below_prev = False

    last_c = float(daily["Close"].iloc[-1])
    final_eq = eq_curve[-1]
    cf_dates.append(idx[-1])
    cf_amts.append(final_eq)

    closed_df = pd.DataFrame(closed)
    open_ages = [(idx[-1] - t["t_open"]).days for t in tranches]
    open_cost = sum(t["cost"] * t["shares"] for t in tranches)
    open_mv = sum(t["shares"] for t in tranches) * last_c

    return {
        "irr": xirr(cf_dates, cf_amts),
        "final_equity": final_eq,
        "total_deposit": -sum(a for a in cf_amts[:-1]),
        "dd_twr": nav_drawdown(idx, eq_curve, flow_curve),
        "n_closed": len(closed_df),
        "hit_rate": float((closed_df["ret"] > 0).mean()) if len(closed_df) else np.nan,
        "max_lock_days": int(max(
            [closed_df["days"].max() if len(closed_df) else 0] + open_ages)),
        "n_open_end": len(tranches),
        "open_unrealized": open_mv - open_cost,
        "n_interventions": len(interventions),
        "total_borrowed": total_borrowed,
        "max_debt": max_debt,
        "days_with_debt": days_with_debt,
        "n_debt_episodes": debt_episodes,
        "avg_debt_episode_days": days_with_debt / debt_episodes if debt_episodes else 0.0,
        "interest_paid": interest_paid,
        "final_debt": max(0.0, -cash),
        "peak_debt_equity": peak_debt_equity,
        "min_margin_ratio": float(min_margin_ratio) if np.isfinite(min_margin_ratio) else np.nan,
        "n_danger_events": danger_events,
        "days_below_maint": days_below_maint,
        "skipped_buys": skipped_buys,
        "avg_invested_frac": float(np.mean(
            [m / e for m, e in zip(mv_curve, eq_curve) if e > 0])),
        "_equity": pd.DataFrame({"Date": idx, "equity": eq_curve, "flow": flow_curve,
                                 "mv": mv_curve}),
        "_interventions": interventions,
        "_closed": closed_df,
    }


def simulate_dca(daily: pd.DataFrame) -> dict:
    """Control: identical deposits, buy in full at that day's open, never sell."""
    idx = daily.index
    dep_days = deposit_days(idx)
    cash, shares = 0.0, 0.0
    eq_curve, flow_curve, mv_curve, cf_dates, cf_amts = [], [], [], [], []
    for ts, row in daily.iterrows():
        o, c = float(row["Open"]), float(row["Close"])
        flow = 0.0
        if ts in dep_days:
            cash += DEPOSIT
            flow = DEPOSIT
            cf_dates.append(ts)
            cf_amts.append(-DEPOSIT)
            px = o * (1 + SLIP)
            shares += cash * (1 - FEE) / px
            cash = 0.0
        eq_curve.append(cash + shares * c)
        flow_curve.append(flow)
        mv_curve.append(shares * c)
    final_eq = eq_curve[-1]
    cf_dates.append(idx[-1])
    cf_amts.append(final_eq)
    return {
        "irr": xirr(cf_dates, cf_amts),
        "final_equity": final_eq,
        "total_deposit": -sum(a for a in cf_amts[:-1]),
        "dd_twr": nav_drawdown(idx, eq_curve, flow_curve),
        "avg_invested_frac": float(np.mean(
            [m / e for m, e in zip(mv_curve, eq_curve) if e > 0])),
        "_equity": pd.DataFrame({"Date": idx, "equity": eq_curve, "flow": flow_curve,
                                 "mv": mv_curve}),
    }


def simulate_cash(daily: pd.DataFrame) -> dict:
    idx = daily.index
    dep_days = deposit_days(idx)
    bal = 0.0
    prev = idx[0]
    eq_curve, flow_curve = [], []
    n_dep = 0
    for ts in idx:
        bal *= (1 + CASH_RATE) ** ((ts - prev).days / 365.25)
        prev = ts
        flow = 0.0
        if ts in dep_days:
            bal += DEPOSIT
            flow = DEPOSIT
            n_dep += 1
        eq_curve.append(bal)
        flow_curve.append(flow)
    return {
        "irr": CASH_RATE,
        "final_equity": bal,
        "total_deposit": n_dep * DEPOSIT,
        "dd_twr": 0.0,
        "_equity": pd.DataFrame({"Date": idx, "equity": eq_curve, "flow": flow_curve}),
    }


# ---------------------------------------------------------------- runner


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = daily_bars()
    windows = {
        "W1": d.loc["2023-07-01":],
        "W2": d,
        "W1train": d.loc["2023-07-01":"2025-06-30"],
        "W1val": d.loc["2025-07-01":],
    }
    for name, w in windows.items():
        print(f"{name}: {w.index[0].date()} .. {w.index[-1].date()}  ({len(w)} days)")

    grid_rows, interv_rows, curve_rows = [], [], []

    for wname, w in windows.items():
        # controls
        dca = simulate_dca(w)
        cash = simulate_cash(w)
        for label, res in [("DCA_hold", dca), ("cash_4pct", cash)]:
            grid_rows.append({
                "window": wname, "strategy": label, "step": np.nan, "tp": np.nan,
                "unit": np.nan, "loan_cap": np.nan,
                **{k: v for k, v in res.items() if not k.startswith("_")},
            })
            for _, r in res["_equity"].iterrows():
                curve_rows.append({"window": wname, "strategy": label,
                                   "date": r["Date"], "equity": r["equity"]})

        for stp in STEPS:
            for tp in TPS:
                for unit in UNITS:
                    for cap_label, cap_kind, cap_value in LOAN_CAPS:
                        res = simulate_ladder(w, stp, tp, unit, cap_kind, cap_value)
                        key = f"s{int(stp*100)}_tp{int(tp*100)}_{UNIT_LABEL[unit]}_{cap_label}"
                        grid_rows.append({
                            "window": wname, "strategy": key, "step": stp, "tp": tp,
                            "unit": unit, "loan_cap": cap_label,
                            "cap_kind": cap_kind, "cap_value": cap_value,
                            **{k: v for k, v in res.items() if not k.startswith("_")},
                        })
                        for ev in res["_interventions"]:
                            interv_rows.append({"window": wname, "strategy": key,
                                                "step": stp, "tp": tp, "unit": unit,
                                                "loan_cap": cap_label, **ev})
                        res_curves[(wname, key)] = res["_equity"]

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "grid_results.csv", index=False)
    pd.DataFrame(interv_rows).to_csv(OUT / "intervention_points.csv", index=False)

    # ---- keep equity curves: best-on-train + best-on-val (per loan tier, safe
    # combos preferred) + user example + top W2
    lad = grid[grid["step"].notna()]
    keep = []
    for sel_win in ["W1train", "W1val"]:
        sub_w = lad[lad["window"] == sel_win]
        for cap_label, _, _ in LOAN_CAPS:
            sub = sub_w[sub_w["loan_cap"] == cap_label].sort_values("irr", ascending=False)
            if len(sub):
                keep.append(sub.iloc[0]["strategy"])
    keep.append("s10_tp12_3mo_L0")          # user's worked example (250 -> 280)
    w2 = lad[(lad["window"] == "W2") & (lad["loan_cap"] == "L0")]
    keep.append(w2.sort_values("irr", ascending=False).iloc[0]["strategy"])
    keep = list(dict.fromkeys(keep))
    for (wname, key), frame in res_curves.items():
        if key in keep:
            for _, r in frame.iterrows():
                curve_rows.append({"window": wname, "strategy": key,
                                   "date": r["Date"], "equity": r["equity"]})
    pd.DataFrame(curve_rows).to_csv(OUT / "best_equity_curves.csv", index=False)
    print(f"kept curves for: {keep}")
    print(f"grid rows: {len(grid)}, intervention events: {len(interv_rows)}")


res_curves: dict = {}

if __name__ == "__main__":
    main()
