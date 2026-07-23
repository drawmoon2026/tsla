"""
Swing-based V event detection using ZigZag swings on 5m TSLA data.

Definitions:
- ZigZag swings with absolute percentage thresholds (e.g., 0.5%, 1%).
- Swing V event: consecutive swings High -> Low -> High where
  drop_pct >= X and rebound_pct >= Y.

Outputs:
- Counts per day/week/month.
- Distributions of drop_pct and rebound_pct.
- Average bars to rebound.
- Volatility regime clustering.
- Idealized equity curve (upper-bound alpha) assuming buy at trough, sell at rebound swing high.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data_io import load_bars  # noqa: E402


DATA_CSV_DEFAULT = "data/TSLA_5m_60d.csv"
EVENTS_DIR_DEFAULT = "outputs/zigzag"
SWING_THRESHOLDS = [0.005, 0.01]  # 0.5%, 1%
X_GRID = [0.01, 0.02, 0.03, 0.05]
Y_GRID = [0.01, 0.02, 0.03]

# One physical V event == one (peak_t, trough_t) pair. The X/Y (and swing_th)
# grids are nested, so the same swing sequence is detected by many combos;
# distribution stats must be computed on the deduplicated set.
UNIQUE_KEY = ["peak_t", "trough_t"]


@dataclass
class SwingPoint:
    idx: int
    ts: pd.Timestamp
    price: float
    kind: str  # "H" or "L"


@dataclass
class VEvent:
    swing_threshold: float
    x: float
    y: float
    peak: SwingPoint
    trough: SwingPoint
    rebound: SwingPoint
    drop_pct: float
    rebound_pct: float
    bars_to_rebound: int


def load_data(path: str) -> pd.DataFrame:
    return load_bars(path)


def zigzag_swings(close: pd.Series, threshold: float) -> List[SwingPoint]:
    """Return list of swing highs/lows using percent threshold."""
    prices = close.to_numpy()
    idxs = close.index
    n = len(prices)
    if n == 0:
        return []

    pivots: List[SwingPoint] = []
    last_pivot_idx = 0
    last_pivot_price = prices[0]
    last_pivot_kind = "H"  # provisional, will flip once trend known
    trend = 0  # 0 unknown, 1 up, -1 down
    extreme_idx = 0
    extreme_price = prices[0]

    for i in range(1, n):
        price = prices[i]

        if trend == 0:
            move = (price - last_pivot_price) / last_pivot_price
            if abs(move) >= threshold:
                trend = 1 if move > 0 else -1
                extreme_idx = i
                extreme_price = price
                last_pivot_kind = "L" if trend == 1 else "H"
        elif trend == 1:
            # uptrend, tracking highs
            if price > extreme_price:
                extreme_price = price
                extreme_idx = i
            elif (price - extreme_price) / extreme_price <= -threshold:
                # reversal to downtrend
                pivots.append(SwingPoint(extreme_idx, idxs[extreme_idx], extreme_price, "H"))
                trend = -1
                extreme_idx = i
                extreme_price = price
        else:
            # downtrend, tracking lows
            if price < extreme_price:
                extreme_price = price
                extreme_idx = i
            elif (price - extreme_price) / extreme_price >= threshold:
                pivots.append(SwingPoint(extreme_idx, idxs[extreme_idx], extreme_price, "L"))
                trend = 1
                extreme_idx = i
                extreme_price = price

    # add last extreme as final pivot
    kind = "H" if trend == 1 else "L"
    pivots.append(SwingPoint(extreme_idx, idxs[extreme_idx], extreme_price, kind))
    # ensure first pivot kind alternates; if not, prepend start
    if pivots and pivots[0].idx != 0:
        pivots.insert(0, SwingPoint(0, idxs[0], prices[0], "H" if trend == -1 else "L"))

    return pivots


def detect_v_events(swings: List[SwingPoint], x: float, y: float, swing_threshold: float) -> List[VEvent]:
    events: List[VEvent] = []
    for i in range(len(swings) - 2):
        a, b, c = swings[i], swings[i + 1], swings[i + 2]
        if a.kind != "H" or b.kind != "L" or c.kind != "H":
            continue
        drop_pct = (a.price - b.price) / a.price
        rebound_pct = (c.price - b.price) / b.price
        if drop_pct >= x and rebound_pct >= y:
            events.append(
                VEvent(
                    swing_threshold=swing_threshold,
                    x=x,
                    y=y,
                    peak=a,
                    trough=b,
                    rebound=c,
                    drop_pct=drop_pct,
                    rebound_pct=rebound_pct,
                    bars_to_rebound=c.idx - b.idx,
                )
            )
    return events


def volatility_regime(close: pd.Series, window: int = 78) -> pd.Series:
    rets = close.pct_change()
    vol = rets.rolling(window).std() * np.sqrt(252 * 78)  # annualize: 78 five-minute bars/day
    quantiles = vol.quantile([1 / 3, 2 / 3])
    def label(v):
        if np.isnan(v):
            return np.nan
        if v <= quantiles.iloc[0]:
            return "low"
        elif v <= quantiles.iloc[1]:
            return "mid"
        else:
            return "high"
    return vol.map(label)


def run_analysis(df: pd.DataFrame) -> dict:
    close = df["Close"]
    results = []
    all_events: List[VEvent] = []

    for th in SWING_THRESHOLDS:
        swings = zigzag_swings(close, th)
        for x in X_GRID:
            for y in Y_GRID:
                events = detect_v_events(swings, x, y, th)
                all_events.extend(events)
                results.append({"swing_th": th, "X": x, "Y": y, "count": len(events)})

    return {"events": all_events, "grid": pd.DataFrame(results)}


def summarize(events: List[VEvent], df: pd.DataFrame) -> dict:
    # DataFrame of events
    rows = []
    for e in events:
        rows.append(
            {
                "swing_th": e.swing_threshold,
                "X": e.x,
                "Y": e.y,
                "peak_t": e.peak.ts,
                "trough_t": e.trough.ts,
                "rebound_t": e.rebound.ts,
                "drop_pct": e.drop_pct,
                "rebound_pct": e.rebound_pct,
                "bars_to_rebound": e.bars_to_rebound,
            }
        )
    ev_df = pd.DataFrame(rows)
    if ev_df.empty:
        return {"events_df": ev_df, "events_unique": ev_df}

    # Deduplicate to physical events: the swing_th/X/Y grids are nested, so the
    # same (peak_t, trough_t) appears under many parameter combos. All
    # daily/weekly/monthly, drop/rebound and regime stats below use this
    # deduplicated set (representative row = loosest combo that detected it).
    uniq = ev_df.sort_values(UNIQUE_KEY + ["swing_th", "X", "Y"]).copy()
    n_combos = uniq.groupby(UNIQUE_KEY).size().rename("n_param_combos")
    uniq = uniq.drop_duplicates(subset=UNIQUE_KEY, keep="first")
    uniq = uniq.merge(n_combos, left_on=UNIQUE_KEY, right_index=True)
    uniq = uniq.sort_values("peak_t").reset_index(drop=True)

    uniq["date"] = uniq["trough_t"].dt.tz_convert("America/New_York").dt.date
    uniq["week"] = uniq["trough_t"].dt.to_period("W").dt.start_time
    uniq["month"] = uniq["trough_t"].dt.to_period("M").dt.start_time

    daily = uniq.groupby(["date"]).size()
    weekly = uniq.groupby(["week"]).size()
    monthly = uniq.groupby(["month"]).size()

    drop_stats = uniq["drop_pct"].describe(percentiles=[0.1, 0.5, 0.9])
    reb_stats = uniq["rebound_pct"].describe(percentiles=[0.1, 0.5, 0.9])
    avg_bars = uniq["bars_to_rebound"].mean()

    # volatility regime (reindex is NaN-safe if a trough timestamp is missing)
    regime = volatility_regime(df["Close"])
    uniq["regime"] = regime.reindex(uniq["trough_t"]).to_numpy()
    regime_counts = uniq.groupby("regime").size()

    return {
        "events_df": ev_df,
        "events_unique": uniq,
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "drop_stats": drop_stats,
        "rebound_stats": reb_stats,
        "avg_bars": avg_bars,
        "regime_counts": regime_counts,
    }


def best_combo_equity(ev_df: pd.DataFrame) -> Tuple[pd.Series, dict]:
    # find combo with highest compounded return
    equities = {}
    for (th, x, y), g in ev_df.sort_values("trough_t").groupby(["swing_th", "X", "Y"]):
        eq = (1 + g["rebound_pct"]).cumprod()
        equities[(th, x, y)] = eq
    best = max(equities.items(), key=lambda kv: kv[1].iloc[-1])
    (th, x, y), curve = best
    return curve, {"swing_th": th, "X": x, "Y": y, "final_return": curve.iloc[-1] - 1, "trades": len(curve)}


def plots(ev_uniq: pd.DataFrame, ev_grid: pd.DataFrame, df: pd.DataFrame, outdir: Path) -> None:
    """Distribution plots use the deduplicated physical events (ev_uniq);
    the per-combo equity curve uses the full grid table (ev_grid)."""
    outdir.mkdir(parents=True, exist_ok=True)
    # Histogram of drop and rebound (deduplicated physical events)
    plt.figure(figsize=(8, 4))
    plt.hist(ev_uniq["drop_pct"], bins=40, alpha=0.6, label="drop_pct")
    plt.hist(ev_uniq["rebound_pct"], bins=40, alpha=0.6, label="rebound_pct")
    plt.legend()
    plt.title("V magnitude distribution (deduped physical events)")
    plt.xlabel("pct")
    plt.tight_layout()
    plt.savefig(outdir / "v_magnitude_hist.png", dpi=150)
    plt.close()

    # Time-of-day heatmap (ET, trough time)
    et_times = ev_uniq["trough_t"].dt.tz_convert("America/New_York")
    tod = et_times.dt.hour * 12 + et_times.dt.minute // 5  # 5m slots 0-287
    counts = tod.value_counts().sort_index()
    hours = np.arange(24)
    matrix = np.zeros((24, 12))
    for slot, cnt in counts.items():
        h = slot // 12
        mslot = slot % 12
        if h < 24 and mslot < 12:
            matrix[h, mslot] = cnt
    plt.figure(figsize=(10, 5))
    plt.imshow(matrix, aspect="auto", origin="lower")
    plt.yticks(range(24), range(24))
    plt.xticks(range(0, 12, 3), [f"{m*5:02d}" for m in range(0, 60, 15)])
    plt.colorbar(label="V count")
    plt.xlabel("Minute within hour")
    plt.ylabel("Hour of day (ET)")
    plt.title("Time-of-day heatmap of V occurrences (trough time, deduped)")
    plt.tight_layout()
    plt.savefig(outdir / "v_time_of_day_heatmap.png", dpi=150)
    plt.close()

    # Equity curve for best combo (per-combo grid, no cross-combo duplication)
    curve, meta = best_combo_equity(ev_grid)
    plt.figure(figsize=(8, 4))
    plt.plot(curve.index, curve.values)
    plt.title(f"Idealized equity curve (best combo th={meta['swing_th']}, X={meta['X']}, Y={meta['Y']})")
    plt.xlabel("Event #")
    plt.ylabel("Equity (start=1)")
    plt.tight_layout()
    plt.savefig(outdir / "equity_curve.png", dpi=150)
    plt.close()
    meta_path = outdir / "equity_meta.txt"
    meta_path.write_text(str(meta))


def write_meta(events_dir: Path, df: pd.DataFrame, input_csv: str, n_grid: int, n_unique: int) -> None:
    """Sidecar metadata so consumers (layered_backtest) can verify that the
    events file and the price cache cover the same period."""
    meta = {
        "source_csv": input_csv,
        "data_start": df.index.min().isoformat(),
        "data_end": df.index.max().isoformat(),
        "n_bars": int(len(df)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dedup_key": UNIQUE_KEY,
        "note": (
            "zigzag_events.csv holds deduplicated physical events (one row per "
            "(peak_t, trough_t); representative params = loosest combo, "
            "n_param_combos = combos that detected it). Full per-combo grid in "
            "zigzag_events_grid.csv — filter it on (swing_th, X, Y) for "
            "single-parameter-group consumption."
        ),
        "n_events_grid_rows": n_grid,
        "n_events_unique": n_unique,
    }
    (events_dir / "zigzag_events_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DATA_CSV_DEFAULT)
    parser.add_argument("--output_dir", default="figures/zigzag")
    parser.add_argument("--events_dir", default=EVENTS_DIR_DEFAULT)
    args = parser.parse_args()

    df = load_data(args.input_csv)
    res = run_analysis(df)
    summary = summarize(res["events"], df)
    ev_df = summary["events_df"]

    outdir = Path(args.output_dir)
    events_dir = Path(args.events_dir)
    if not ev_df.empty:
        ev_uniq = summary["events_unique"]
        events_dir.mkdir(parents=True, exist_ok=True)
        date_fmt = "%Y-%m-%dT%H:%M:%S%z"
        # zigzag_events.csv = deduplicated physical events (backtest input);
        # zigzag_events_grid.csv = full parameter grid (one row per combo hit).
        ev_uniq.to_csv(events_dir / "zigzag_events.csv", index=False, date_format=date_fmt)
        ev_df.to_csv(events_dir / "zigzag_events_grid.csv", index=False, date_format=date_fmt)
        summary["daily"].to_csv(events_dir / "zigzag_daily.csv", date_format="%Y-%m-%d")
        summary["weekly"].to_csv(events_dir / "zigzag_weekly.csv", date_format="%Y-%m-%d")
        summary["monthly"].to_csv(events_dir / "zigzag_monthly.csv", date_format="%Y-%m-%d")
        summary["regime_counts"].to_csv(events_dir / "zigzag_regime_counts.csv")
        write_meta(events_dir, df, args.input_csv, len(ev_df), len(ev_uniq))
        plots(ev_uniq, ev_df, df, outdir)

        curve, meta = best_combo_equity(ev_df)
        print(f"Events: grid rows={len(ev_df)}, deduped physical={len(ev_uniq)} -> {events_dir}/zigzag_events.csv")
        print(f"Best combo: th={meta['swing_th']}, X={meta['X']}, Y={meta['Y']}, final_return={meta['final_return']:.2%}, trades={meta['trades']}")
        print(f"Avg bars to rebound (deduped): {summary['avg_bars']:.2f}")
        print("Drop pct stats (deduped):\n", summary["drop_stats"])
        print("Rebound pct stats (deduped):\n", summary["rebound_stats"])
    else:
        print("No events detected.")


if __name__ == "__main__":
    main()
