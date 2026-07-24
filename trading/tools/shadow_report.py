"""Forward-run report from a shadow journal.sqlite (Phase 5 weekly report).

    .venv/bin/python -m trading.tools.shadow_report \
        --db outputs/shadow_live/journal.sqlite [--run <run_id prefix>]

Reads the tables the shadow Runner writes every bar (runs / bars / orders /
equity_curve) and prints, per shadow run and in aggregate:

- coverage: calendar days since the first run started, trading days and
  bar counts actually received;
- feed health: gap count and receive lag (from the per-bar health snapshot
  journaled with each bar; lag_ms > 0 marks live-polled bars, backfilled
  bars carry the previous value or 0);
- the signal stream (entry orders journaled by the NullBroker path);
- hypothetical fills & PnL: each entry is replayed against the journaled
  bars with the SAME pessimistic execution model as the backtest
  (src.common.execution: next-bar-open entry with adverse slippage, strict
  TP crossing, gap-through SL, SL-priority, fees both sides), TTL and
  15:55 ET end-of-day flatten included. Overlapping signals are skipped
  (a real position would have blocked them), so the trade stream matches
  what a backtest over the same bars would produce.

This is a REPORT, not a broker: numbers are hypothetical, on iex-feed bars
(see trading/data/live_alpaca.py for the iex-vs-sip caveat).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.execution import CostModel, entry_fill, settle_bracket

_ET = ZoneInfo("America/New_York")


@dataclass
class BarRow:
    start: datetime
    duration_s: int
    open: float
    high: float
    low: float
    close: float
    gap_count: int
    lag_ms: float


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _cfg_get(cfg: dict, path: list[str], default):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_bars(db: sqlite3.Connection, run_id: str) -> list[BarRow]:
    rows = db.execute(
        "SELECT start, duration_s, open, high, low, close, gap_count, lag_ms"
        " FROM bars WHERE run_id=? ORDER BY start", (run_id,)).fetchall()
    return [BarRow(_parse_ts(r[0]), int(r[1]), *map(float, r[2:6]),
                   int(r[6] or 0), float(r[7] or 0.0)) for r in rows]


def _simulate(order: dict, bars: list[BarRow], cfg: dict) -> Optional[dict]:
    """Replay one entry order against journaled bars. None = no next bar yet."""
    d = 1 if order["side"] == "buy" else -1
    dur = bars[0].duration_s if bars else 300
    # Entry bar: the strategy tag pins the execution-window start exactly
    # ("vrev:%Y-%m-%dT%H:%M" in ET = end of the signal bar). Fall back to
    # created_at (wall/sim clock ~= signal bar end) for foreign tags.
    m = re.search(r":(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})", order["tag"] or "")
    if m:
        anchor = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M").replace(tzinfo=_ET)
        idx = next((i for i, b in enumerate(bars) if b.start >= anchor), None)
    else:
        created = _parse_ts(order["created_at"])
        idx = next((i for i, b in enumerate(bars)
                    if b.start > created - timedelta(seconds=dur)), None)
    if idx is None:
        return None
    entry_bar = bars[idx]
    cost = CostModel(
        fee_bp=_cfg_get(cfg, ["fill", "fee_bp"], 1.0),
        slippage_bp=_cfg_get(cfg, ["fill", "slippage_bp"], 2.0),
    )
    tp_pct = _cfg_get(cfg, ["strategy", "tp"], 0.03)
    sl_pct = _cfg_get(cfg, ["strategy", "sl"], 0.01)
    signal_s = _cfg_get(cfg, ["strategy", "signal_seconds"], 3600)
    eod_h, eod_m = tuple(_cfg_get(cfg, ["eod_flat_et"], (15, 55)))

    deadline = entry_bar.start + timedelta(seconds=signal_s)
    day = entry_bar.start.astimezone(_ET).date()
    window = [b for b in bars[idx:]
              if b.start < deadline
              and b.start.astimezone(_ET).date() == day
              and (b.start.astimezone(_ET).hour, b.start.astimezone(_ET).minute)
              < (eod_h, eod_m)]
    if not window:
        return None
    entry_px = entry_fill(window[0].open, d, cost)
    df = pd.DataFrame(
        {"Open": [b.open for b in window], "High": [b.high for b in window],
         "Low": [b.low for b in window], "Close": [b.close for b in window]},
        index=[b.start for b in window])
    res = settle_bracket(df, d, entry_px=entry_px,
                         tp_px=entry_px * (1 + tp_pct * d),
                         sl_px=entry_px * (1 - sl_pct * d), cost=cost)
    return {
        "entry_time": window[0].start, "exit_time": res.exit_time,
        "direction": d, "entry": entry_px, "exit": res.exit_px,
        "hit": res.hit, "ret": res.ret, "tag": order["tag"],
    }


def _fmt_et(ts: datetime) -> str:
    return ts.astimezone(_ET).strftime("%m-%d %H:%M")


def report(db_path: str, run_filter: Optional[str] = None) -> None:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    runs = db.execute(
        "SELECT run_id, started_at, mode, config_json FROM runs"
        " WHERE mode='shadow' ORDER BY started_at").fetchall()
    if run_filter:
        runs = [r for r in runs if r[0].startswith(run_filter)]
    if not runs:
        print(f"No shadow runs found in {db_path}")
        return

    print("=" * 72)
    print(f"Shadow forward-run report — {db_path}")
    print(f"generated {datetime.now(_ET):%Y-%m-%d %H:%M ET}")
    print("=" * 72)

    all_days: set = set()
    all_trades: list[dict] = []
    total_signals = 0
    total_gaps = 0
    first_started: Optional[datetime] = None
    capital = 10_000.0

    for run_id, started_at, _mode, config_json in runs:
        try:
            cfg = json.loads(config_json)
            if not isinstance(cfg, dict) or not isinstance(cfg.get("strategy"), dict):
                cfg = {}
        except (TypeError, json.JSONDecodeError):
            cfg = {}
        started = _parse_ts(started_at)
        first_started = min(first_started or started, started)
        capital = float(cfg.get("capital", capital) or capital)
        bars = _load_bars(db, run_id)
        days = sorted({b.start.astimezone(_ET).date() for b in bars})
        all_days.update(days)
        live_bars = [b for b in bars if b.lag_ms > 0]
        gaps = max((b.gap_count for b in bars), default=0)
        total_gaps += gaps

        entries = [dict(zip(("id", "created_at", "side", "tag"), r))
                   for r in db.execute(
                       "SELECT id, created_at, side, tag FROM orders"
                       " WHERE run_id=? AND type='market'"
                       " AND tag NOT IN ('tp','sl','ttl_close','eod')"
                       " ORDER BY created_at", (run_id,))]
        total_signals += len(entries)
        n_fills = db.execute("SELECT COUNT(*) FROM fills WHERE run_id=?",
                             (run_id,)).fetchone()[0]

        strat = cfg.get("strategy", {})
        print(f"\nRun {run_id[:12]}…  started {started:%Y-%m-%d %H:%M%z}  "
              f"live={_cfg_get(cfg, ['live'], '?')}")
        print(f"  params: trigger={strat.get('trigger', '?')} tp={strat.get('tp', '?')} "
              f"sl={strat.get('sl', '?')} signal={strat.get('signal_seconds', '?')}s "
              f"capital={cfg.get('capital', '?')}")
        if days:
            print(f"  bars:   {len(bars)} over {len(days)} trading day(s) "
                  f"({days[0]} .. {days[-1]}), live-polled: {len(live_bars)}")
        else:
            print("  bars:   0 (no bars journaled)")
        lag_txt = "n/a"
        if live_bars:
            lags = [b.lag_ms for b in live_bars]
            lag_txt = f"mean {sum(lags)/len(lags)/1000:.1f}s / max {max(lags)/1000:.1f}s"
        print(f"  feed:   gaps={gaps}  live receive lag: {lag_txt}")
        print(f"  signals: {len(entries)} entries journaled, "
              f"broker fills: {n_fills} (NullBroker: expected 0)")

        # hypothetical replay (skip signals that overlap an open trade)
        busy_until: Optional[datetime] = None
        for o in entries:
            t = _simulate(o, bars, cfg)
            if t is None:
                print(f"    {_fmt_et(_parse_ts(o['created_at']))} ET  "
                      f"{o['side'].upper():4s} {o['tag']}  -> pending (no bar yet)")
                continue
            if busy_until is not None and t["entry_time"] < busy_until:
                print(f"    {_fmt_et(t['entry_time'])} ET  {o['side'].upper():4s} "
                      f"{o['tag']}  -> skipped (position open)")
                continue
            busy_until = t["exit_time"] + timedelta(seconds=bars[0].duration_s)
            all_trades.append(t)
            print(f"    {_fmt_et(t['entry_time'])} ET  {o['side'].upper():4s} "
                  f"{o['tag']}  -> {t['hit']} @ {t['exit']:.2f}  "
                  f"ret={t['ret']:+.2%}  (exit {_fmt_et(t['exit_time'])} ET)")

    # -- aggregate ---------------------------------------------------------
    equity = capital
    for t in all_trades:
        equity *= 1 + t["ret"]
    wins = sum(1 for t in all_trades if t["ret"] > 0)
    hits = {k: sum(1 for t in all_trades if t["hit"] == k) for k in ("tp", "sl", "close")}
    elapsed = (datetime.now(first_started.tzinfo) - first_started).days if first_started else 0

    print("\n" + "=" * 72)
    print(f"Aggregate over {len(runs)} shadow run(s)")
    print(f"  forward window: {elapsed} calendar day(s) since first run start; "
          f"{len(all_days)} trading day(s) of bars")
    print(f"  feed gaps (sum of per-run totals): {total_gaps}")
    print(f"  signals: {total_signals}   hypothetical trades: {len(all_trades)}")
    if all_trades:
        print(f"  hypothetical PnL (full-equity compounding, {capital:.0f} start): "
              f"{equity:.2f}  ({equity / capital - 1:+.2%})")
        print(f"  win rate: {wins / len(all_trades):.0%}   "
              f"tp/sl/close: {hits['tp']}/{hits['sl']}/{hits['close']}")
    else:
        print("  no completed hypothetical trades yet")
    print("  (hypothetical fills use the backtest execution model on iex bars —")
    print("   see trading/data/live_alpaca.py for the iex-vs-sip caveat)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Shadow forward-run report from journal.sqlite")
    ap.add_argument("--db", default="outputs/shadow_live/journal.sqlite")
    ap.add_argument("--run", default=None, help="Filter to run_id prefix")
    args = ap.parse_args()
    report(args.db, args.run)


if __name__ == "__main__":
    main()
