"""CLI entry point.

    # backtest (CSV replay through SimBroker)
    python -m trading.run --mode backtest --feed data/TSLA_5m_60d.csv \
        --out outputs/trading_backtest

    # shadow against the REAL market (Alpaca REST polling, iex feed,
    # NullBroker journals signals/equity to <out>/journal.sqlite):
    python -m trading.run --mode shadow --live \
        --trigger 0.0173 --tp 0.0285 --sl 0.0148 --out outputs/shadow_live

    # bounded smoke run: stop after N bars / at the session close
    # (16:00 ET; 13:00 on NYSE half days via intel.market_calendar)
    python -m trading.run --mode shadow --live --max-bars 160 --once-session ...

Outputs: <out>/trades.csv (same columns as run_sim's sim_trades.csv, for
regression diffing via trading/tools/compare_trades.py), <out>/summary.txt,
<out>/journal.sqlite (orders/fills/equity_curve/bars/halt_state/runs).
Weekly forward-run report: python -m trading.tools.shadow_report --db ...

Note: --trigger/--tp/--sl override the config (with PyYAML missing the YAML
is ignored and defaults apply, so pass the candidate params explicitly).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trading.config.schema import Config
from trading.engine.runner import build

DEFAULT_CONFIG = str(Path(__file__).parent / "config" / "backtest.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified trading runner (backtest/shadow/paper/live).")
    ap.add_argument("--mode", default="backtest",
                    choices=["backtest", "shadow", "paper", "live"])
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="YAML config (defaults used if missing or PyYAML absent)")
    ap.add_argument("--feed", default=None, help="Override data_path (CSV)")
    ap.add_argument("--out", default=None, help="Override out_dir")
    # live feed (shadow mode)
    ap.add_argument("--live", action="store_true",
                    help="shadow: poll real-time Alpaca bars (iex) instead of CSV replay")
    ap.add_argument("--once-session", action="store_true",
                    help="live: exit after the session close (16:00 ET; 13:00 on NYSE "
                         "half days via intel.market_calendar) instead of waiting overnight")
    ap.add_argument("--max-bars", type=int, default=None,
                    help="live: stop after N bars (bounded smoke runs)")
    # strategy selection: e2 = 1H breakout-follow (default, existing launchd);
    # e8a = frozen E8-A+S2 candidate (models/e8a, docs/strategy-lab.md E11)
    ap.add_argument("--strategy", default="e2", choices=["e2", "e8a"],
                    help="Strategy line (default e2; e8a = E8-A+S2 GBDT V-rebound)")
    # strategy overrides (E2 candidate: --trigger 0.0173 --tp 0.0285 --sl 0.0148)
    ap.add_argument("--trigger", type=float, default=None, help="Override strategy.trigger")
    ap.add_argument("--tp", type=float, default=None, help="Override strategy.tp")
    ap.add_argument("--sl", type=float, default=None, help="Override strategy.sl")
    args = ap.parse_args()

    if args.live and args.mode != "shadow":
        ap.error("--live is only supported with --mode shadow in this phase")

    cfg = Config.from_yaml(args.config)
    if args.feed:
        cfg.data_path = args.feed
    if args.out:
        cfg.out_dir = args.out
    cfg.live = args.live
    cfg.once_session = args.once_session
    cfg.max_bars = args.max_bars
    cfg.strategy_name = args.strategy
    if args.trigger is not None:
        cfg.strategy.trigger = args.trigger
    if args.tp is not None:
        cfg.strategy.tp = args.tp
    if args.sl is not None:
        cfg.strategy.sl = args.sl

    runner = build(args.mode, cfg)
    try:
        runner.run()
    except Exception as exc:
        # P0-1/P1-9: a crashed shadow session must still leave a heartbeat
        if args.mode == "shadow":
            runner.write_status(error=repr(exc))
        raise


if __name__ == "__main__":
    main()
