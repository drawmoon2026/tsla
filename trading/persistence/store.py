"""SQLite journal: orders / fills / equity_curve / halt_state / runs.

One file, stdlib sqlite3. Backtest and live write the SAME tables so one
set of analysis code serves both. Crash recovery: ``load_open_state()``
returns still-open orders and fill-reconstructed positions, which the
engine reconciles against ``broker.positions()`` before resuming.

Timestamps are stored as ISO-8601 UTC strings (bar-START semantics).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from trading.core.types import Fill, Order, OrderSide, OrderStatus, OrderType, Position

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, mode TEXT, config_json TEXT);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, symbol TEXT,
    side TEXT, qty REAL, type TEXT, limit_price REAL, stop_price REAL,
    status TEXT, broker_id TEXT, tag TEXT, notional REAL, tif TEXT);
CREATE TABLE IF NOT EXISTS fills (
    order_id TEXT, ts TEXT, qty REAL, price REAL, fee REAL, run_id TEXT);
CREATE TABLE IF NOT EXISTS equity_curve (
    run_id TEXT, ts TEXT, cash REAL, equity REAL);
CREATE TABLE IF NOT EXISTS halt_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, reason TEXT, active INTEGER);
CREATE TABLE IF NOT EXISTS bars (
    run_id TEXT, symbol TEXT, start TEXT, duration_s INTEGER,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    gap_count INTEGER, lag_ms REAL);
"""


class Store:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._run_id: Optional[str] = None

    def start_run(self, run_id: str, started_at: datetime, mode: str, config: dict) -> None:
        self._run_id = run_id
        self._db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?)",
            (run_id, started_at.isoformat(), mode, json.dumps(config, default=str)),
        )
        self._db.commit()

    def journal_order(self, order: Order) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (order.id, self._run_id,
             order.created_at.isoformat() if order.created_at else None,
             order.symbol, order.side.value, order.qty, order.type.value,
             order.limit_price, order.stop_price, order.status.value,
             order.broker_id, order.tag, order.notional, order.tif),
        )
        self._db.commit()

    def journal_fill(self, fill: Fill) -> None:
        self._db.execute(
            "INSERT INTO fills VALUES (?,?,?,?,?,?)",
            (fill.order_id, fill.ts.isoformat(), fill.qty, fill.price,
             fill.fee, self._run_id),
        )
        self._db.commit()

    def journal_bar(self, bar, gap_count: int = 0, lag_ms: float = 0.0) -> None:
        """Journal one market bar plus the feed-health snapshot at receipt
        (shadow/paper/live modes: the forward-run report replays these)."""
        self._db.execute(
            "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self._run_id, bar.symbol, bar.start.isoformat(), bar.duration_s,
             bar.open, bar.high, bar.low, bar.close, bar.volume,
             gap_count, lag_ms),
        )
        self._db.commit()

    def snapshot_equity(self, ts: datetime, cash: float, equity: float) -> None:
        self._db.execute(
            "INSERT INTO equity_curve VALUES (?,?,?,?)",
            (self._run_id, ts.isoformat(), cash, equity),
        )
        self._db.commit()

    # -- halt persistence (survives restarts; cleared manually) ------------
    def set_halt(self, ts: datetime, reason: str) -> None:
        self._db.execute("INSERT INTO halt_state (ts, reason, active) VALUES (?,?,1)",
                         (ts.isoformat(), reason))
        self._db.commit()

    def get_halt(self) -> Optional[str]:
        row = self._db.execute(
            "SELECT reason FROM halt_state WHERE active=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def clear_halt(self) -> None:
        self._db.execute("UPDATE halt_state SET active=0")
        self._db.commit()

    # -- crash recovery ----------------------------------------------------
    def load_open_state(self) -> tuple[list[Order], dict[str, Position]]:
        """Open (submitted/partial) orders + positions rebuilt from fills."""
        orders: list[Order] = []
        for r in self._db.execute(
            "SELECT id, created_at, symbol, side, qty, type, limit_price,"
            " stop_price, status, broker_id, tag, notional, tif FROM orders"
            " WHERE status IN ('submitted','partial')"
        ):
            orders.append(Order(
                id=r[0],
                created_at=datetime.fromisoformat(r[1]) if r[1] else None,
                symbol=r[2], side=OrderSide(r[3]), qty=r[4], type=OrderType(r[5]),
                limit_price=r[6], stop_price=r[7], status=OrderStatus(r[8]),
                broker_id=r[9], tag=r[10], notional=r[11], tif=r[12],
            ))
        positions: dict[str, Position] = {}
        for sym, qty in self._db.execute(
            "SELECT o.symbol, SUM(f.qty) FROM fills f JOIN orders o"
            " ON f.order_id = o.id GROUP BY o.symbol"
        ):
            if qty:
                positions[sym] = Position(symbol=sym, qty=qty)
        return orders, positions

    def close(self) -> None:
        self._db.close()
