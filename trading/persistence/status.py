"""Shadow-session status summary: outputs/shadow_status.json.

Product-review P0-1 / P1-9 ("空目录无人报警"): every shadow session ends by
writing a machine-readable heartbeat that the intel dashboard consumes.
One file, one key per strategy, merge-updated atomically so the E2 and E8-A
shadow jobs never clobber each other.

Schema (per strategy key):
    strategy        "e2" | "e8a"
    mode            runner mode ("shadow")
    live            true = real-time Alpaca feed, false = CSV replay
    out_dir         journal/trades directory of that strategy's shadow line
    session_start   ISO-8601 UTC, this session's start
    session_end     ISO-8601 UTC, when this summary was written
    bars            bars consumed this session
    gap_count       feed gaps this session
    signals         entry signals (intents that became journaled orders)
    last_signal_at  ISO-8601 UTC of the last entry order, or null
    halted          risk circuit-breaker state at session end
    s2              e8a only: {off, dd_pct, asof, threshold, n_days} or null
    error           null on clean exit, else the exception repr
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATUS_PATH = Path("outputs") / "shadow_status.json"


def write_shadow_status(entry: dict, strategy: str,
                        path: Optional[Path] = None) -> None:
    """Merge `entry` under `strategy` into the status file (atomic replace).
    Never raises — status reporting must not take down a trading session."""
    p = Path(path) if path else STATUS_PATH
    try:
        doc: dict = {"strategies": {}}
        if p.exists():
            try:
                old = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(old.get("strategies"), dict):
                    doc = old
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file: rebuild from scratch
        doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        doc["strategies"][strategy] = entry
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
        print(f"[status] shadow status written -> {p} (strategy={strategy})")
    except Exception as exc:  # noqa: BLE001
        print(f"[status] failed to write shadow status: {exc!r}")
