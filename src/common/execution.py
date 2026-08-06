"""Shared bracket-order settlement with pessimistic fill assumptions.

Every simulator in this repo must settle TP/SL through here so the execution
model lives in exactly one auditable place. Assumptions (deliberately
pessimistic — see docs/review-2026-07-23/):

- Stop loss is a stop-market order: if a bar OPENS through the stop, the fill
  is the open (gap-through), not the stop price; adverse slippage on top.
- Take profit is a limit order: it needs a strict crossing (high > tp for
  longs) to fill at tp, or fills at the open when the bar opens beyond it.
  A bar that OPENS at/beyond TP fills the limit at the open immediately —
  an SL touched later inside that same bar cannot pre-empt it (rebuttal
  review 2026-08-06 R2 fix; zero historical occurrences in all archived
  streams, so no archived number moves).
- Otherwise, if one bar touches both TP and SL, SL wins (worst case for
  the trader).
- Fees are charged per side; entry gets adverse slippage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    fee_bp: float = 1.0        # per side, basis points of notional
    slippage_bp: float = 2.0   # adverse, applied to entry and stop exits

    @property
    def fee(self) -> float:
        return self.fee_bp / 10_000

    @property
    def slip(self) -> float:
        return self.slippage_bp / 10_000


@dataclass(frozen=True)
class BracketResult:
    hit: str            # "tp" | "sl" | "close"
    exit_px: float      # fill price after slippage
    exit_time: pd.Timestamp
    ret: float          # signed net return incl. fees, relative to raw entry


def entry_fill(raw_open: float, direction: int, cost: CostModel) -> float:
    """Market entry at next bar open with adverse slippage."""
    return raw_open * (1 + cost.slip * direction)


def settle_bracket(
    window: pd.DataFrame,
    direction: int,
    entry_px: float,
    tp_px: float,
    sl_px: float,
    cost: CostModel,
) -> BracketResult:
    """Walk `window` (OHLC bars after entry) and settle a TP/SL bracket.

    `window` must start at the entry bar. Exit at last bar close if neither
    level is hit ("close").
    """
    if window.empty:
        raise ValueError("settle_bracket needs a non-empty window")

    hit = "close"
    exit_px = float(window["Close"].iloc[-1])
    exit_time = window.index[-1]

    for ts, row in window.iterrows():
        o, h, l = float(row["Open"]), float(row["High"]), float(row["Low"])
        if direction == 1:
            gap_tp = o >= tp_px
            gap_sl = o <= sl_px
            sl_hit = gap_sl or l <= sl_px
            tp_hit = gap_tp or h > tp_px
            sl_fill = min(o, sl_px) * (1 - cost.slip)
            tp_fill = max(o, tp_px)
        else:
            gap_tp = o <= tp_px
            gap_sl = o >= sl_px
            sl_hit = gap_sl or h >= sl_px
            tp_hit = gap_tp or l < tp_px
            sl_fill = max(o, sl_px) * (1 + cost.slip)
            tp_fill = min(o, tp_px)

        if gap_tp:  # bar opens at/beyond the TP limit: fills at the open
            # immediately; an intrabar SL later in the bar cannot pre-empt it
            hit, exit_px, exit_time = "tp", tp_fill, ts
            break
        if sl_hit:  # SL priority: pessimistic when both touch in one bar
            hit, exit_px, exit_time = "sl", sl_fill, ts
            break
        if tp_hit:
            hit, exit_px, exit_time = "tp", tp_fill, ts
            break

    ret = direction * (exit_px - entry_px) / entry_px - 2 * cost.fee
    return BracketResult(hit=hit, exit_px=exit_px, exit_time=exit_time, ret=ret)
