"""Broker protocol (docs/architecture.md section 7).

Submission is asynchronous: results arrive via the registered callbacks.
Simulated brokers additionally expose ``on_bar(bar)`` (called by the Runner
each bar) to drive fills off replayed data; live brokers ignore it — their
fills arrive from the venue's stream.
"""

from __future__ import annotations

from typing import Callable, Protocol

from trading.core.types import AccountState, Bar, Fill, Order, Position


class Broker(Protocol):
    def submit(self, order: Order) -> None: ...
    def cancel(self, order_id: str) -> None: ...
    def positions(self) -> dict[str, Position]: ...   # broker-side truth (reconcile)
    def account(self) -> AccountState: ...
    def set_callbacks(
        self,
        on_fill: Callable[[Fill], None],
        on_order_update: Callable[[Order], None],
        on_error: Callable[[Exception], None],
    ) -> None: ...
    def on_bar(self, bar: Bar) -> None: ...           # no-op for live brokers
