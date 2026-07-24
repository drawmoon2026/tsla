"""NullBroker: shadow mode — records every order, never fills anything.

The shadow run's product is the order/signal journal, which is diffed
against a backtest over the same session (architecture section 10).
"""

from __future__ import annotations

from typing import Callable

from trading.core.types import AccountState, Bar, Fill, Order, OrderStatus, Position


class NullBroker:
    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.cancelled: list[str] = []

    def set_callbacks(
        self,
        on_fill: Callable[[Fill], None],
        on_order_update: Callable[[Order], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        pass  # never calls back: nothing ever fills

    def submit(self, order: Order) -> None:
        order.status = OrderStatus.SUBMITTED
        self.submitted.append(order)

    def cancel(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    def positions(self) -> dict[str, Position]:
        return {}

    def account(self) -> AccountState:
        return AccountState(cash=0.0, equity=0.0)

    def on_bar(self, bar: Bar) -> None:
        pass
