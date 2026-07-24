"""AlpacaBroker interface shell — NOT implemented in this phase.

Design (architecture section 7): paper and live are the SAME code path;
the only differences are the base URL and the API key pair:

    paper:  https://paper-api.alpaca.markets   (paper=True)
    live:   https://api.alpaca.markets          (paper=False)

That swap is the whole "paper-validated -> go live" step. Implementation
requirements when this shell is filled in:

- every ``submit`` sends ``order.id`` as Alpaca ``client_order_id`` so
  network-timeout retries are idempotent (no double orders);
- ``reconcile()`` on startup: persisted local state vs ``positions()``;
  refuse to run on mismatch until a human confirms;
- fills/updates arrive over the trade-updates websocket and are forwarded
  to the callbacks registered via ``set_callbacks``.
"""

from __future__ import annotations

from typing import Callable

from trading.core.types import AccountState, Bar, Fill, Order, Position

_MSG = "AlpacaBroker requires alpaca-py + API keys (not part of this phase)"


class AlpacaBroker:
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, key_id: str = "", secret_key: str = "", paper: bool = True) -> None:
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        self.paper = paper
        raise NotImplementedError(_MSG)

    def submit(self, order: Order) -> None:
        raise NotImplementedError(_MSG)

    def cancel(self, order_id: str) -> None:
        raise NotImplementedError(_MSG)

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError(_MSG)

    def account(self) -> AccountState:
        raise NotImplementedError(_MSG)

    def set_callbacks(
        self,
        on_fill: Callable[[Fill], None],
        on_order_update: Callable[[Order], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        raise NotImplementedError(_MSG)

    def reconcile(self) -> None:
        """Compare persisted local state against broker truth; refuse to
        start on mismatch (human confirmation required)."""
        raise NotImplementedError(_MSG)

    def on_bar(self, bar: Bar) -> None:
        pass  # live fills come from the websocket, not from bars
