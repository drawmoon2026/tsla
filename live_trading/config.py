from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal

@dataclass
class Config:
    provider: Literal['alpaca_paper','ibkr_sim','sim_only'] = 'sim_only'
    symbol: str = 'TSLA'
    trigger: float = 0.015
    tp: float = 0.03
    sl: float = 0.01
    trigger_std: float | None = None  # k * rolling std of 1H returns
    std_window: int = 20
    capital: float = 10000.0
    offset_minutes: int = 30
    tz: str = 'America/New_York'
    slip_bp: float = 2.0
    fee_bp: float = 1.0
    allowed_hours: tuple[int, ...] = (9, 10, 11, 12, 15)  # ET hours eligible for entry

    # account-level risk controls
    daily_loss_stop: float = 0.02  # stop opening new trades for the day past this loss

    # grid/martin parameters
    grid_steps: tuple[float, ...] = (-0.005, -0.01, -0.015, -0.02)  # relative steps from anchor (long); sign flipped for short
    grid_mults: tuple[float, ...] = (1, 2, 4, 8)  # position size multipliers per level
    grid_base_dollars: float | None = None  # dollars at mult=1; default capital/sum(mults) (full ladder = capital)
    grid_tp: float = 0.015  # take profit relative to average cost
    grid_stop: float = 0.02  # stop loss relative to average cost
    grid_timeout_bars: int = 12  # bars (5m) until forced close
    global_stop: float = 0.10  # max account drawdown before flatten
    max_leverage: float = 1.0  # 1 = no leverage

    alpaca_key: str = os.getenv('ALPACA_KEY_ID','')
    alpaca_secret: str = os.getenv('ALPACA_SECRET_KEY','')
    alpaca_base: str = os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets')

    def __post_init__(self):
        if self.grid_base_dollars is None:
            self.grid_base_dollars = self.capital / sum(self.grid_mults)
        self.validate()

    def validate(self):
        # 负 trigger 会让 abs(ret) < trigger 恒 False，静默变成"每小时无条件开仓"
        if self.trigger < 0:
            raise ValueError(f"trigger must be >= 0, got {self.trigger} (negative trigger means 'always trade')")
        if not (0 < self.sl < 1):
            raise ValueError(f"sl must be in (0, 1), got {self.sl}")
        if not (0 < self.tp < 1):
            raise ValueError(f"tp must be in (0, 1), got {self.tp}")
        if self.trigger_std is not None and self.trigger_std <= 0:
            raise ValueError(f"trigger_std must be > 0 when set, got {self.trigger_std}")
        if not (0 < self.grid_tp < 1) or not (0 < self.grid_stop < 1):
            raise ValueError("grid_tp and grid_stop must be in (0, 1)")
        if self.grid_base_dollars is not None and self.grid_base_dollars <= 0:
            raise ValueError(f"grid_base_dollars must be > 0, got {self.grid_base_dollars}")
        if not (0 < self.daily_loss_stop < 1):
            raise ValueError(f"daily_loss_stop must be in (0, 1), got {self.daily_loss_stop}")

def load_config():
    return Config()
