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
    trigger_std: float = None  # k * rolling std of 1H returns
    std_window: int = 20
    capital: float = 10000.0
    offset_minutes: int = 30
    tz: str = 'America/New_York'
    slip_bp: float = 0.0
    fee_bp: float = 0.0
    allowed_hours: tuple[int, ...] = (9, 10, 11, 12, 15)  # ET hours eligible for entry

    # grid/martin parameters
    grid_steps: tuple[float, ...] = (-0.005, -0.01, -0.015, -0.02)  # relative steps from anchor (long); sign flipped for short
    grid_mults: tuple[float, ...] = (1, 2, 4, 8)  # position size multipliers per level
    grid_tp: float = 0.015  # take profit relative to average cost
    grid_stop: float = 0.02  # stop loss relative to average cost
    grid_timeout_bars: int = 12  # bars (5m) until forced close
    global_stop: float = 0.10  # max account drawdown before flatten
    max_leverage: float = 1.0  # 1 = no leverage

    alpaca_key: str = os.getenv('ALPACA_KEY_ID','')
    alpaca_secret: str = os.getenv('ALPACA_SECRET_KEY','')
    alpaca_base: str = os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets')

def load_config():
    return Config()
