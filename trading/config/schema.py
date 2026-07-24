"""Typed configuration (stdlib dataclasses; PyYAML optional).

``Config.from_yaml`` degrades gracefully: if PyYAML is not installed or the
file is missing, defaults are used (a warning is printed). Unknown keys in
the YAML raise — silent typos in trading configs are how accounts die.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class FillCfg:
    fee_bp: float = 1.0
    slippage_bp: float = 2.0


@dataclass
class StrategyCfg:
    trigger: float = 0.015
    tp: float = 0.03
    sl: float = 0.01
    allowed_hours: tuple[int, ...] = (9, 10, 11, 12, 15)
    signal_seconds: int = 3600


@dataclass
class RiskCfg:
    # defaults mirror config/backtest.yaml (run_sim parity: full equity per
    # trade) so a missing YAML/PyYAML still reproduces the reference stream
    sizing: str = "full_equity"          # "risk_budget" | "full_equity"
    max_position_pct: float = 1.0
    per_trade_risk_pct: float = 0.005
    daily_loss_stop_pct: float = 0.02
    max_drawdown_stop_pct: float = 0.10
    max_open_orders: int = 4
    worst_case_loss_pct: float = 0.05


@dataclass
class Config:
    mode: str = "backtest"
    strategy_name: str = "e2"            # "e2" (1H breakout) | "e8a" (E8-A+S2)
    symbol: str = "TSLA"
    capital: float = 10_000.0
    bar_seconds: int = 300
    data_path: str = "data/TSLA_5m_60d.csv"
    out_dir: str = "outputs/trading_backtest"
    db_path: Optional[str] = None        # default: <out_dir>/journal.sqlite
    eod_flat_et: tuple[int, int] = (15, 55)
    fill: FillCfg = field(default_factory=FillCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)

    # Live feed (shadow --live): AlpacaPollingFeed instead of CSV replay
    live: bool = False
    once_session: bool = False           # live: return at 16:00 ET close
    max_bars: Optional[int] = None       # live: stop after N bars (smoke tests)

    # Alpaca (shell only in this phase; keys via env in real deployments)
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "Config":
        if not path or not Path(path).exists():
            if path:
                print(f"[config] {path} not found - using defaults")
            return cls()
        try:
            import yaml  # type: ignore
        except ImportError:
            print(f"[config] PyYAML not installed - ignoring {path}, using defaults")
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        def build(dc_cls, data: dict[str, Any]):
            names = {f.name: f for f in dataclasses.fields(dc_cls)}
            unknown = set(data) - set(names)
            if unknown:
                raise ValueError(f"{dc_cls.__name__}: unknown config keys {sorted(unknown)}")
            kwargs = {}
            for k, v in data.items():
                f = names[k]
                if dataclasses.is_dataclass(f.type) or f.name in ("fill", "strategy", "risk"):
                    sub = {"fill": FillCfg, "strategy": StrategyCfg, "risk": RiskCfg}[f.name]
                    kwargs[k] = build(sub, v or {})
                elif isinstance(v, list):
                    kwargs[k] = tuple(v)
                else:
                    kwargs[k] = v
            return dc_cls(**kwargs)

        return build(cls, raw)

    def resolved_db_path(self) -> str:
        return self.db_path or str(Path(self.out_dir) / "journal.sqlite")
