# 交易系统架构设计：回测 / 模拟 / 实盘统一框架

> 设计日期：2026-07-23
> 目标：**策略代码不知道自己在回测还是实盘**。回测 → paper → 实盘只换两个适配器（DataFeed、Broker），信号、风控、持仓管理是同一份代码。

## 0. 四种运行形态

| 形态 | DataFeed | Broker | Clock | 用途 |
|---|---|---|---|---|
| `backtest` | HistoricalFeed（CSV/Parquet 回放） | SimBroker（悲观成交模型） | 虚拟时钟（数据驱动） | 策略研究、参数寻优 |
| `shadow` | LiveFeed（实时订阅） | NullBroker（只记录不下单） | 真实时钟 | 影子对账：验证实时信号 = 回测预期 |
| `paper` | LiveFeed | AlpacaBroker(paper=True) | 真实时钟 | 验证整条下单链路 |
| `live` | LiveFeed | AlpacaBroker(paper=False) | 真实时钟 | 小仓位实盘 |

四种形态用同一个 `engine.Runner`，只在装配（wiring）时注入不同实现。

## 1. 目录结构（目标形态）

```
trading/
├── core/
│   ├── types.py          # Bar, Order, Fill, Position, AccountState（全系统共用数据类型）
│   ├── events.py         # BarEvent, SignalEvent, OrderEvent, FillEvent, ErrorEvent
│   └── clock.py          # Clock 协议：SimClock（数据驱动）/ WallClock
├── data/
│   ├── feed.py           # DataFeed 协议
│   ├── historical.py     # HistoricalFeed：CSV/Parquet 回放
│   ├── live_alpaca.py    # AlpacaLiveFeed：websocket 订阅
│   ├── live_poll.py      # PollingFeed：REST 轮询兜底（yfinance 等无流式源）
│   └── bar_builder.py    # 时间戳语义显式化的 K 线聚合
├── strategy/
│   ├── base.py           # Strategy 协议（纯函数式，无副作用）
│   └── v_reversal.py     # V 形反转策略（从现有 signal.py 迁移）
├── risk/
│   └── manager.py        # RiskManager：仓位规模、熔断、日亏损上限
├── portfolio/
│   └── portfolio.py      # 持仓状态机 + 账户核算
├── broker/
│   ├── base.py           # Broker 协议
│   ├── sim.py            # SimBroker：回测成交模型（跳空/滑点/手续费）
│   ├── alpaca.py         # AlpacaBroker：paper/live 同一份代码
│   └── null.py           # NullBroker：shadow 模式，只记录
├── persistence/
│   └── store.py          # SQLite：订单/成交/持仓/权益曲线，重启恢复
├── engine/
│   └── runner.py         # 主循环：四种形态共用
├── research/             # 现有 src/ 的统计与研究脚本迁入（v_stats 等）
└── config/
    ├── schema.py         # 类型化配置（pydantic 或 dataclass + 校验）
    └── *.yaml            # backtest.yaml / paper.yaml / live.yaml
```

## 2. 核心类型（core/types.py）

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

@dataclass(frozen=True)
class Bar:
    symbol: str
    start: datetime        # tz-aware UTC。语义固定：bar 覆盖 [start, start+duration)
    duration_s: int        # 300 = 5m
    open: float; high: float; low: float; close: float
    volume: float
    n_ticks: int = 0       # 聚合自多少根子 bar/tick；残缺 bar 检测用
    complete: bool = True  # LiveFeed 中未收盘的 bar 标记为 False，策略永远只吃 complete

class OrderSide(Enum):  BUY = "buy"; SELL = "sell"
class OrderType(Enum):  MARKET = "market"; LIMIT = "limit"; STOP = "stop"
class OrderStatus(Enum):
    PENDING = "pending"; SUBMITTED = "submitted"; PARTIAL = "partial"
    FILLED = "filled"; CANCELLED = "cancelled"; REJECTED = "rejected"

@dataclass
class Order:
    id: str                # 本地幂等 ID（uuid），broker_id 另存
    symbol: str
    side: OrderSide
    qty: float             # 股数（美元化仓位在 RiskManager 里换算，Order 只认股数）
    type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    broker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    tag: str = ""          # 归因：由哪个信号/哪层网格产生

@dataclass(frozen=True)
class Fill:
    order_id: str
    ts: datetime
    qty: float             # 带符号：买正卖负
    price: float           # 实际成交价（SimBroker 在此实现悲观假设）
    fee: float

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class AccountState:
    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
```

设计要点：
- `Bar.start` 语义**在类型层写死**为 bar 起始时刻（yfinance 约定），解决审查发现的 A2/P0-4 类边界错位——所有聚合、重采样必须经 `bar_builder`，禁止散落的 `df.resample()`。
- `Order.id` 本地生成、幂等：重启后靠它和 broker 对账，避免重复下单。
- `Fill.price` 与订单价分离：滑点/跳空穿价体现在 Fill，策略层永远看不到"理想成交价"。

## 3. 数据层（data/feed.py）

```python
from typing import Protocol, Iterator, Callable

class DataFeed(Protocol):
    def subscribe(self, symbol: str, bar_seconds: int) -> None: ...
    def bars(self) -> Iterator[Bar]:
        """统一入口：阻塞迭代器，产出已完成（complete=True）的 bar。
        - HistoricalFeed：按时间序回放，虚拟时钟随之推进
        - LiveFeed：websocket/轮询驱动，收盘才 yield
        迭代器结束 = 数据源枯竭（回测）或收到停止信号（实盘）。"""
    def health(self) -> "FeedHealth": ...   # 最后心跳时间、缺口计数、延迟

class FeedHealth:
    last_bar_at: datetime
    gap_count: int          # 检测到的缺失 bar 数
    lag_ms: float           # 实时源：bar 收盘到收到的延迟
```

实现要求（对应审查发现）：
- **HistoricalFeed**：加载时统一 `pd.to_datetime(utc=True)`（修 A3 的 7 处副本）；启动断言绝大多数 bar 落在 ET 9:30–16:00（修 P1-15 静默时区错误）。
- **LiveFeed（Alpaca websocket 为主，PollingFeed 兜底）**：断线自动重连 + 补拉缺口（REST 回填）；重连不上时发 `ErrorEvent`，由 Runner 决定是否平仓避险。
- **缺口语义**：`bars()` 不静默跳过缺口——检测到缺 bar 时产出的下一根 bar 附带 gap 标记，策略/风控可选择跳过该信号（修"缺口窗口照常交易"）。

### bar_builder（data/bar_builder.py）

```python
def aggregate(bars_5m: list[Bar], target_seconds: int,
              session_start: time = time(9, 30)) -> list[Bar]:
    """5m -> 1h 等聚合。约定：
    - 输入 Bar.start 语义（bar 起始）已由类型保证
    - 桶边界从 session_start 对齐（9:30/10:30/...），closed=left
    - 跨交易日永不合桶（隔夜跳空不会混进"1H 收益"，修 P0-5/A2）
    - n_ticks 不足桶容量 80% 的桶标记 complete=False，策略默认忽略"""
```

## 4. 策略层（strategy/base.py）

```python
@dataclass(frozen=True)
class SignalIntent:
    """策略的输出：意图，不是订单。不含仓位大小——sizing 是风控的职责。"""
    symbol: str
    direction: int              # +1 / -1 / 0（0 = 平仓意图）
    kind: str                   # "entry" | "exit" | "scale_in"
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    ttl_bars: Optional[int] = None   # 超时平仓
    confidence: float = 1.0     # 预留：GBDT 过滤器输出的概率挂在这里
    tag: str = ""

class StrategyContext(Protocol):
    """策略能看到的全部世界——只读。"""
    def history(self, symbol: str, bar_seconds: int, n: int) -> list[Bar]: ...
    def position(self, symbol: str) -> Position: ...
    def now(self) -> datetime: ...          # 来自 Clock，回测=虚拟时间

class Strategy(Protocol):
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> list[SignalIntent]: ...
    def warmup_bars(self) -> int: ...       # 需要多少历史才能出信号
```

设计要点：
- `on_bar` **纯函数式**：不下单、不改状态、不知道 broker 存在。这保证回测和实盘跑的是逐字节相同的策略逻辑。
- `ctx.history()` 只回看，接口上不可能拿到未来数据——前视偏差（A1）从架构上被封死，而不是靠 code review。
- `confidence` 字段是给后续 GBDT 过滤器预留的挂载点（见 ai-algo-survey.md）：过滤器实现为 Strategy 的装饰器，对 intent 打分，低于阈值的丢弃——不侵入策略本身。

## 5. 风控层（risk/manager.py）

```python
@dataclass
class RiskLimits:
    max_position_pct: float = 0.30      # 单标的市值 ≤ 权益 30%
    per_trade_risk_pct: float = 0.005   # 单笔风险预算（到 SL 的损失）≤ 权益 0.5%
    daily_loss_stop_pct: float = 0.02   # 日亏 2% 停止开新仓
    max_drawdown_stop_pct: float = 0.10 # 回撤 10% 全局熔断（平仓 + 停机）
    max_open_orders: int = 4
    worst_case_loss_pct: float = 0.05   # 马丁/网格：满层全部成交并打 SL 的损失上限

class RiskManager:
    def size(self, intent: SignalIntent, account: AccountState,
             last_price: float) -> Optional[Order]:
        """意图 -> 定量订单。仓位 = 风险预算 / SL 距离（波动率倒数 sizing）。
        任一限额不满足返回 None 并记日志。网格类意图必须先通过
        worst_case_loss 检查（按美元计算满层损失，修 P0-3）。"""
    def on_fill(self, fill: Fill, account: AccountState) -> list[RiskAction]:
        """成交后复查：触发熔断时返回 [FlattenAll, Halt]。前置于新信号处理。"""
```

设计要点：
- 熔断是**前置检查 + 独立组件**（修 P2-13"事后 break"）：`Halt` 状态持久化，重启后仍然生效，人工确认才解除。
- 所有 sizing 以**美元/股数**为单位（修 P0-3 的无量纲乘数问题）。

## 6. 持仓状态机（portfolio/portfolio.py）

```python
class PositionState(Enum):
    FLAT = "flat"
    PENDING_ENTRY = "pending_entry"    # 入场单已提交未成交
    HOLDING = "holding"                # 有仓，TP/SL/TTL 挂单在场
    PENDING_EXIT = "pending_exit"      # 平仓单已提交
    # 迁移规则（每 symbol 一个状态机）：
    # FLAT --entry order submitted--> PENDING_ENTRY --fill--> HOLDING
    # PENDING_ENTRY --cancel/reject/timeout--> FLAT
    # HOLDING --exit submitted--> PENDING_EXIT --fill--> FLAT
    # HOLDING 中收到新 entry 信号：按策略配置 ignore | reverse（先平后反）
    # 日终（15:55 ET）：强制 HOLDING -> PENDING_EXIT（市价平仓）

class Portfolio:
    def state(self, symbol: str) -> PositionState: ...
    def apply_fill(self, fill: Fill) -> None: ...       # 唯一改仓位的入口
    def on_bar(self, bar: Bar) -> list[Order]:          # TTL 到期、日终强平检查
    def snapshot(self) -> AccountState: ...
```

设计要点：
- 修 P1-7：连续信号、重复开仓、日终遗留仓位在状态机里有确定行为，不再依赖"回测恰好逐窗口平仓"这一模拟特性。
- `apply_fill` 是仓位变更的唯一入口，且每次变更同步写 persistence——状态永远可从 Fill 流重建。

## 7. 券商适配层（broker/base.py）

```python
class Broker(Protocol):
    def submit(self, order: Order) -> None: ...       # 异步；结果经 on_fill/on_order_update 回调
    def cancel(self, order_id: str) -> None: ...
    def positions(self) -> dict[str, Position]: ...   # broker 侧真实持仓（对账用）
    def account(self) -> AccountState: ...
    def set_callbacks(self,
        on_fill: Callable[[Fill], None],
        on_order_update: Callable[[Order], None],
        on_error: Callable[[Exception], None]) -> None: ...
```

### SimBroker（broker/sim.py）——回测成交模型，悲观假设集中地

```python
@dataclass
class FillModel:
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    # 全部修自审查发现 P1-8/9/10、B2：
    # - STOP 单：exit = min(bar.open, stop_price) - slippage   （跳空穿价按开盘成交）
    # - LIMIT 止盈：需 high > limit（严格穿越）才成交；成交价 = max(bar.open, limit)
    # - MARKET：下一根 bar open ± slippage（含 1 bar 延迟）
    # - 同 bar 同时触及 TP 和 SL：默认按悲观顺序（先 SL）；可配 "ohlc_path" 模式
    # - 网格多层：按层价格逐层判定，平仓滑点随总仓位规模放大
```

### AlpacaBroker（broker/alpaca.py）

- `paper=True/False` 只改 base_url 和 key——**这就是"模拟跑完直接实测"的那一步**。
- 所有下单带本地幂等 ID（client_order_id），网络超时重试不会造成双重下单。
- 启动时 `reconcile()`：本地持久化状态 vs broker 实际持仓，不一致时拒绝启动并报告差异（人工确认）。

## 8. 持久化与可观测性（persistence/store.py）

```python
class Store:
    """SQLite，单文件。表：orders / fills / equity_curve / halt_state / runs"""
    def journal_order(self, order: Order): ...
    def journal_fill(self, fill: Fill): ...
    def snapshot_equity(self, ts: datetime, account: AccountState): ...
    def load_open_state(self) -> tuple[list[Order], dict[str, Position]]: ...
    # 崩溃恢复流程：load_open_state() -> broker.positions() 对账 -> 继续或人工介入
```

- 日志：结构化 JSON（一行一事件），trade-level 审计（信号→订单→成交全链路 tag 关联），修 P1-16。
- 回测与实盘写同一套表——回测结果分析、实盘复盘用同一份分析代码。

## 9. 主循环（engine/runner.py）

```python
class Runner:
    def __init__(self, feed: DataFeed, strategy: Strategy, risk: RiskManager,
                 portfolio: Portfolio, broker: Broker, store: Store, clock: Clock): ...

    def run(self) -> None:
        for bar in self.feed.bars():                  # 回测/实盘唯一差异：bar 从哪来
            self.portfolio.on_bar(bar)                # 1. TTL / 日终强平
            if self.risk.halted: continue             # 2. 熔断检查
            intents = self.strategy.on_bar(bar, self.ctx)   # 3. 纯函数出意图
            for intent in intents:
                order = self.risk.size(intent, self.portfolio.snapshot(), bar.close)
                if order and self.portfolio.state(order.symbol).allows(order):
                    self.store.journal_order(order)   # 4. 先落盘再下单
                    self.broker.submit(order)
        # broker 回调（异步）：fill -> portfolio.apply_fill -> store.journal_fill
        #                          -> risk.on_fill（熔断复查）
```

装配示例（config/paper.yaml + 入口）：

```python
def build(mode: str, cfg: Config) -> Runner:
    match mode:
        case "backtest": feed, broker = HistoricalFeed(cfg.data_path), SimBroker(cfg.fill_model)
        case "shadow":   feed, broker = AlpacaLiveFeed(cfg), NullBroker()
        case "paper":    feed, broker = AlpacaLiveFeed(cfg), AlpacaBroker(cfg, paper=True)
        case "live":     feed, broker = AlpacaLiveFeed(cfg), AlpacaBroker(cfg, paper=False)
    return Runner(feed, strategy, risk, portfolio, broker, store, clock)
```

## 10. 影子对账（shadow 模式的核心产出)

paper 之前必跑的验证：shadow 模式实时记录信号流；每日收盘后，用当天数据跑一遍 backtest 模式，逐信号 diff：

```
shadow_signals.jsonl  vs  backtest_signals.jsonl
→ 时间戳、方向、TP/SL 完全一致？不一致 = 回测器存在隐含假设（数据到达顺序、
  残缺 bar、时钟边界），修到零 diff 才有资格进 paper。
```

## 11. 迁移路径（从现有代码到此架构）

| 现有代码 | 去向 |
|---|---|
| `live_trading/signal.py` + `run_sim.py` 的信号逻辑 | `strategy/v_reversal.py`（修 P0-1、P1-6 后迁入） |
| `live_trading/bar_builder.py` | `data/bar_builder.py`（按第 3 节约定重写） |
| `live_trading/execution.py` + 各 run_sim 的成交判定 | `broker/sim.py` 的 FillModel（悲观假设集中实现） |
| `live_trading/run_sim_grid.py` 马丁逻辑 | `strategy/` 一个策略 + RiskManager 的 worst_case 检查（美元化，修 P0-3） |
| `live_trading/auto_opt.py` | `research/optimize.py`，底层调 backtest 模式 Runner，强制 walk-forward |
| `src/data_fetch.py` + 7 处 CSV 加载 | `data/historical.py` 单一实现 |
| `src/v_stats.py`、`zigzag_v.py` 等统计脚本 | `research/`（消费同一 `data/` 层，与交易链路解耦） |

实施顺序建议：**先修现有代码的 P0 bug（让当前回测数字先可信），再按本文搭骨架迁移**——骨架搭好前不接任何实时数据源；迁移完成的标志是：同一策略配置在旧 run_sim 和新 backtest Runner 上产出一致的交易流（差异仅来自已知的成交模型改进）。
