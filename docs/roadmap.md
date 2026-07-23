# 优化路线图（2026-07-23）

综合三份审查/调研报告：
- [src/ 代码审查](review-2026-07-23/src-review.md)（代理 A）
- [live_trading/ 代码审查](review-2026-07-23/live-trading-review.md)（代理 B）
- [AI 交易算法综述](review-2026-07-23/ai-algo-survey.md)（代理 C）
- [目标架构设计](architecture.md)

## 总判断

**当前所有回测/寻优数字都不可信**，三类根因叠加：
1. 前视偏差：信号用当根收盘、回到当根开盘进场（A1）；重采样边界泄漏 5 分钟未来（A2/P0-4）
2. 假交易与崩溃 bug：std 模式无信号也记一笔交易（P0-1）；`--trigger -0.02` 实为"每小时无条件开仓"（P1-6）
3. 乐观执行假设：止损无跳空穿价、零/半吊子成本模型、马丁仓位风险被无量纲化（P0-3、P1-8/9、B2）

加上全样本网格选优无验证集（B1、P1-19），报告收益是"噪声上界的上界"。

## Phase 0：止血（半天）

- [ ] P0-1 假交易 bug（`run_sim.py:50-56`，几行）
- [ ] P0-2 groupby 长度崩溃（`run_sim_intraday_drop.py:33-42`，一行）
- [ ] P1-6 参数校验：`trigger >= 0`、`0 < sl < tp`（防止负 trigger 静默变义）
- [ ] A4 陈旧数据防护：layered_backtest 的 skipped 计数 + 超阈值 raise（当前在跑空数据）
- [ ] 清理根目录旧产物（zigzag_*.csv 为 2025-11 旧数据）

## Phase 1：让回测数字可信（1–2 周）

- [ ] A3/D1 统一数据加载：`pd.to_datetime(utc=True)` 单一入口，7 处收敛为 1 处
- [ ] A1 信号前视：`prev_r = ret_prev.iloc[i-1]`；vol 阈值 `shift(1)`
- [ ] A2/P0-4/P0-5 重采样重写：closed=left、按交易日分桶、残桶过滤（单一 bar_builder）
- [ ] A5 事件去重：`(peak_t, trough_t)` 唯一键；分布统计只在单参数组内做
- [ ] B2/P1-8/9 悲观成交模型：跳空穿价按开盘成交、出入场对称滑点、统一手续费；成本敏感性（0/2/5/10bp）
- [ ] P0-3 马丁美元化：逐层记录股数与成本，worst-case（满层+SL）损失断言
- [ ] 验收标准：修复后重跑全部网格——**预期收益大幅缩水甚至转负，这是诚实基线，不是退步**

## Phase 2：方法论（1 周）

- [ ] B1/P1-19 walk-forward：40d 选参 / 20d 验证滚动；报告只认验证集
- [ ] auto_opt 重建：去掉 target_return 提前停止；综合目标（收益/回撤 + min_trades）；参数邻域扰动测试
- [ ] 数据深度：yfinance 60 天不够支撑 27 路网格——接入 Polygon/Alpaca 历史分钟数据（2+ 年）

## Phase 3：架构重构（2–3 周，见 architecture.md）

- [ ] core/types + data 层（Feed 协议、bar_builder）
- [ ] Strategy 纯函数化迁移 + RiskManager + 持仓状态机
- [ ] SimBroker（Phase 1 的成交模型迁入）+ persistence + Runner
- [ ] 一致性验收：旧 run_sim 与新 backtest Runner 交易流对齐

## Phase 4：AI 增强（与 Phase 3 并行可选，见 ai-algo-survey.md）

- [ ] GBDT 信号过滤器（LightGBM，purged walk-forward），挂 `SignalIntent.confidence`
- [ ] Regime 开关（波动率分档起步，HMM 备选）
- [ ] 滚动重估参数替代静态网格
- 明确不做：多代理 LLM 框架、端到端 RL、零样本时序大模型

## Phase 5：上线路径

1. shadow 模式影子对账（实时信号 vs 回测重放，diff 到零）
2. Alpaca paper trading ≥ 2–4 周（验证断线恢复、状态机、对账）
3. 小仓位实盘（paper→live 只换 broker key）；实盘滑点 vs SimBroker 假设周度复盘
