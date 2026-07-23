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

## Phase 0：止血（半天）— ✅ 完成 2026-07-23

- [x] P0-1 假交易 bug（`run_sim.py`）
- [x] P0-2 groupby 长度崩溃（`run_sim_intraday_drop.py`）
- [x] P1-6 参数校验：`Config.validate()`，负 trigger 直接 ValueError
- [x] A4 陈旧数据防护：layered_backtest skipped 计数 + >5% raise
- [x] 清理根目录旧产物

## Phase 1：让回测数字可信（1–2 周）— ✅ 完成 2026-07-23

- [x] A3/D1 统一数据加载：`src/common/data_io.py`，7 处收敛为 1 处
- [x] A1 信号前视：全部改 shift(1) + 同交易日检查
- [x] A2/P0-4/P0-5 重采样重写：`resample_bars`（closed=left、ET 交易日分桶、残桶过滤、不跨日）
- [x] A5 事件去重：`(peak_t, trough_t)` 唯一键，`v_events_*_unique.csv` + `trough_confirm_t` 列
- [x] B2/P1-8/9 悲观成交模型：`src/common/execution.py`（跳空按开盘、严格穿越、SL 优先、fee 1bp + slip 2bp 默认）
- [x] P0-3 马丁美元化：grid_base_dollars、逐层股数成本、最坏损失估计（满层 $10k、最坏 -$206 = 2.1%）
- [x] A6 swing_generic ZigZag 重写（极值/锚点分离 + 滚动 ATR 阈值）；A9/D5 data_fetch 缓存 max-age、main.py 统一约定
- [x] 验收（诚实基线）：hourly 网格最佳 **288% → 5.56%**；上界 96.8% → 9.96%；run_sim **11.9% → 5.91%**；grid **6.74% → 0.22%**；layered **正 → -6.18%**；1h 事件日计数 27 → 1

## Phase 2：方法论（1 周）— ✅ 完成 2026-07-23（数据深度除外）

- [x] B1 walk-forward：`hourly_signal_backtest.py --walkforward`（40d/20d 滚动，60 天数据仅 1 折：训练 +3.84% → 样本外 +1.66%）
- [x] P1-19 auto_opt 重建：训练/验证按日切分（70/30）、验证集 score=ret/|dd| 选优、min_trades 门槛、±10% 邻域扰动测试、去掉达标即停。首跑即判定当前最优参数"邻域塌方=噪声"
- [ ] 数据深度：⛔ 阻塞——需要 Polygon 或 Alpaca 的 API key（用户提供后接入 2+ 年分钟数据）

## Phase 3：架构重构 — ✅ 骨架完成 2026-07-23

- [x] `trading/` 包：core 类型 + Clock + HistoricalFeed + bar_builder + Strategy 纯函数化（v_reversal 迁移）+ RiskManager + 持仓状态机 + SimBroker/NullBroker + SQLite journal + Runner + CLI（`python -m trading.run`）
- [x] 一致性验收：3 组参数 × 108 笔交易与修复后 run_sim 逐笔一致（max |ret diff| < 1e-15）；回归脚本 `trading/tools/compare_trades.py`
- [x] shadow 模式冒烟通过（信号入 journal、零成交）
- [ ] AlpacaLiveFeed / AlpacaBroker 实装（接口壳已就位；paper 阶段做）
- [ ] 崩溃恢复流程接入 Runner（load_open_state 已有，缺 broker reconcile）
- 已知限制记录在 Phase 3 代理汇报：半日市未建模、执行桶完整性因果等价假设、无部分成交

## Phase 4：AI 增强 — ✅ 管线完成 2026-07-23（结论待深数据）

- [x] GBDT 信号过滤器管线：`research/ml_filter.py`（三重障碍标签、防泄漏特征、purged walk-forward + 1 日 embargo）
- [x] 首跑结论（60 天/243 事件）：OOF AUC 0.485 = 无信号；管线可用，结论以多年数据重跑为准
- [ ] **新发现待修**：v_stats 事件表只收录反弹成功的触发（幸存者偏差）——需把失败触发也落表，GBDT 重训前必修
- [ ] Regime 开关、滚动重估：待深数据接入后做（60 天不够分档）
- 明确不做：多代理 LLM 框架、端到端 RL、零样本时序大模型

## 数据深度（原 Phase 2 遗留）— ✅ 完成 2026-07-23

- [x] 接入模块 `src/alpaca_data.py`；key 已入 `.env`；2 年 SIP 数据落地 `data/TSLA_5m_alpaca.csv`（501 交易日 / 39,078 bar，与 yfinance 重叠段逐 bar 一致）
- [x] walk-forward 重跑（24 折）：164 笔样本外 +1.31%，各折正负交替 → **1H 动量跟随扣费后无 edge**
- [x] auto_opt 重跑：首个通过全部检验的候选 trigger=1.73%/tp=2.85%/sl=1.48%（验证集 7 个月 +4.34%、dd -0.53%、邻域稳定）
- [x] v_stats 幸存者偏差修复：timeout 触发落表（outcome 列），旧正例率虚高 4-5pp 得到确认
- [x] GBDT 重训（2604 样本、8 折）：**OOF AUC 0.602，8 折全部 >0.55**，precision 随阈值单调升（0.510→0.610）——首次出现真实信号迹象（仍属弱信号，未扣成本，单一 regime）
- [ ] 下一步：按 GBDT 概率分层的成本后回测（比继续调模型更有价值）；更多年份数据扩展 regime 覆盖

## Phase 5：上线路径

1. shadow 模式影子对账（实时信号 vs 回测重放，diff 到零）——原型已可用，待 LiveFeed
2. Alpaca paper trading ≥ 2–4 周（验证断线恢复、状态机、对账）
3. 小仓位实盘（paper→live 只换 broker key）；实盘滑点 vs SimBroker 假设周度复盘
