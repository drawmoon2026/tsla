# TSLA live_trading 脚手架代码审查报告

> 审查日期：2026-07-23，审查者：Claude Fable 5（代理 B）
> 审查范围：`live_trading/` 下 8 个 Python 文件（共约 740 行）+ `run.txt`。

总体判断：这是一个**回测/模拟脚手架，距离"半实盘"还有本质差距**——没有任何真实的订单状态机、状态持久化、异常处理和日志体系；且存在 2 处会直接产生错误交易记录的正确性 bug、1 处必然崩溃的 bug。以下按 资金安全/正确性 > 执行真实性 > 风控 > 健壮性 > 结构 排序。

---

## 一、资金安全 / 交易逻辑正确性（最高优先级）

### [P0-1] trigger_std 模式下 direction=0 不会跳过，产生大量虚假 SL 交易
- **位置**：`live_trading/run_sim.py:50-56`
- **问题**：`continue` 只写在 `else` 分支里：
  ```python
  if cfg.trigger_std:
      thr = cfg.trigger_std * prev["ret_std"]
      direction = 0 if pd.isna(thr) or abs(prev["ret_prev"]) < thr else ...
  else:
      direction = compute_trigger(prev, cfg.trigger)
      if direction == 0:
          continue          # 只有非 std 模式才跳过！
  ```
  当 `trigger_std` 启用且 direction=0 时，代码继续往下走：`tp_px = sl_px = entry_price`（因为 `1 + tp*0 = 1`），随后进入 `else`（做空）分支判定，`low <= entry` 且 `high >= entry` 几乎必然同时成立，被记为一笔 "sl" 交易，`ret = 0 - 2*fee_bp`。
- **影响**：std 模式下每个合规小时都产生一笔零收益、纯扣手续费的假交易；交易笔数、胜率、SL 率、总收益全部失真。若此逻辑照搬到实盘接单层，等于无信号也下单。
- **建议**：把 `if direction == 0: continue` 移出 `else`，两种模式共用。

### [P0-2] `groupby` 键与过滤后 DataFrame 长度不匹配，遇到盘前盘后数据必然崩溃
- **位置**：`live_trading/run_sim_intraday_drop.py:33-42`
- **问题**：`et = df.index.tz_convert(cfg.tz)` 在 RTH 过滤**之前**计算，随后 `df = df[rth]` 缩短了 df，但 `df.groupby(et.date)` 仍用旧长度的 `et`。当前只因 yfinance 5m CSV 恰好只含 RTH 数据（掩码全 True）才侥幸不崩。换任何含盘前盘后的数据源即 `ValueError: Grouper and axis must be same length`。
- **建议**：过滤后重算 `et = df.index.tz_convert(cfg.tz)`，或直接 `df.groupby(df.index.tz_convert(tz).date)`。

### [P0-3] 马丁格尔仓位与权益复利模型完全脱节，风险被系统性低估
- **位置**：`live_trading/run_sim_grid.py:57-82, 161-166`；`config.py:23-24`
- **问题**：`grid_mults = (1, 2, 4, 8)` 是经典马丁加仓（最深层 8 倍），但：
  1. `simulate_grid` 返回的 `ret` 是**相对均价的百分比收益**，外层 `equity *= (1 + ret)` 把它当作 100% 权益满仓的收益来复利——无论实际只成交了第 1 层（1 个单位）还是全部 15 个单位，对权益的影响相同。仓位规模信息在收益归因时被完全丢弃。
  2. 杠杆检查 `prospective_notional = (cash + mults[li] * price) / capital`（第 67 行）中 `mults` 是无量纲乘数、`cash` 是"乘数×价格"，量纲混乱：TSLA 价格 ~300 时 15 个单位名义 ~4500，除以 capital=10000 恒 < 1，`max_leverage` 检查形同虚设。
  3. `simulate_grid(window, direction, cfg, capital, equity)` 的 `equity` 参数从未被使用（第 37 行），杠杆检查永远基于初始 capital 而非当前权益。
- **影响**：回测结果对真实马丁风险严重失真——马丁策略的核心风险恰恰是"满层加仓后 2% 止损 = 15 单位 × 止损幅度的实际美元亏损"，当前模型无法反映；据此上实盘会低估爆仓概率。
- **建议**：以美元为单位定义每层投入（如 `grid_dollars`），逐层记录持仓数量与成本，PnL 用美元计算后再更新权益；杠杆检查用当前 equity；同时补一个"全层成交 + 打止损"的最坏单笔亏损上限断言（应 ≤ 权益的某个比例）。

### [P0-4] 1H K 线合成的 closed/label 约定与 5m 时间戳语义错位（bar_builder 边界问题）
- **位置**：`live_trading/bar_builder.py:12`；配合 `run_sim.py:65-67`
- **问题**：5m CSV 的 `Datetime` 是 **bar 开始时间**（yfinance 惯例），而 `resample("1h", closed="right", label="right")` 按时间戳分箱：标记为 14:30 结束的 1H bar 实际包含 13:35~14:30 开始的 5m bars，即真实覆盖 13:35–14:35，整体右移 5 分钟；每天 9:30 那根 5m bar 被归入上一个（隔夜空）箱。虽然凑巧不构成前视（信号可得时刻 ≈ 执行窗第一根 5m open 时刻），但：
  1. "对齐 9:30" 的设计意图（offset=30）实际没有实现，小时边界全部错 5 分钟；
  2. 若未来换成 bar 结束时间戳的数据源，同一段代码立即变成**用当根未收盘信息发信号**的前视偏差，且无任何断言防护。
- **建议**：明确数据时间戳语义并写入断言；对 bar-start 时间戳应使用 `closed="left", label="right"`，同时把执行窗口取法从 `df.index > window_start` 改为 `>=`（否则会漏掉紧邻信号后的第一根 5m bar）。

### [P0-5] 部分小时 bar（30 分钟 / 5 分钟数据）被当作完整 1H bar 发信号
- **位置**：`live_trading/bar_builder.py:12-14`；`run_sim.py:40, 48-61`
- **问题**：offset=30min 使日内箱为 9:30/10:30/…/15:30；RTH 数据下，15:30–16:30 箱只有 30 分钟数据，早间 8:30–9:30 箱只含 1 根 5m bar。更严重的是 `Close_ret = pct_change()` 跨日计算：每天第一根"1H bar"的收益 = **隔夜跳空 + 5 分钟**，而 `allowed_hours` 含 9，隔夜跳空会直接触发开盘信号。这与"1 小时动量"策略假设完全不符。
- **影响**：信号分布被隔夜跳空和残缺 bar 污染；实盘中开盘时段本就是滑点最大、最难成交的时段。
- **建议**：按 bar 内 5m 根数过滤（如 `count >= 10` 才有效）；`Close_ret` 按交易日分组计算或显式剔除每日首根；隔夜跳空作为单独信号显式建模而非混入。

### [P1-6] 参数无校验，`run.txt` 中 `--trigger -0.02` 实际含义是"每小时都交易"
- **位置**：`live_trading/signal.py:12`；`run.txt:1-2`
- **问题**：`abs(ret) < trigger` 在 trigger 为负时恒 False，等价于阈值 0——历史运行命令用的负 trigger 让策略每个合规小时无条件按上小时方向开仓。用户可能以为负号代表"做反向"，实则是静默改变语义。同理 `tp=0.1`（1 小时内 10% 止盈）基本不可能触发，实际退化为"持有到窗口结束"。
- **建议**：`Config.__post_init__` 中断言 `trigger >= 0`、`0 < sl < tp` 等；若需要"反转"或"总是交易"语义，用显式参数表达。

### [P1-7] 作为"实盘脚手架"完全没有持仓状态机
- **位置**：`live_trading/run_sim.py` 全文；`config.py:8`（provider 字段无任何消费方）
- **问题**：模拟里每笔交易被强制限制在一个 1H 窗口内平仓，所以回测不会重复开仓/留仓；但这依赖"逐窗口顺序回放"这一模拟特性。代码里没有 Position/Order 对象、没有"有仓时忽略新信号 or 反手"的规则、没有收盘强平逻辑，`provider='alpaca_paper'` 等配置没有任何对接代码。直接照此上 paper/实盘，连续两个小时都触发信号时的行为是未定义的。
- **建议**：抽出显式状态机（FLAT → PENDING_ENTRY → HOLDING → EXITING），定义信号冲突规则（忽略/反手）、日终强平、以及 broker 回报驱动的状态迁移。

---

## 二、执行真实性

### [P1-8] 止损成交价假设过于乐观：无跳空穿价处理
- **位置**：`live_trading/run_sim.py:89-92`；`execution.py:37-42`；`run_sim_grid.py:88-97`；`run_sim_intraday_drop.py:66-69`
- **问题**：四处止损全部假设精确成交在 `sl_px`。若某根 5m bar **开盘即低于止损价**（新闻、盘中熔断、隔夜跳空后的首根 bar），真实止损单成交在开盘价或更差。当前写法把跳空损失截断在 sl 处，系统性高估策略收益——这正是 TSLA 这类高波动股票最致命的场景。
- **建议**：多头止损用 `exit_px = min(bar["Open"], sl_px)`（空头对称），止盈限价单用 `exit_px = max(bar["Open"], tp_px)`（多头，反而更优）；并在止损出场价上叠加滑点。

### [P1-9] 止盈按"触及即成交"，且出场无滑点、grid 无手续费
- **位置**：`run_sim.py:94-100`（slip 只加在入场，第 72 行）；`run_sim_grid.py:47, 70, 88-131`（`fee_bp` 全程未使用，出场无 slip）；`execution.py:17-48`（fee 完全缺失）
- **问题**：`high >= tp_px` 即按 tp_px 全额成交，忽略限价单排队/仅触碰不成交的情形；出场（尤其 SL 市价单、grid 一次性平 15 个单位）零滑点；grid 模拟连手续费都没扣。多因素叠加，回测收益整体偏乐观。
- **建议**：止盈要求 `high > tp_px`（严格穿越）或加成交概率折扣；出入场对称计滑点；grid 平仓按仓位规模放大滑点；统一在一处结算 fee。

### [P1-10] grid 同一根 bar 内"先补仓、后判止损"的顺序假设不可知
- **位置**：`live_trading/run_sim_grid.py:61-107`
- **问题**：一根 5m bar 内先执行所有触及的网格补单，再用**补仓后的新均价**判定 TP/SL。真实路径可能是价格先击穿止损再回落触发网格价（或反之），bar 数据无法区分。当前顺序对马丁策略是偏乐观方向（更深补仓拉低均价 → 更难触发 SL）。
- **建议**：至少提供悲观模式（先按补仓前均价判 SL）做区间估计；或改用 1m 数据回放降低 bar 内歧义。

### [P2-11] 零延迟假设与 `--mode` 参数未实现
- **位置**：`run_sim.py:71-72`（信号可得瞬间即按下一根 open 成交，无下单延迟）；`run_sim.py:145-146` 定义了 `--mode path5m/bar_hl` 但函数体从未读取 `args.mode`；`execution.py` 的 `simulate_bar` 被 import（`run_sim.py:25`）却从未调用——整个文件是死代码。
- **建议**：删除或实现 `--mode`；给入场价加 1 根 bar 延迟或固定秒级延迟做敏感性分析。

---

## 三、风控

### [P1-12] run_sim 主策略没有任何账户级风控
- **位置**：`live_trading/run_sim.py:44-115`
- **问题**：每笔交易 100% 权益满仓复利（第 102 行），无单笔仓位上限、无日亏损上限、无最大回撤熔断（`global_stop` 只在 grid 版实现）、无连亏降仓。sl=1% 满仓即单笔 -1%（跳空时更多），连续止损无任何减速机制。
- **建议**：引入按波动率的仓位规模（如目标单笔风险 = 权益的 0.5%，仓位 = 风险预算 / sl 距离）；把 `global_stop` 回撤熔断提升到所有 runner 共用。

### [P2-13] grid 的全局止损实现有缺陷且只是"停止回测"
- **位置**：`live_trading/run_sim_grid.py:179-182`
- **问题**：回撤检查在每笔**完成后**才做——单笔损失本身不受限（满层马丁 + 2% 均价止损的美元损失见 P0-3）；`break` 只是终止模拟循环，不是可复用的实盘风控组件；且 `max(t["eq_after"] ...)` 每笔 O(n) 重算，应维护 running peak。
- **建议**：把回撤熔断做成前置检查（开新仓前判断），并作为独立 RiskManager 供实盘复用。

### [P2-14] 无隔夜风险场景，但也无停牌/涨跌停/流动性场景
- **位置**：各 runner 均在窗口/日内强平，隔夜风险为零（这点是对的）；但没有对"5m 数据中断期间持仓"的处理——实盘 feed 断了、仓还在，当前架构（无状态、无恢复）对此毫无办法。见 P1-16。

---

## 四、健壮性与工程质量

### [P1-15] 时区假设写死：裸时间戳一律按 UTC 处理
- **位置**：`run_sim.py:31-32`、`run_sim_grid.py:32-33`、`run_sim_intraday_drop.py:24-25`（三处复制粘贴的 `load_feed`）
- **问题**：CSV 若是交易所本地时间（常见导出格式），`tz_localize("UTC")` 后 `allowed_hours`、RTH 过滤全部错位 4~5 小时，且**静默错**——不会报错，只会得到一份看似合理实则时段全错的回测。
- **建议**：加数据健康检查：转换到 ET 后断言绝大多数 bar 落在 9:30–16:00；`load_feed` 去重合并为一个模块。

### [P1-16] 无状态持久化、无异常处理、无日志、无数据缺口检测
- **位置**：全部文件
- **问题**：
  - 唯一的输出手段是 `print` 和结果 CSV；没有 `logging`，没有交易/订单级审计日志。
  - 全程零 `try/except`：CSV 缺列、空文件、NaN 价格直接裸崩。
  - 无 bar 缺口检测：5m 序列中间缺 1 小时，窗口只剩 1-2 根 bar 也照常"交易"（`run_sim.py:67-69` 只判 empty）。
  - 无任何状态落盘：崩溃/重启后持仓、已成交网格层、当日交易计数全部丢失——对声称的"实时部署脚手架"（`run_sim.py:6`）这是硬缺口。
- **建议**：接入 `logging`（含 trade-level JSON 审计日志）；窗口 bar 数下限校验；实盘态用 SQLite/JSON 持久化持仓与订单状态，启动时 reconcile broker 实际持仓。

### [P2-17] config 硬编码与类型问题
- **位置**：`live_trading/config.py`
- **问题**：`trigger_std: float = None` 类型注解不实（应为 `float | None`），且 `run_sim.py:50` 用真值判断——`trigger_std=0.0` 会被当作"未启用"；`load_config()` 只返回默认值，没有文件/环境加载能力，所有策略参数改动需改源码（CLI 覆盖只有 run_sim 支持，grid 和 intraday_drop 的参数无法从命令行调整）；API 密钥直接进 dataclass 默认值（import 时即读环境）。
- **建议**：`Optional[float]` + `is not None` 判断；支持 YAML/JSON 配置文件；三个 runner 统一 CLI 覆盖机制。

### [P3-18] 小的代码质量问题
- `run_sim_grid.py:64`：`hit = (l <= price <= h) if direction == 1 else (l <= price <= h)` 三元两支完全相同——要么是笔误（本想区分多空），要么应删掉。
- `run_sim_grid.py:170`：`exit_time` 用 `bars_held - 1` 索引窗口，但 `bars_held` 相对 `first_fill_idx` 而非窗口起点，首次成交非第 0 根时出场时间记录错误。
- `execution.py:37-42`：`if sl_hit and tp_hit` 与 `elif sl_hit` 两分支体完全相同，可合并。
- `run_sim.py:15,25`：`numpy` 与 `simulate_bar` import 未使用。
- `run_sim.py:40-41` 的 `ret_prev` 与 `bar_builder` 的 `Close_ret` 重复计算。
- `run_sim_intraday_drop.py:47`：日内高点用 `Close` 而非 `High` 追踪，回撤触发口径偏松且与真实盘中观察不一致。

---

## 五、auto_opt.py 的寻优方法

### [P1-19] 全样本寻优 + "达标即停"，是教科书式的过拟合流程
- **位置**：`live_trading/auto_opt.py:32-56`
- **问题**：
  1. **无训练/验证切分**：在同一份 60 天 5m 数据上随机采样参数、按同一数据的总收益选优——选出的参数只是对这 60 天噪声的最优拟合。
  2. **`target_return` 提前终止**（第 52 行）：语义是"抽签抽到 10% 就停"，等价于对随机结果做择优发表，进一步放大多重比较偏差。
  3. **评价指标单一**（第 44 行）：只看 `total_return`，不看交易笔数（3 笔碰运气 20% 会胜过 100 笔稳定 8%）、最大回撤、收益分布；也不惩罚参数敏感性。
  4. 底层 `run_sim` 的 P0-1/P0-4/P0-5 缺陷会被寻优器**放大**——优化器最擅长找到利用模拟器漏洞（乐观止损、部分 bar 信号）的参数。
- **影响**：`best_summary.txt` 里的参数上实盘几乎必然表现远差于回测，这是本项目除马丁风险外最大的"资金陷阱"。
- **建议**：
  - 时间序列切分：前 70% 寻优、后 30% 验证，报告只认验证集成绩；数据量允许时做 walk-forward。
  - 目标函数改为综合分（如 验证集收益/最大回撤，附加 `min_trades` 门槛）。
  - 去掉 `target_return` 提前终止，或仅作为"验证集达标"条件。
  - 对最优参数做邻域扰动测试（±10% 参数收益不应塌方）。
  - 附带：每次迭代写一个 `run_{i}` 目录（第 41 行）产生上百个目录垃圾，寻优期传 `no_report` 语义并只落 search_log。

---

## 优先修复顺序建议

1. **P0-1**（假交易 bug）、**P0-2**（崩溃 bug）——几行即可修复；
2. **P0-3**（马丁美元化仓位模型）+ **P1-8/9**（悲观成交假设）——否则一切回测数字不可信；
3. **P0-4/P0-5**（K 线边界与部分 bar）——决定信号本身是否成立；
4. **P1-19**（寻优加验证集）——在上面修完之前，auto_opt 的输出不应作为任何实盘依据；
5. **P1-7/P1-16**（状态机 + 持久化 + 日志）——上 paper trading 前的必要工程建设。

一句话结论：当前代码可以作为研究原型，但在修复 P0 级问题并重建执行假设（跳空止损、滑点、马丁仓位美元化）之前，任何回测/寻优结果都不具备指导真金白银的效力；尤其 `grid_mults=(1,2,4,8)` 的马丁结构在现有模拟器下的风险被显著低估。
