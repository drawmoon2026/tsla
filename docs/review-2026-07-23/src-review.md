# TSLA V 形反转项目代码审查报告（src/）

> 审查日期：2026-07-23，审查者：Claude Fable 5（代理 A）
> 以下路径均相对项目根目录。按 正确性 > 方法论 > 性能 > 结构 排序。

---

## 一、正确性 / 统计有效性（最高优先级）

### A1.【严重前视偏差】信号使用当前 bar 收盘价，却在当前 bar 开盘进场
- **位置**：`src/hourly_signal_backtest.py:54-61`（`simulate`）、`src/hourly_signal_backtest.py:126-135`（`simulate_dyn`）、`src/hourly_signal_backtest_1m.py:77-89`（`simulate_intrabar`）
- **问题**：`prev_r = ret_prev.iloc[i]` 是 bar i 自身的 close-to-close 收益（依赖 bar i 的收盘价），但入场价却是 bar i 的 `Open`（1m 版是 bar i 窗口第一分钟的 Open）。docstring 声称 "previous 1H return → enter next bar open"，实现却是"用本 bar 收盘确认信号、回到本 bar 开盘进场"。
- **影响**：三个执行类回测的收益全部无效——信号 bar 本身往往就是大动量 bar，等于免费吃到整根 bar 的方向性收益。所有 grid_search 结果、report.txt、"best params" 均不可信。
- **建议**：改为 `prev_r = ret_prev.iloc[i-1]`，即用 bar i-1 的收益，在 bar i 开盘进场；1m 版同理，信号确认时刻必须早于入场时刻。另外 `simulate_dyn` 中 `vol.iloc[i]` 的 rolling 窗口也包含 bar i 自身收益（`src/hourly_signal_backtest.py:127`），阈值应用 `vol.iloc[i-1]`（或 `vol.shift(1)`）。

### A2.【重采样边界错误】closed/label="right" 与 yfinance 时间戳约定不符，1H bar 错位一根且泄漏未来
- **位置**：`src/v_stats.py:78`、`src/hourly_signal_backtest.py:30`、`src/hourly_signal_backtest_1m.py:60`、`src/hourly_mv_backtest.py:35`
- **问题**：yfinance 的 intraday 时间戳是 bar **起始**时刻（9:30 的 5m bar 覆盖 9:30–9:35）。用 `closed="right", label="right"` 时，桶 (8:30, 9:30] 会把 ts=9:30 的开盘 bar 吞进上一个桶。实际后果（已按当前缓存核实）：每天产生一个只含 1 根 5m bar 的 "9:30 小时 bar" 和一个只含 25 分钟的 "16:30 残根 bar"；正常小时 bar 的 Open 丢了前 5 分钟、Close 却是**下一小时**第一根 5m bar 的收盘——收盘价泄漏 5 分钟未来。1m 版 `src/hourly_signal_backtest_1m.py:85` 的 `df1m.index <= bar_end` 同样把下一小时首分钟并入本 bar。
- **影响**：所有 1H close-to-close 收益错位；残根 bar 制造虚假信号（9:30 单根 bar 的"1H 收益"实际是隔夜跳空+5 分钟）；与 A1 叠加进一步夸大收益。
- **建议**：统一改 `closed="left", label="left", offset="30min"`，并在重采样后过滤掉 bar 数不足的残桶（如要求每桶 ≥ 满额 5m bar 数的某个比例）。

### A3.【时区健壮性】读缓存 CSV 时对新版 yfinance（America/New_York 索引）不健壮，跨 DST 必崩
- **位置**：`src/v_stats.py:52-57`、`src/data_fetch.py:68-70`、`src/hourly_signal_backtest.py:22-25`、`src/hourly_mv_backtest.py:26-29`、`src/layered_backtest.py:56-59`、`src/swing_generic.py:25-28`、`src/zigzag_v.py:60-64`
- **问题**：当前缓存 `data/TSLA_5m_60d.csv` 已是 ET 偏移（`2026-04-27T09:30:00-0400`，全窗口恰好都在 EDT，偶然可解析为 fixed-offset）。一旦 60 天窗口跨越 3 月或 11 月的 DST 切换，CSV 中会出现 `-0400`/`-0500` 混合偏移，`pd.read_csv(parse_dates=...)` 会返回 **object dtype** 索引，随后 `df.index.tz` 直接 `AttributeError`；而旧 UTC 缓存又依赖 `tz is None → tz_localize("UTC")` 的分支。两种格式各有一半代码路径不健壮。
- **影响**：管线在特定日期区间随机崩溃，或（更糟）naive 索引被误标为 UTC 导致所有 ET 相关统计（time-of-day 热力图、日计数）偏移 4-5 小时。
- **建议**：所有加载点统一为一个函数：`idx = pd.to_datetime(df.index, utc=True)`（`utc=True` 对混合偏移、单一偏移、naive 都能正确处理，naive 情况需先判断来源约定），再按需 `tz_convert("America/New_York")`。这是应提取的公共模块的首要职责（见 D1）。

### A4.【静默数据不一致】layered_backtest 消费的事件文件与价格缓存不同期，KeyError 被吞
- **位置**：`src/layered_backtest.py:77-82`
- **问题**：`price.loc[peak_t]` 失败时 `except KeyError: continue` 静默跳过。当前仓库状态就是活例子：根目录 `zigzag_events.csv` 是 2025-11 的旧数据，而 `data/TSLA_5m_60d.csv` 是 2026-04~07——**所有事件都会被静默丢弃**，`metrics()` 在空 DataFrame 上要么报错要么输出全 NaN 的"回测结果"。
- **建议**：统计并打印 skipped 数量，skipped 比例超阈值（如 5%）直接 raise；事件文件中写入数据源指纹（时间范围 hash）做一致性校验。

### A5.【事件跨参数重复计数】同一物理 V 事件按 27 套参数重复入库，日计数/分布统计/回测全部重复
- **位置**：`src/v_stats.py:187-194`（`all_events.extend` 跨 x/y/T 合并）、`src/v_stats.py:272`（summary.md Top10 来自合并集）、`src/v_plots.py:66-78`（`compute_daily_counts` 直接 value_counts）、`src/zigzag_v.py:170-180` + `src/zigzag_v.py:204-219`（X/Y 网格嵌套：X=0.01 的事件集是 X=0.02 的超集，daily/weekly/monthly/drop_stats/regime 统计全部叠加）、`src/layered_backtest.py:70`（逐行遍历含重复事件的 `zigzag_events.csv`，同一 V 被交易多次）
- **影响**：`daily_v_count.csv` 的"每日 V 次数"被放大最多 27 倍且放大倍数随参数覆盖度不均匀变化，无统计意义；summary.md Top10 很可能是同一物理事件的多参数副本；layered 回测的收益、换手、资金占用被成倍虚增。
- **建议**：事件表加 `(peak_t, trough_t)` 唯一键；分布类统计（日计数、drop/rebound 分布、Top10、regime）只在**单一参数组**内做或先去重；回测输入必须按单一参数组过滤（layered 应接受 `--swing_th/--x/--y` 并 query 过滤）。

### A6.【算法 bug】swing_generic 的 ZigZag 无法追踪趋势内极值，pivot 输出错误
- **位置**：`src/swing_generic.py:43-67`
- **问题**：循环里 `trend >= 0` 分支中 `if move >= 0: continue` 会跳过第 62-67 行的 extreme 更新代码——而更新条件 `prices[i] > last_pivot_price` 恰恰只在 move>0 时成立，即**更新逻辑不可达**（下行趋势对称同理）。结果 H pivot 记录的是上一次反转点价格而非趋势内最高点。
- **影响**：`swings.csv`/`segments.csv` 的波段划分整体错误，"覆盖历史 95 笔波段"的目标基于错误输出。
- **附带**：`src/swing_generic.py:117` TR 用 `High-Low`，未含隔夜 gap 项（真 TR = max(H-L, |H-prevC|, |L-prevC|)），ATR 系统性偏低；`src/swing_generic.py:122` `atr_pct` 取全样本均值——既非自适应（是常数阈值），又使用了全样本信息（前视）。
- **建议**：删掉这套实现，复用 `src/zigzag_v.py:68-123` 的正确版本（extreme 与 pivot 分离追踪），阈值改用逐 bar 滚动 ATR。

### A7. v_stats 事件定义的隐性前视与配对缺陷
- **位置**：`src/v_stats.py:89-96`、`src/v_stats.py:127-134`、`src/v_stats.py:171-173`
- **问题**：(1) 局部极值用 `lookforward=3` 根未来 bar 确认——作为纯历史统计可以，但 summary.md 把事件表述成"可捕捉的机会"时，peak/trough 时刻实际不可实时识别（trough 确认最早在 trough_t+3 根 bar，而 rebound 搜索却从 trough_i+1 开始，反弹可能发生在确认之前）；(2) `while j < n - lf` 的 trough 搜索**无时间上限**，peak 可与任意远的 trough 配对，可跨隔夜/周末 gap，且中途出现更高 peak 不会重置锚点，`drop_pct` 不是相对最近峰值的跌幅；(3) 超时后 `i = limit + 1` 会跳过窗口内的其他 peak，漏检事件。
- **建议**：给 peak→trough 加最大 bar 数限制并禁止跨交易日（或单独标记跨日事件）；trough 搜索途中遇到更高 close 时更新 peak 锚点；在事件表加 `trough_confirm_t = trough_t + lf*bar` 列，回测类消费方一律以确认时刻为最早可行动时刻。

### A8. layered_backtest 的成交假设含多重前视
- **位置**：`src/layered_backtest.py:88`（slice1 在 swing peak 价格买入——peak 是事后经 threshold 回撤才确认的点，实时无法在峰值成交）、`src/layered_backtest.py:103`（未 TP 的仓位一律按 rebound swing high **最高点**结清——同样是事后确认点）、`src/layered_backtest.py:104`（`hit_tp` 时 `hold_bars` 仍按 rebound_t 计算，与"提前 TP 离场"不一致）、`src/layered_backtest.py:88-94`（-5%/-10% 补仓时点按峰谷线性插值、精确限价成交，无跳空穿价处理）
- **影响**：入场偏早偏高、出场按最优点位、且事件样本本身是"最终成功反弹才被记录"的条件样本（幸存者偏差），三者叠加使 `expected_return` 是不可实现的上界。加之 A5 的重复事件与重叠事件资金复用无约束（`src/layered_backtest.py:143-150` 逐事件顺序累加 pnl，未检查时间重叠期总敞口 > CAPITAL）。
- **建议**：若定位为"上界估计"需在输出中明示；否则改为逐 bar 重放：入场条件改为"自峰值确认后回撤 x% 挂限价"，出场用真实路径首次触及 TP 判定，并维护跨事件的统一资金账本。

### A9. 杂项正确性问题
- `src/zigzag_v.py:153`：年化系数 `np.sqrt(252 * (78 * 5))` 错误，78 已是每日 5m bar 数，应为 `sqrt(252 * 78)`；当前 vol 数值虚高 √5≈2.24 倍（不影响基于分位数的 regime 分类，但输出的波动率数值错误）。`src/zigzag_v.py:157` 分位数取全样本（前视），滚动分位更严谨。
- `src/hourly_signal_backtest.py:169-171`：`simulate_dyn` 的 `pnl_dollar = capital * ret_trade` 用初始本金而非当笔权益，与 `simulate` 口径不一致；`eq_before = eq / (1 + ret_trade)` 是事后反推，可读性差且有除零隐患（ret_trade=-100% 时）。
- `src/hourly_signal_backtest.py:274`：`if args.trigger and args.tp and args.sl` —— 参数为 0 时被判 falsy 落入 grid 分支，应使用 `is not None`。
- `src/hourly_signal_backtest.py` / `src/hourly_mv_backtest.py`：`pct_change()` 跨隔夜（前日末残根 → 次日 9:30 残根），隔夜跳空被当作可交易的"1H 动量"；应剔除每日首根 bar 的信号或将隔夜与日内分开统计。
- `src/zigzag_v.py:218`：`regime.loc[t]` 在 t 缺失（数据缺口/停牌）时抛 KeyError 使整个 pipeline 崩溃，应改 `regime.reindex(...)`。
- `src/main.py:24` 用 `US/Eastern`、`src/data_fetch.py:60` 用 `America/New_York`、`src/main.py:15` `auto_adjust=False` vs `src/data_fetch.py:39` `auto_adjust=True`——同一项目两套抓取约定，缓存与即时抓取的价格体系（是否复权）不一致。

---

## 二、回测方法论

### B1. 无样本内/样本外划分，网格选优即报告——过拟合风险最高的一处
- **位置**：`src/hourly_signal_backtest.py:288-300`（27 组合选 `total_return` 最大者写入 report）、`src/zigzag_v.py:233-241`（`best_combo_equity` 从 12+ 组合挑最终净值最高者）
- **影响**：60 天单一标的，1H 仅约 470 根 bar、有效交易几十笔，27 路多重比较后取最大值，报告的收益几乎必然是噪声上界；`make_report` 把它写成"实盘可执行信号回测报告"极具误导性。
- **建议**：最少做 walk-forward（如 40d 选参 / 20d 验证滚动）；对 best 组合做 block bootstrap 或 White's Reality Check 给出 p 值；报告中强制标注样本期、交易笔数与"样本内选参"字样。

### B2. 零成本假设
- **位置**：所有 simulate 函数（`src/hourly_signal_backtest.py:48`、`src/hourly_signal_backtest_1m.py:64`、`src/layered_backtest.py:66`）
- **问题**：无佣金、无点差、无滑点、做空 TSLA 无借券费/无融券可用性约束；TP/SL 假设精确触价成交（快速行情中 SL 常穿价）。策略 tp 1–2%、sl 0.5–1%，往返成本 2–5bp 起并不可忽略，高频全仓复利下成本影响被复利放大。
- **建议**：加 `--fee_bps --slippage_bps` 参数，入场按 open+slippage、SL 按 sl_price−slippage 成交；报告成本敏感性（0/2/5/10bp 四档）。

### B3. hourly_mv_backtest 是不可实现的"完美捕捉"上界
- **位置**：`src/hourly_mv_backtest.py:39-69`，尤其 `src/hourly_mv_backtest.py:125` 把结果与 "~141.6% compounded" 目标对比打印
- **问题**：假设事前知道方向、以 close-to-close 全额捕获（含隔夜 gap）。docstring 有说明，但输出口径容易被当成可达收益。
- **建议**：输出中明确标注 "look-ahead upper bound, not tradable"，并与 A1 修复后的可执行版本并列展示差距。

### B4. 1m 重播版样本期与 5m 版不可比
- **位置**：`src/hourly_signal_backtest_1m.py:193`（默认 `period=7d`）
- **问题**：1m 重播只覆盖 7–30 天，与 60 天 5m 版的参数/结论直接对比无效；且每次运行现抓数据不落盘，结果不可复现。
- **建议**：1m 数据同样走 `data_fetch` 缓存路径；对比时对齐时间窗口。

---

## 三、性能

- `src/v_stats.py:83-96` + `src/v_stats.py:109`：`find_local_extrema` 是 Python 双层循环，且在 `detect_v_events` 内被每个 (x,y,T) 组合重复调用——3 tf × 27 组合 = 81 次重算。极值 mask 只依赖 (lb, lf)，应在 `run_grid_for_tf` 外层算一次传入；实现可向量化：`s == s.rolling(lb+lf+1, center=True).max()`。
- `src/v_stats.py:127`：trough 无界搜索使 worst-case O(n²)（大 x 值时每个 peak 扫到序列尾）。加上限后自然消除；进一步可利用网格嵌套性：只按最小 x/y 检出候选事件一次，再向量化过滤出各组合子集，把 27 次扫描降为 1 次。
- `src/hourly_signal_backtest.py:179-197`：`grid_search` 27 次调用 `simulate`，每次重算 `pct_change`，逐 bar Python 循环。信号判定与单 bar 持仓的 TP/SL 结果完全可向量化（布尔 mask + `np.select`），27 组合可共享一次性预计算的 entry/high/low/close 数组；若保留循环版，至少用 `joblib`/`multiprocessing` 并行网格。
- `src/hourly_signal_backtest_1m.py:85` + `src/hourly_signal_backtest_1m.py:98`：每笔交易对全量 1m 表做布尔切片 O(N)，内层 `iterrows` 逐分钟判断。应 `np.searchsorted` 定位窗口，首触判定用 `np.argmax(touch_mask)` 向量化。
- `src/layered_backtest.py:70` / `src/zigzag_v.py:218`：`iterrows` + 逐行 `.loc` 查价，可改为一次 `reindex`/merge。
- `src/hourly_signal_backtest.py:227`：`make_report` 内部重复 `import numpy as np`（模块顶部已导入）。

---

## 四、代码结构

### D1. CSV 加载 + 时区归一化在 7 处重复且各不相同
`src/v_stats.py:51`、`src/hourly_mv_backtest.py:25`、`src/hourly_signal_backtest.py:21`、`src/layered_backtest.py:55`、`src/swing_generic.py:24`、`src/zigzag_v.py:59`、`src/data_fetch.py:65`。这是 A3 时区 bug 存在 7 个副本的根因。建议建 `src/common/data_io.py`：`load_bars(path) -> DataFrame[UTC index]`（内部 `pd.to_datetime(utc=True)`）、`resample_rth(df, rule)`（内置 A2 的正确 closed/label 约定与残桶过滤）。

### D2. 回测器重复
`resample_1h` 三份（`src/hourly_mv_backtest.py:32`、`src/hourly_signal_backtest.py:28`、`src/hourly_signal_backtest_1m.py:58`）；`Trade` dataclass、TP/SL 判定块、`make_report`、max_dd 计算在 `src/hourly_signal_backtest.py` 与 `src/hourly_signal_backtest_1m.py` 近乎逐行重复。TP/SL 判定的 6 行 if/elif 在 4 处出现——一旦修 A1/A2 需要改 4 处，极易漏改。建议提取 `src/common/execution.py`（单 bar 持仓结算函数）与 `src/common/report.py`。

### D3. 两套 ZigZag 实现并存
`src/zigzag_v.py:68`（正确）与 `src/swing_generic.py:31`（有 A6 bug）。保留前者，swing_generic 只保留 ATR 阈值计算并调用统一实现。

### D4. 配置硬编码
`src/layered_backtest.py:24-28`（CAPITAL/SLICES/TP/BAR_MIN 及 -5%/-10% 档位散在 `:87-94` 逻辑里）、`src/hourly_signal_backtest.py:282` 与 `:285`（动态触发分支的 tp=0.025/sl=0.01 写死两处）、`src/hourly_signal_backtest.py:288-290`（网格）、`src/zigzag_v.py:33-35`、`src/v_stats.py:12-18`、`src/layered_backtest.py:166-167`（输入路径写死）。建议集中到一个 `config.py` 或 YAML，并让所有脚本接受 `--config`。

### D5. 输出布局与流水线
- 事件/摘要 CSV 写到仓库根目录（`src/v_stats.py:254`、`src/zigzag_v.py:307-311`、`src/v_plots.py:128`），与 `outputs/`、`figures/` 并存三套约定，且正是 A4 陈旧文件被误用的温床——统一写入 `outputs/<run_id>/`。
- `run_all.py:19-28` 只覆盖 fetch→v_stats→v_plots 三步，回测脚本不在流水线内；subprocess 方式无法传递 period/interval 等参数。
- `src/data_fetch.py:94-101`：缓存键是 `{symbol}_{interval}_{period}`，"60d" 是滚动窗口——缓存一旦生成就永不过期（不加 `--refresh` 时今天跑的还是几个月前的数据，当前仓库正是如此）。建议缓存按实际日期区间命名或加 max-age 检查。
- `src/main.py` 与 `src/data_fetch.py` 职责重叠且约定冲突（见 A9），建议删除 main.py 或改为薄封装。

---

## 修复优先级建议（前 5 项）

1. **A1**（信号前视）+ **A2**（重采样错位）——不修则全部回测结论作废，两者都修在 `common` 模块一次到位。
2. **A3**（时区统一 `utc=True`）——防崩溃、防静默错位，7 处收敛为 1 处。
3. **A5**（事件去重/单参数组统计）——所有"V 事件频率"类结论目前被放大数倍。
4. **A4** + **D5**（数据/事件文件一致性校验 + 输出目录整治）——当前仓库里 layered 回测实际在跑空数据。
5. **B1** + **B2**（walk-forward + 交易成本）——修完正确性后，让"best params"结论具备最低限度的统计效力。
