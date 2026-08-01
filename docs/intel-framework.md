# 情报分析 + 股市数据 预测体系（草案 v0.1）

> 2026-07-24 立项。核心命题（用户）：**情报信息是高位者放出的信号或事实**。
> 体系目标：把"高位者信号"变成可量化、可回测、可前向验证的特征，与股市数据合流，
> 喂给已验证的决策管线（GBDT/规则 + walk-forward + shadow），预测走势。
> 红线：只用**公开、合法**的信息渠道。高位者的非公开信息 = 内幕交易，不碰；
> 高位者**依法必须公开披露**的行为轨迹 = 本体系的富矿。

## 一、"高位者信号"的可机读来源清单（按信号强度排序）

### T1 — 真金白银的行为（最强信号：他们做了什么）
| 来源 | 内容 | 机读通道 | 时延 | TSLA 相关性 |
|---|---|---|---|---|
| SEC Form 4 | 高管/董事/10%股东的买卖（含 Musk） | EDGAR API（免费） | 交易后 2 个工作日 | ★★★ |
| SEC 13F | 机构季度持仓变化 | EDGAR API | 季度+45 天（钝） | ★★ |
| 国会议员交易披露 | 议员及配偶的股票交易 | Capitol Trades / QuiverQuant API | ≤45 天 | ★★ |
| SEC 8-K | 重大事件强制披露（并购/高管变动/合同） | EDGAR 实时流 | 事件后 4 个工作日内 | ★★★ |
| 期权异动 | 大单/未平仓量突变（知情资金的足迹） | CBOE / Polygon options | 当日 | ★★★ |
| 空头利益 | 做空仓位变化 | FINRA（双周） | 半月（钝） | ★★ |

### T2 — 有约束力的话（他们承诺了什么）
| 来源 | 内容 | 机读通道 | TSLA 相关性 |
|---|---|---|---|
| 财报电话会 | 管理层措辞、指引修正、回避的问题 | 转录文本（API 多家） | ★★★ |
| FOMC/美联储 | 利率路径、点阵图、主席措辞变化 | Fed 官网日历+文本 | ★★（贴现率敏感股） |
| 监管机构 | NHTSA 召回/调查、FSD 审批进展 | 官网公示 | ★★★ |

### T3 — 放风与叙事（他们想让你相信什么——需最强怀疑折扣）
| 来源 | 内容 | 备注 |
|---|---|---|
| Musk 的 X 帖子 | TSLA 特有的最大单一叙事源，有实证价格冲击 | 需区分"事实披露"vs"情绪表达"；只能前向使用（LLM 记得历史推文） |
| 大行评级/目标价 | 常滞后于价格而非领先 | 当反向拥挤度指标可能更有用 |
| 财经媒体头条密度 | 恐慌/狂热的温度计 | 情绪 proxy，非事实 |

## 二、衡量体系：每条情报打四个分

任何一条情报进入系统前，先过四维评分（0-1），乘积为情报权重：

1. **身位分**（谁说的）：真金白银行为 1.0 > 有约束力承诺 0.7 > 放风叙事 0.3
2. **事实分**（可证伪性）：已发生的事实 1.0 > 可验证的承诺 0.6 > 观点/预期 0.2
3. **时效分**：信息公开时刻到现在的衰减（半衰期按来源类型标定）
4. **意外分**：与市场共识的偏离度（共识内的信息无价值；用发布前后期权隐含波动/分析师预期偏差度量）

> 这套评分本身就是特征工程：四个分 + 来源类别 = 每条情报变成一个特征向量，
> 与既有价格特征（rv20、回撤深度、V 事件属性）拼接，进 GBDT。

**v1 实现状态（2026-08-01，`intel/scoring.py`）**：
身位/事实/时效三分为纯规则——身位分按渠道 tier 映射（T0/T1=1.0、T2=0.7、
衍生信号=0.5、T3=0.3，表 `POSITION_BY_TIER`）；事实分按事件 type 规则映射
（法定披露/数据发布=1.0、日历预告=0.6、新闻报道/发帖=0.2，表 `FACT_BY_TYPE`
+ 前缀规则）；时效分为半衰期指数衰减 exp(−ln2·age/半衰期)，半衰期按渠道类型
标定（数据类 14 天、新闻类 2 天、日历类不衰减，注册表 `HALF_LIFE_DAYS` +
`SOURCE_CLASS`）。**意外分 v1 已实现：期权 IV 基线**——事件 ET 日的 TSLA
ATM IV（`options_chain` 快照表，DTE 15-60 中最近 30 天到期、现价 ±2% 带内
call+put 合并均值）相对**前 5 个有效快照日均值**的相对变化，线性归一化到
0-1（相对变化 ≥30% 记满分，`SURPRISE_REL_FULL`）。选用依据（2026-08-01
实测记录）：Yahoo IV 字段多数快照日质量合格（TSLA ATM IV 44-49%，量级合理），
但存在与 OI suspect_zero 同源的抽风日（07-24、07-27 整链 IV 近零）——按质量
门槛剔除（有效 IV ∈ [10%, 300%] 且 ATM 有效报价 ≥3 条，坏日按缺数据处理），
故未启用备选口径（近 20 日已实现波动率变化率，留作 IV 渠道死亡时的替代）。
快照 2026-07-24 起前向积累，历史事件不可回补：事件日无有效覆盖 → 意外分
诚实返回 None，总权重按三项乘积（partial 标注）；四项齐时总权重 = 四乘积。
意外分为日级口径（同 ET 日事件共享），是 v1 已知粗糙点。评分在查询层实时
计算，不回写 events 表；仪表盘情报流每行显示总权重（悬停看四项分解），
渠道卡权重列双显"人工先验 + 当前平均四维分"，脚注展示当前覆盖率。
自测 `python -m intel.scoring`（单调性断言 + 意外分覆盖/缺席行为断言）。

## 三、与决策管线的接合（复用已验证资产，不另起炉灶）

```
情报采集层（本框架 T1-T3 源）
   │  防前视原则：每条情报带"公开时间戳"，特征只在时间戳之后可用
   ▼      （借用 AI-Trader v1 的模拟时钟工具层设计——待缺口分析后移植）
LLM 信息处理层（只做提炼，不做决策）
   │  职责：非结构化文本 → 结构化特征（四维评分、事件分类、情绪极性）
   │  历史段标注不可信（模型记得历史），只能前向积累 —— 与 shadow 机器同节奏
   ▼
特征合流：情报特征 + 市场数据特征
   ▼
决策层：GBDT 排序 / 规则触发（已验证的管线，E8 跨标的训练框架）
   ▼
验证层：walk-forward → 崩盘压测 → shadow 前向（现成，launchd 已在跑）
```

**关键推论**：情报特征无法回测（LLM 标注历史段被污染），所以这个体系的
验证只能靠**前向积累**——从现在开始每天采集、标注、入库，8-12 周后才有第一批
干净样本。shadow 机器已经在跑，情报层与它同步积累是自然节奏。

## 四、AI-Trader v1 覆盖度对比（2026-07-24 解剖完成，读代码结论）

> 解剖对象：`/Users/tom/project/ai-trader/v1/`。5 个 MCP 服务 + 数据管线逐个过了一遍源码。
> 一句话结论：**v1 的信息面 = 本地行情 + AlphaVantage 通用财经新闻（+ 默认关闭的 Jina 网页搜索）。
> 第一节 T1 六项全缺、T2 三项全缺、T3 仅"媒体情绪"有现成字段（且被代码注释掉没喂给 LLM）。
> v1 是 LLM 竞技场不是情报系统——情报层要自建；值得搬的是它的模拟时钟/审计/MCP 封装骨架。**

### 4.1 工具层逐个解剖（agent_tools/，端口见 start_mcp_services.py）

| 服务(端口) | 文件 | 数据源 | 返回什么 | 防前视处理 |
|---|---|---|---|---|
| math (8000) | tool_math.py | 无 | add/multiply 两个算术函数 | 不适用 |
| search (8001，默认) | tool_alphavantage_news.py | AlphaVantage `NEWS_SENTIMENT` API | API 返回 title/url/summary/time_published(分钟级)/source/整体+分 ticker 情绪分/topics；**但格式化后只给 LLM Title+Summary 前 1000 字**，时间/来源/情绪字段全被注释掉（L295-299） | `time_to=TODAY_DATE`、`time_from=前30天`，靠 API 服务端过滤；日线模式 TODAY_DATE 解析为当日 00:00，即只可见前一日为止的新闻（保守，好）；**本地不复验 time_published**（信任服务端） |
| search 备选（默认关闭） | tool_jina_search.py | s.jina.ai 搜索 + r.jina.ai 抓取 | 随机抽 1 条 URL 的 title/description/content 前 1000 字/publish_time | 只在**搜索结果层**按 date 过滤且 fail-open（日期缺失/解析失败则保留）；相对日期"2 days ago"用真实 `datetime.now()` 解析而非模拟时钟；**抓取的是当前实时页面内容，抓取层零过滤**——页面事后更新即泄露未来 |
| trade (8002) | tool_trade.py | 非信息源 | buy/sell，成交价=当日 `1. buy price`(开盘价)，写 position.jsonl（date/id/this_action/positions 增量），fcntl 文件锁；A 股 100 股一手 + T+1 | 只能按开盘价成交，无法"看到收盘再下单" |
| price (8003) | tool_get_price_local.py | 本地 data/merged.jsonl（按后缀路由 A_stock/、crypto/） | OHLCV；`date == TODAY_DATE` 时 high/low/close/volume 返回 "You can not get..." 字符串，只给 buy price | 见 4.3 —— 有一个真漏洞 |
| crypto (8005) | tool_crypto_trade.py / tool_get_price_crypto.py | ccxt 接 bybit/binance 等 fetch_ticker/fetch_ohlcv | 实时报价/K线/24h 涨幅榜 | **完全无模拟时钟**，`datetime.now()`，只能 live-forward 用 |

时延与额度：AlphaVantage 免费 key 25 请求/天（新闻和行情共享额度，代码无限流/重试预算管理）；新闻 time_published 分钟级；行情拉取用 `entitlement=delayed`（15 分钟延迟档）。Jina 免费额度按 token，计量在 Jina 侧。

### 4.2 数据层（data/）

- **美股 merged.jsonl**：AlphaVantage `TIME_SERIES_INTRADAY` 60min（`extended_hours=false`，仅常规时段；另有 `TIME_SERIES_DAILY` 和 yfinance 两套备用脚本）。覆盖 NASDAQ-100 全部 101 标的+QQQ；当前文件 2025-10-01 → 2025-11-10，60min 粒度。`merge_jsonl.py` 把 `1. open`→`1. buy price`、`4. close`→`4. sell price` 改名，并把**文件中最新一个时间戳截断为只剩 buy price**（数据层防前视）。
- **A 股**：日线 Tushare `pro.daily`（上证50 成分，`pro.index_weight` 选股，需 TUSHARE_TOKEN，本机 .env 未配置）；小时级用 efinance；merge 脚本同样改名+截断最新一日。另有 AlphaVantage 版 A 股脚本备用。
- **加密**：AlphaVantage `DIGITAL_CURRENCY_DAILY`（BTC/ETH 等 10 币）生成本地文件 + ccxt 实时。
- **模拟时钟总线**：`TODAY_DATE` 由 base_agent.run_date_range 每个交易日写入 `RUNTIME_ENV_PATH` 指向的 runtime_env.json；所有 MCP 子进程经 `get_config_value("TODAY_DATE")` 读同一文件——跨进程一致的"时钟"，这是整个防前视设计的核心。

### 4.3 防前视机制评价（要移植的关键设计，好坏都写清）

成立的部分（live-forward 模式下）：
1. 三层防线：数据文件层（merge 截断最新日）→ 工具层（`date == TODAY_DATE` 掩码）→ 新闻层（API `time_to`）；
2. "当日只知开盘价、只能按开盘价成交"贯彻到 trade 工具，逻辑自洽；
3. 时钟集中于 runtime_env.json，工具进程无各自为政的时间源。

漏洞（按严重度）：
1. **price 工具没有 `date > TODAY_DATE` 的拒绝逻辑**（tool_get_price_local.py 只判 `==`）：只要 merged.jsonl 里存在未来日期，agent 查询未来日期就返回**完整 OHLCV**。当前仓库正是这个状态——文件覆盖到 2025-11-10 而 default_config 从 2025-10-01 开始回放，即**出厂配置下漏洞是活的**。只有"文件只更新到今天"的 live-forward 运行方式在物理上堵住它。小时级同病：同日未来小时的 bar 可读。移植时必须补 `>` 拒绝。
2. **Jina 搜索基本不设防**（见 4.1）：fail-open + 真实时钟解析相对日期 + 抓取层实时内容零过滤。默认关闭是对的；若启用等于给回测开天窗。不建议按原样移植。
3. AlphaVantage 新闻**只信服务端过滤**，本地不对返回的 time_published 做二次校验（防 API 边界 bug 的最后一道防线缺失）；顺带把时间/情绪字段注释掉，LLM 连新闻发生时间都看不到——防前视没问题，但信息量自废。
4. crypto 实时工具无时钟约束（live 模式设计使然，但与回测工具混在同一工具箱，误用无护栏）。
5. LLM 权重记得历史（回放段对新模型是"记忆内"数据）——框架不处理，与本文档第三节"情报特征只能前向积累"的判断一致。

### 4.4 对照第一节清单逐行标注缺口

| 第一节条目 | v1 覆盖？ | 说明 |
|---|---|---|
| T1 SEC Form 4 | ✗ | 无 EDGAR 通道；AlphaVantage 新闻或有二手报道，非结构化不可依赖 |
| T1 SEC 13F | ✗ | 无 |
| T1 国会议员交易 | ✗ | 无 |
| T1 SEC 8-K | ✗ | 无 EDGAR 流；同 Form 4，仅可能被新闻间接覆盖 |
| T1 期权异动 | ✗ | 无任何期权数据源 |
| T1 空头利益 | ✗ | 无 |
| T2 财报电话会转录 | ✗ | `topics=earnings` 只有新闻报道，非转录文本 |
| T2 FOMC/美联储 | ✗ | `topics=economy_monetary` 只有报道，无官网日历/原文 |
| T2 监管机构（NHTSA 等） | ✗ | 无 |
| T3 Musk X 帖子 | ✗ | 无社交媒体源 |
| T3 大行评级/目标价 | △ | 新闻报道会捎带，无结构化目标价数据 |
| T3 媒体头条密度/情绪 | △ | NEWS_SENTIMENT 自带整体+分 ticker 情绪分——**唯一现成的 T3 特征**，但 v1 代码把它注释掉了没用 |
| 行情数据（第三节合流用） | ✓ | NASDAQ-100 60min/日线 OHLCV，够用但仅常规时段、15 分钟延迟档 |

**回答"是否有缺失、还不足的地方"：有，且是系统性的。** v1 覆盖面只有行情+通用财经新闻；T1（真金白银行为）整层为零，T2（有约束力的话）整层为零，T3 只有一个未启用的情绪分。本框架的情报采集层需按第一节清单全部自建（EDGAR API、QuiverQuant、CBOE/Polygon、Fed 日历、X 抓取），v1 帮不上情报源，只帮得上"骨架"。

### 4.5 可移植清单（进 tsla 项目的候选，按性价比排序）

| 模块 | 源文件 | 移植成本 | 备注 |
|---|---|---|---|
| 模拟时钟工具层 | tools/general_tools.py（get/write_config_value + runtime_env.json）+ base_agent.run_date_range 的 TODAY_DATE 写入 | 低（约半天，核心 <100 行） | **必须补 `date > TODAY_DATE` 拒绝**再用；正是第三节"防前视原则"要借的件 |
| JSONL 决策审计 | base_agent._log_message（log/{date}/log.jsonl 会话流水）+ position.jsonl（date/id/this_action/positions 增量账本） | 低（照抄模式即可） | 与 shadow 机器的前向积累天然契合，情报标注也可用同一账本格式 |
| MCP 工具封装模式 | FastMCP `@mcp.tool` + streamable-http + start_mcp_services.py 进程管理 + MultiServerMCPClient 接入 | 低-中（约 1 天） | 每个情报源做成一个 MCP 工具，天然隔离+可单测；模板直接套 |
| AlphaVantage 新闻工具 | tool_alphavantage_news.py | 低（半天） | 可直接复用作 T3 媒体源；改进：本地复验 time_published、恢复情绪/时间字段输出、加 25 req/天限流记账 |
| 不建议移植 | tool_jina_search.py（防前视形同虚设，需整体重写）；crypto 实时工具（无时钟）；price_tools 的全文件线性扫描（性能差，tsla 单标的用不着） | — | — |

## 五、诚实性预警（先于结果写死）

1. "高位者信号"里最诱人的是 Form 4 内部人买入——学术实证：内部人**买入**有弱预测力
   （年化超额 ~6-9%，衰减中），**卖出**几乎无信息（卖的理由太多）。预期要校准。
2. 国会交易披露的公开研究：跟单议员组合并不稳定跑赢，时延 45 天是硬伤。
3. 期权异动信号在散户工具普及后拥挤化，2020 前的实证不能外推。
4. 本体系的多重比较记账并入 strategy-lab.md 计数器；每个情报源单独立假设检验，
   不允许"全家桶一起上然后挑好的"。

## 六、哨兵 v0 实施状态（2026-07-24 上线）

代码 `intel/`（store.py + collectors/ + run_sentinel.py），库 `data/intel/sentinel.sqlite`
（events 双时间戳：event_time_utc 发布时刻 / observed_time_utc 首见时刻，差值=渠道时延；
sources 渠道注册表；poll_log 健康监控；v_latency 时延视图）。详见 intel/README.md。

| 渠道 | 层级 | 状态 | 首采 | 说明 |
|---|---|---|---|---|
| edgar（Form4/8-K, acceptanceDateTime） | T1 | ✅ | 6 | 稳态时延≈5min 轮询间隔 |
| fed_fomc（日历页） | T2 | ✅ | 55 场 | 预告类，event_time 可在未来 |
| uspto（PatentsView） | T2 | ⚠️ 降级 | 0 | 需免费 key 且端点本网络不通；备选 api.uspto.gov |
| youtube（5 频道 RSS） | T3 | ✅ | 18 | Tesla 官方全收，媒体按关键词 |
| news_rss（Yahoo/CNBC/MarketWatch/GoogleNews） | T3 | ✅ | 123 | 关键词 tesla/tsla/musk |
| x_nitter（nitter.net RSS） | T3 | ✅ 脆弱 | 40 | 免费镜像唯一存活通道；死亡则启用 X API Basic ~$200/月（sources 已登记） |

首轮时延分布是回填口径（跑数天后看增量事件才是稳态真值）。调度：launchd 模板
`intel/deploy/com.tsla.sentinel.plist`（未 load）：盘中 5min / 盘外 30min 一轮。
第三节"前向积累 8-12 周出首批干净样本"的时钟从今天起表。
