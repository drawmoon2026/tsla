# TSLA 量化研究与情报系统

单标的（TSLA）量化研究项目：**策略实验室 + 情报采集 + 因果探测器 + 仪表盘 + 前向验证**，全链路本机运行。

- **策略实验室（E 系列）**：E1–E13 共 ~3400 组合，达标线先于结果写死（docs/strategy-lab.md）。当前唯一冻结候选 **E8-A + S2**（跨标的 GBDT V 反弹门控 + 熊市停用开关），参数已冻结、裁决权在 shadow 前向白跑 ≥8 周。
- **哨兵（Sentinel）情报采集**：13 渠道（EDGAR/FINRA/期权链/Polymarket/FOMC/财报日历/USPTO/YouTube/新闻 RSS/X 等）24h 采集入 SQLite，**双时间戳**（发布时刻/首见时刻）在 schema 层防前视。
- **因果探测器（N 系列）**：N1–N6 历史研究收敛出的两腿信号（空头利益跳升 × Musk 发帖密集 → risk-off F20），已冻结做成状态机前向值班（标定期至 ~2026-08-20）。
- **三页面**：值班仪表盘 / 模拟探测（历史+前向逐日推演，每日自动续演）/ 棋谱预案（分支决策树）。
- **shadow 前向验证**：E2 与 E8-A+S2 两条策略线每交易日实时白跑（NullBroker 只记账不下单）。

## 快速开始

### 环境

```bash
make install          # uv 创建 .venv（Python 3.12）并装依赖
# 或手动：uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
```

所有命令一律用 `.venv/bin/python`。`.env`（gitignore）需含 Alpaca 数据/券商 key：
`ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL`（当前已配置，换机需重配）。

### 常驻守护进程（launchd，4 个 label / 6 个自动组件）

| label | 做什么 | 节奏 | 日志 |
|---|---|---|---|
| `com.tsla.sentinel` | 哨兵 13 渠道采集 + 末尾跑因果探测器状态机 | 盘中 5min / 盘外 30min | `outputs/sentinel/launchd.log` |
| `com.tsla.dashboard` | 仪表盘 → 棋谱 → 推演每日续演（三模块串跑） | 每 5min | `outputs/sentinel/dashboard.log` |
| `com.tsla.shadow` | E2 候选 shadow 白跑（`--strategy e2`） | 每交易日 21:00 UTC 一个会话 | `outputs/shadow_live/` |
| `com.tsla.shadow-e8a` | 冻结候选 E8-A+S2 shadow 白跑（`--strategy e8a`） | 同上 | `outputs/shadow_e8a/` |

四个 plist 均已 load 在 `~/Library/LaunchAgents/`（模板在 `intel/deploy/`、`trading/deploy/`）。启停：

```bash
launchctl list | grep tsla                                        # 看状态
launchctl unload ~/Library/LaunchAgents/com.tsla.sentinel.plist   # 停
launchctl load   ~/Library/LaunchAgents/com.tsla.sentinel.plist   # 启
# 改模板后生效 = cp 到 LaunchAgents 再 unload + load
```

电脑必须开机才会跑；漏跑的会话不补。

### 三个页面（本地 HTML，浏览器直接开）

```bash
open data/intel/dashboard.html            # 值班仪表盘：今日合议/晨间简报/态势/走势/渠道/情报流
open data/intel/dashboard.html'#replay'   # 模拟探测：探测器历史推演 + E8-A+S2 全系统逐笔回放（每日续演）
open data/intel/playbook.html             # 棋谱预案：涨/跌分支树，系统规则×个人参数×历史频率
```

页面每 5 分钟由 `com.tsla.dashboard` 重生成，自带数据龄徽章（不装新鲜）。

### 常用 CLI

```bash
.venv/bin/python -m intel.run_sentinel --status       # 哨兵健康：各渠道事件量/时延/最近轮询
.venv/bin/python -m intel.run_sentinel --once         # 手动采一轮（--force 忽略节流）
.venv/bin/python -m intel.detector_report             # 探测器值班报告（--no-price 离线）
.venv/bin/python -m trading.tools.shadow_report --db outputs/shadow_live/journal.sqlite   # E2 周报
.venv/bin/python -m trading.tools.shadow_report --db outputs/shadow_e8a/journal.sqlite    # E8-A+S2 周报
.venv/bin/python -m intel.replay_refresh              # 推演每日续演（幂等；--offline 跳过取数）
.venv/bin/python -m intel.playbook                    # 重生成棋谱页
.venv/bin/python -m intel.dashboard                   # 重生成仪表盘页
```

## 目录结构

| 目录 | 职责 |
|---|---|
| `src/common/` | 全项目公共底座：统一 UTC 数据加载/重采样（data_io）+ 悲观成交结算（execution），所有回测必须走这里 |
| `src/` | 早期 V 事件统计与回测脚本（v_stats/v_plots/hourly_*），Alpaca 数据接入（alpaca_data.py） |
| `research/` | 策略实验室：E 系列（e8_pooled_gbdt/e11_bear_switch/e13_salary_ladder…）与 N 系列（n1–n6）实验脚本，一实验一文件 |
| `intel/` | 哨兵情报层：collectors/ 采集器、store 双时间戳库、detector 因果探测器、dashboard/playbook 页面生成、replay_refresh 续演、splits 拆股口径 |
| `trading/` | 统一交易架构：backtest/shadow/paper/live 同一 Runner，策略纯函数化，SQLite journal；入口 `python -m trading.run` |
| `models/` | 冻结模型工件：e8a/（model.joblib + meta.json 冻结参数 + holdout_ref.csv 漂移防线） |
| `data/` | 价格数据（5m/1h/日线）与 `data/intel/`（sentinel.sqlite、历史情报 CSV、position.json、页面产物） |
| `outputs/` | 各实验/回测/shadow/推演产物（`replay_current/` 为每日续演数据集） |
| `docs/` | 全部文档（见下） |
| `live_trading/` | 旧脚手架（已被 trading/ 取代，仅存档） |

## 关键文档

- **docs/strategy-lab.md** — 实验日志（E/N 全系列假设→判决，多重比较计数器 ~3400）；**一切结论以此为准**
- **docs/roadmap.md** — 路线图与进度勾选（Phase 0–5）
- **docs/architecture.md** — trading/ 统一架构设计（含接口定义）
- **docs/intel-framework.md** — 情报体系框架（T0–T3 源分级、四维评分、防前视原则）
- **intel/README.md** — 哨兵实操手册（渠道清单/时延实测/拆股口径/加渠道方法）
- **docs/review-2026-07-23/** — 全面体检三报告（src 审查/live_trading 审查/AI 算法调研）
- **docs/report.html** — 给人读的总报告（浏览器打开）

## 诚实状态声明（2026-07-31）

- **没有任何策略上过真钱**，也没有达到上钱标准。
- E1–E13 中 **10 个方向判死**（含用户直觉系的 E4/E5/E13：高胜率是会计口径不是 edge）；E2 降级存疑；唯一冻结候选 **E8-A + S2**——留出段 80.6% 胜率/+16bp（n=62，Bonferroni 后不显著），崩盘压测靠 S2 开关才过线（避损靠空仓、非选时）。**shadow ≥8 周（约至 2026-09 下旬）前不做任何裁决。**
- 情报线：机读高位者信号**无独立可交易 alpha**（N1）；幸存的是风险过滤信号（N2/N3-H，拆股修正后复核仍过 Bonferroni，但仅 2 段样本、窄谱）；探测器**标定期至 ~2026-08-20**，期满前只观察不出信号。E12（情报特征进 GBDT）未过门槛，特征标 INACTIVE。
- 近期日历：**08-20** 探测器出闸 → 周度看 shadow_report + detector_report → **10-22** TSLA 财报。

## 用户待办

1. `data/intel/position.json`：`account_value_usd` 仍是示例值 100000，改成真实权益后棋谱页金额/股数才有意义；`trim_line_pct`/`max_pain_pct` 待定（默认不设，有 E5/E10 证据旁注）。
2. USPTO 渠道降级中：申请免费 key 后写入 `.env` 的 `PATENTSVIEW_API_KEY`。
3. 未来 TSLA/池标的拆股：人工补 `intel/splits.py` 的 `SPLITS` 表（采集告警 + SPLIT_GUARD 双层兜底，但口径修正靠补表）。
4. `outputs/hunnei_caichan_xieyi.pdf` 为私人文件，建议移出仓库目录。
