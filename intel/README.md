# intel/ — 哨兵（Sentinel）情报采集层

两套并存的管线：

1. **哨兵 v0（本节，2026-07-24 起）**：多渠道前向流式采集 → SQLite 双时间戳库
   `data/intel/sentinel.sqlite`。目标是从今天起积累"干净"的前向情报流。
2. **历史批量管线（附节）**：`intel/edgar.py / fomc.py / musk_tweets.py` → CSV，
   2018 起的历史回填，供事件研究；与哨兵互不干扰。

## 架构

```
intel/store.py                SQLite 存储层（建表/去重入库/poll_log/时延视图）
intel/collectors/base.py      采集器基类：fetch → normalize → dedupe → store
                              统一限速 HTTP（UA 带联系邮箱，全局 ~6.7 req/s < SEC 红线 10）
intel/collectors/<渠道>.py    每渠道一个模块（见下表）
intel/run_sentinel.py         总入口：跑一轮所有渠道 + 慢渠道节流 + launchd 节奏门控
intel/deploy/                 launchd plist 模板（未加载）+ 部署说明
```

数据库三表一视图：

- `sources` 渠道注册表：tier(T1-T3)/method/poll_interval_s/cost/weight_source(身位分初值)/notes
- `events` 事件表，**双时间戳**是全库核心：
  - `event_time_utc`：信息发生/发布时刻（源头声称）
  - `observed_time_utc`：哨兵首次看到时刻（入库时写死，不随重复轮询变）
  - 特征只能在 `observed_time_utc` 之后可用——防前视在 schema 层写死
  - `event_id = sha256(source_id|dedupe_key)[:16]`，INSERT OR IGNORE 去重
  - 索引 (source_id, event_time_utc)
- `poll_log` 每次轮询一行（成败/抓到数/新增数/耗时/错误）——渠道健康监控
- `v_latency` 视图：每渠道 observed-event 秒级分布（min/avg/max；p50 由
  `store.latency_stats()` 补算）

## 渠道清单与实测（2026-07-24 首采）

| source_id | 层级 | 通道 | 状态 | 首采入库 | 时延（首轮实测，见口径注） |
|---|---|---|---|---|---|
| edgar | T1 | data.sec.gov submissions API（acceptanceDateTime） | ✅ | 6（90 天内 Form4/8-K，含 7/22 财报 8-K） | 稳态≈轮询间隔 5min；首轮 min 32h 是回填口径 |
| edgar_t0 | T1 | 同上 submissions API：SC 13D/G(/A) + Form 144 | ✅ | 5（90 天内） | 稳态≈5min；144 电子申报 2023-04 起才强制 |
| finra_short | T1 | api.finra.org consolidatedShortInterest（免 key POST） | ✅ | 205（2017-12 起全历史一次拿全） | 双周口径；event_time 为近似发布时刻（结算+9 交易日 16:00ET，payload 记账） |
| finra_ats | T1 | api.finra.org weeklySummary 聚合行（ATS+OTC 做市） | ✅ | 30（前向窗 30 周） | 周口径，Tier1 发布滞后 2-3 周；event_time=initialPublishedDate（精确到日）；off_exchange_pct 分母 yfinance best-effort |
| options_snapshot | T1 | yfinance 全期权链日快照 → options_chain 表 + 日度摘要事件 | ✅ | 1 摘要 + 5207 合约行 | 历史不可得，2026-07-24 起攒；**Yahoo OI 字段有抽风期（整链 0，oi_quality 旗标）** |
| polymarket | T3 | gamma-api.polymarket.com public-search（tesla/tsla/musk） | ✅ | 120 市场/日 | 快照型：event_time=抓取时刻，每市场每 ET 日一条赔率 |
| fed_fomc | T2 | fomccalendars.htm 日历页解析 | ✅ | 55 场会议（2022→2027） | 日历预告类，lag 为负=会议在未来，正常 |
| uspto | T2 | PatentsView Search API | ⚠️ 降级 | 0 | 需免费 API key，且 search.patentsview.org 本网络连接超时；备选 USPTO ODP api.uspto.gov（401=通但要 key） |
| youtube | T3 | 频道 RSS ×5（Tesla 官方/CNBC/CNBC TV/Bloomberg TV/Yahoo Fin） | ✅ | 18 | min 8.5h（RSS 只含最近 15 条视频，回填口径）；稳态≈15min 轮询 |
| news_rss | T3 | Yahoo-TSLA / CNBC×2 / MarketWatch / GoogleNews RSS | ✅ | 123 | min ≈50min（回填口径）；稳态≈5min 轮询 + 源发布延迟 |
| x_nitter | T3 | nitter.net RSS（elonmusk + Tesla） | ✅ 脆弱 | 40 | min ≈45min（回填）；稳态≈5min 轮询；实例随时可能死 |
| x_api_paid | T3 | X API v2（付费备选） | 未启用 | — | Basic ~$200/月（读 ~1.5 万帖/月）/ Pro ~$5000/月，以 developer.x.com 现价为准 |

> **时延口径注**：首轮的 lag 分布被"回填"支配（事件发布在几小时/几天前，今天才开始观察），
> 不代表稳态时延。稳态时延 = 轮询间隔 + 源侧发布延迟，要跑几天后只看**增量事件**的
> lag 才是真值。`v_latency` 会随积累自动收敛到真值附近（老事件占比下降）。

### X/Twitter 免费通道调研结论（2026-07-24 实测）

| 通道 | 结果 |
|---|---|
| nitter.net RSS | ✅ 可用，返回最近 20 帖、分钟级新鲜度（已做成 x_nitter 采集器）；但连续请求会 429，且 Nitter 实例历史上反复死亡——**按天塌方预期管理** |
| syndication.twitter.com timeline-profile | ✗ 返回空壳 HTML，无时间线数据（需登录 token） |
| nitter.poast.org / lightbrd.com / twiiit.com | ✗ 403 反爬墙 |
| cdn.syndication.twimg.com | ✗ 空响应 |

判定：免费通道短期用 nitter.net，poll_log 连续失败即为死亡信号，届时二选一：
换存活 Nitter 实例（status.d420.de 有实例清单），或启用 x_api_paid（已在 sources 登记价格）。
不做浏览器伪装硬爬 x.com（违反其 ToS 且极易封）。

## 运行方式

```bash
# 单渠道跑一轮
.venv/bin/python -m intel.collectors.edgar --once
.venv/bin/python -m intel.collectors.edgar_t0 --once      # --backfill 出历史 CSV
.venv/bin/python -m intel.collectors.finra_short --once   # --backfill 出历史 CSV
.venv/bin/python -m intel.collectors.finra_ats --once     # --backfill 出历史 CSV
.venv/bin/python -m intel.collectors.options_snapshot --once
.venv/bin/python -m intel.collectors.polymarket --once
.venv/bin/python -m intel.collectors.fed --once
.venv/bin/python -m intel.collectors.uspto --once      # 需 PATENTSVIEW_API_KEY
.venv/bin/python -m intel.collectors.youtube --once
.venv/bin/python -m intel.collectors.news_rss --once
.venv/bin/python -m intel.collectors.x_nitter --once

# 全渠道一轮（慢渠道自动节流；--force 忽略节流；末尾自动跑因果探测器状态机）
.venv/bin/python -m intel.run_sentinel --once

# 因果探测器：单跑一轮 / 值班报告
.venv/bin/python -m intel.detector --once
.venv/bin/python -m intel.detector_report          # --no-price 离线判分

# 库内统计（渠道/事件量/时延分布/最近轮询）
.venv/bin/python -m intel.run_sentinel --status

# 常驻调度（launchd，模板未加载，见 intel/deploy/README.md）
cp intel/deploy/com.tsla.sentinel.plist ~/Library/LaunchAgents/ && \
  launchctl load ~/Library/LaunchAgents/com.tsla.sentinel.plist
```

调度节奏：launchd 每 5 分钟触发 `--once --auto`；盘中（美东 09:30-16:00）每次真跑，
盘外只在整点/半点后 5 分钟窗口内跑（≈30 分钟一轮）。与 `com.tsla.shadow` label 独立。

## 因果探测器（N3 前向值班，2026-07-24 上线）

`intel/detector.py` 把 N3-H 冻结规则做成哨兵常驻值班组件（`run_sentinel` 每轮在
全部采集器之后运行，`COLLECTORS` 末位注册，source_id=`detector`）：

- **冻结规则（不许改）**：Musk 密集发帖日（act=次交易日）× 回看 20 交易日内有
  空头利益 up-jump 发布（events 表 finra_short 的 change_pct >= +10%）→ RISK_OFF，
  持续 F20（触发日起 20 交易日，重叠触发顺延）。窄谱过滤器定位见
  docs/strategy-lab.md N3-H 条目。
- **标定模式（预先声明的校准问题）**：前向放风腿是 x_nitter RSS 口径（post+RT、
  无 reply），≠ 历史 Sprinklr 归档（含 reply），历史密集参考值 65 帖/日不可直接用。
  实现 = 分位数映射：65 在 musk_tweets.csv 全档日计数分布（2018-01→2025-05，
  2683 天，0 补齐）的经验分位数 **0.8789** 写死；前 20 个交易日只累积 nitter 口径
  日计数基线（状态输出 `CALIBRATING`，不出信号），期满后阈值 = nitter 基线
  （扩张窗，持续累积）的同分位数，每日重算。这是标定实现，不是改规则。
- **落库**：每交易日状态写 `detector_state` 表（状态/两腿读数/阈值/基线进度/
  risk_off_until；盘中多次轮询覆盖更新当日行）；状态切换时向 events 表发一条
  source=detector 事件（title 含状态与两腿数值，仪表盘情报流自动显示）；
  RISK_OFF 触发/解除各记一条假想单进 `detector_trades`（yfinance TSLA 价格快照，
  取价失败 NULL 如实记），周报判分用（判分成本线 -6bp，与 N3-H 日记同口径）。
- **值班报告**：`intel/detector_report.py` —— 当前状态、两腿最新读数、标定进度、
  历史状态切换、假想单判分（REDUCE→RESTORE 配对；未平仓用现价浮动判分）。
- **口径边界（如实声明）**：nitter RT 时间为原帖时间；nitter 宕机期间日计数低估
  会污染基线（poll_log 可查）；交易日用 numpy busday（周一至五）近似、不剔美股
  假日，F20 到期日可能偏移 ~1 日。

## 拆股折算口径（FINRA 空头利益，2026-07-24 起全系统正式口径）

FINRA consolidatedShortInterest 的 short_interest 是**未复权股数**：拆股跨期的
change_pct 会出现假跳变（实测 TSLA 2020-08-31 期 5:1 拆股 +345.8%（真实 -10.8%）、
2022-08-31 期 3:1 拆股 +202.2%（真实 +0.7%），审计见 `outputs/n6_split_audit/`）。
N5（跨标的验证）/ N6（紧急复核）的修正逻辑已沉淀为共享模块 **`intel/splits.py`**，
是唯一口径来源：

- **拆股表 `SPLITS`**（硬编码，出处 research/n5_cross_pit.py N5，与 SI×ADV 同步
  跳变侦测交叉核对 2026-07-24）：TSLA 2020-08-31×5、2022-08-25×3；AAPL 2020-08-31×4；
  NVDA 2021-07-20×4、2024-06-10×10；AMZN 2022-06-06×20；GOOGL 2022-07-18×20；
  WMT 2024-02-26×3；AVGO 2024-07-15×10；NFLX 2025-11-17×10。**未来拆股须人工补表**。
- **change_pct 折算**（`adjust_change_pct`，采集器入库/落盘前调用）：只重算
  跨拆股行（本期与上期折算因子不同），其余行保留 FINRA 原值；原值留
  `change_pct_raw`、标记 `split_adjusted=true`。`finra_short` 与 `finra_short_pool`
  两个采集器均已接入（治本）；存量 `data/intel/finra_short.csv`、
  `data/intel/pool_short/*.csv`（8 文件 9 行）与 sentinel.sqlite 的 finra_short
  事件（2 行，poll_log 有 UPDATE 记录）已按同口径批量修正（2026-07-24，
  原件备份 `*.precorrection`）。
- **水平折算**（`adjust_levels`，N5 口径）：short_interest 除以累计因子统一到
  拆前股本基准，供跨期水平比较（si_chg_6wk_pct 等）；**不落盘**（CSV/库保留
  FINRA 公布真实股数），由下游（n4_golden_pit / n5_cross_pit）读取时调用。
- **跳变侦测告警**（`unexplained_jumps`）：折算后 change_pct >= `SPLIT_GUARD_PCT`
  （+50%，与 detector.py 拆股防护**同源引用**此常量；N6 标定：修正后历史真实双周
  变动最大 +34%、最小拆股因子 2 产生 ~+100%，+50% 居中）且拆股表无法解释 →
  疑似未登记拆股：`finra_short` 采集器发 `si_split_alert` 事件入库（不静默），
  `finra_short_pool` 打印 `[SPLIT-ALERT]`。确认为真实跳变则留档不动（历史上
  COIN 2022-05、QCOM 2019-04、INTC 2020-08 等为真实空头波动）；确认为拆股则
  补 `SPLITS` 表后重新折算。
- **META symbolCode 清洗**（finra_short_pool）：FINRA symbolCode 有复用污染——
  META 代码 2021-07→2022-01 被 Roundhill Ball Metaverse ETF 占用、FB 代码
  2025-06 起被 ProShares ETF 复用。META 系列由 FB+META 双 symbolCode 按
  issueName 白名单（Facebook/Meta Platforms 前缀）过滤拼接，剔除 ETF 行。

## 备份与恢复（2026-08-01 起，com.tsla.backup 每日 08:30）

`intel/backup.py` 每日把**不可再生前向数据**打包成
`backups/sentinel-YYYYMMDD.tar.gz`（gitignore，保留最近 30 份滚动删除）：
sentinel.sqlite、position.json、shadow_live/shadow_e8a 的 journal.sqlite
（sqlite 一律 backup API 热备份，绝不 cp 运行中的库）、outputs/n8_scoring/、
outputs/replay_current/meta.json。打包前逐库 PRAGMA integrity_check，
打包后校验 tar 可读且成员数一致。

```bash
# 手动备份一次 / 恢复演练（解包临时目录 + 逐库 integrity_check）
.venv/bin/python -m intel.backup
.venv/bin/python -m intel.backup --check backups/sentinel-20260801.tar.gz

# 恢复：tar 内是项目相对路径，先停相关 launchd 任务再解包覆盖
launchctl unload ~/Library/LaunchAgents/com.tsla.{sentinel,dashboard,shadow,shadow-e8a}.plist
tar -xzf backups/sentinel-YYYYMMDD.tar.gz -C /Users/tom/project/tsla   # 全量覆盖恢复
# 或只恢复单个库：
tar -xzf backups/sentinel-YYYYMMDD.tar.gz -C /tmp data/intel/sentinel.sqlite
cp /tmp/data/intel/sentinel.sqlite data/intel/sentinel.sqlite
launchctl load ~/Library/LaunchAgents/com.tsla.{sentinel,dashboard,shadow,shadow-e8a}.plist
```

## 局域网服务（手机看盘，com.tsla.serve 常驻）

`intel/serve.py`：绑定 0.0.0.0:8765 的只读静态服务，`/` 重定向 dashboard.html，
手机同一 WiFi 访问 `http://<本机IP>:8765`（IP 见仪表盘 footer，动态生成）。
安全边界：仅局域网（无公网映射）、只读 GET/HEAD、只服务 data/intel/ 下
**.html/.json 白名单后缀**——sentinel.sqlite、*.csv、.env、隐藏文件与一切
目录穿越（含 URL 编码）一律 403（2026-08-01 实测通过）。

## 加一个新渠道的方法

1. `intel/collectors/` 下新建模块，继承 `base.Collector`：
   - `SOURCE`：填 source_id/name/tier/method/poll_interval_s/cost/weight_source/notes
   - `fetch()`：发请求（用 `base.http_get`，自带 UA/限速/重试）返回原始数据
   - `normalize(raw)`：返回事件 dict 列表，必填 `dedupe_key`（渠道内稳定唯一键）、
     `event_time_utc`（**信息公开时刻**，ISO8601 UTC，绝不用内部发生时刻）、`type`；
     可选 `symbol/title/url/payload`
   - 文件尾 `if __name__ == "__main__": cli(YourCollector)`
2. 在 `run_sentinel.COLLECTORS` 列表加一行。
3. 跑 `--once` 验证，看 `--status` 里 poll_log 与时延。
去重、入库、poll_log、sources 注册全部由基类完成，单渠道通常 <100 行。

## 维护事项：统一交易日历年度续期（intel/market_calendar.py）

全项目交易日口径的单一来源（2026-08-02 接线，替换零散 busday 近似——探测器
F20/回看窗、判分器定稿时刻、仪表盘 ETA/等待板/晨间简报、shadow 会话收盘与
EOD 强平、replay_refresh 当日截止）。**假日表与半日市表硬编码 2026–2027**
（出处 NYSE 官方 https://www.nyse.com/markets/hours-calendars ）：

- **每年 NYSE 公布次年日历后，人工把新一年补进 `HOLIDAYS` / `HALF_DAYS`**
  （通常提前 2-3 年公布；建议每年 12 月顺手续下一年）。
- 越界年份自动 fallback 到 busday 近似（周一至周五）并在日志打一次警告——
  fallback 状态下假日/半日市重新失准（回到 P2-1/P1-8 的老毛病），不要长期依赖。
- 续期后跑 `.venv/bin/python -m intel.market_calendar --selftest` 与
  `.venv/bin/python -m intel.dashboard --selftest` 验证。

## 已知边界

- edgar 采集器只入库申报元数据（form/items/acceptance 时刻/URL）；Form 4 逐笔
  交易解析在历史管线 `intel/edgar.py`（CSV）里，哨兵侧需要时再移植。
- fed_fomc 是"日历预告"渠道：event_time 是会议末日 14:00 ET（决议常规发布时刻），
  可以在未来；决议**内容**发布流水见历史管线 `intel/fomc.py`。
- youtube/x_nitter 的 RSS 只含最近 15/20 条，渠道断采超过窗口长度会漏帖。
- news_rss 单 feed 挂掉不拖垮渠道（打印 dead feeds 继续）；google_news 是二手
  聚合，event_time 为源文章发布时刻、非收录时刻。

---

# 附：历史批量管线（CSV，2026-07-24 前建）

统一 schema（data/intel/*.csv）：`event_time_utc, source, type, payload`。
event_time_utc 一律取公开披露时刻——与哨兵同一防前视原则。

- **edgar_form4.csv / edgar_8k.csv**（`python -m intel.edgar`）：2018 起全量，
  acceptanceDateTime 口径；Form 4 按申报×交易代码聚合，472 张 → 721 事件行，
  insider_buy 仅 11 笔；8-K 136 张。
- **fomc.csv**（`python -m intel.fomc`）：联储 ne-press.json 新闻流筛 FOMC statement，
  70 条（2018 起），含 2020-03 两次非常规时刻。
- **musk_tweets.csv**（`python -m intel.musk_tweets`）：HuggingFace fdaudens/musk-tweets
  归档，72,743 条，覆盖 2018-01→2025-05-08，此后无免费全量归档——**2025-05 之后的
  Musk 帖子由哨兵 x_nitter 渠道前向接力**（中间有 ~14 个月缺口，记死）。
- **edgar_13dg.csv / edgar_144.csv**（`python -m intel.collectors.edgar_t0 --backfill`，
  2026-07-24 N2 期新增）：13D/G 43 行（2018 起，TSLA 无 13D 全为 13G）；144 75 行
  （2023-04 起——电子申报强制起点，更早纸质件不进 EDGAR，制度性缺口）。
- **finra_short.csv**（`python -m intel.collectors.finra_short --backfill`）：205 行
  双周空头利益，2018-01→今；event_time 为近似发布时刻（结算+9 交易日）。
- **finra_ats.csv**（`python -m intel.collectors.finra_ats --backfill`）：239 周
  ATS/OTC 场外量与占比，**仅 2021-12-27 起（FINRA API 保留期限制）**。

复采：`.venv/bin/python -m intel.edgar && .venv/bin/python -m intel.fomc && .venv/bin/python -m intel.musk_tweets`
