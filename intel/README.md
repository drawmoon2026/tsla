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

# 全渠道一轮（慢渠道自动节流；--force 忽略节流）
.venv/bin/python -m intel.run_sentinel --once

# 库内统计（渠道/事件量/时延分布/最近轮询）
.venv/bin/python -m intel.run_sentinel --status

# 常驻调度（launchd，模板未加载，见 intel/deploy/README.md）
cp intel/deploy/com.tsla.sentinel.plist ~/Library/LaunchAgents/ && \
  launchctl load ~/Library/LaunchAgents/com.tsla.sentinel.plist
```

调度节奏：launchd 每 5 分钟触发 `--once --auto`；盘中（美东 09:30-16:00）每次真跑，
盘外只在整点/半点后 5 分钟窗口内跑（≈30 分钟一轮）。与 `com.tsla.shadow` label 独立。

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
