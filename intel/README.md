# intel/ — 哨兵情报采集层（N1 起用）

统一 schema（data/intel/*.csv）：`event_time_utc, source, type, payload`。
**event_time_utc 一律取公开披露时刻**（不是事件/交易发生时刻）——防前视的第一原则。
payload 为 JSON 字符串，源特有字段全部在内。

## 各源口径

### edgar_form4.csv — TSLA 内部人交易（SEC Form 4）
- 采集：`python -m intel.edgar`。data.sec.gov submissions API（recent 段 + 2018 前归档段），
  CIK 0001318605，逐张申报拉原始 ownershipDocument XML（primaryDocument 去掉 xslF345X0x/
  前缀直取），限速 ~8 req/s（SEC 上限 10），User-Agent 带联系邮箱。
- 时间戳：EDGAR `acceptanceDateTime`（SEC 接收申报时刻，UTC）= 公众最早可见时刻。
  交易日只进 payload.trade_dates。
- 事件行：一张 Form 4 内同交易代码的多笔合并（股数/金额求和、vwap 加权）。
  type：`insider_buy`(code P 公开市场买入) / `insider_sell`(code S) / `code_X`(其他：
  M 行权、F 税务代扣、A 授予、G 赠与等——**不算真金白银信号**)。
- 已知边界：4/A 修正单独成行（payload.is_amendment）；派生表（期权）未采集；
  股数跨 2020(5:1)/2022(3:1) 拆股不可比，用 value_usd 比较。
- 2026-07-24 采集结果：472 张申报 → 721 事件行；insider_buy 仅 **11 笔**（预警条目
  说中：高管几乎只卖）；41 张跳过（无非派生交易/获取失败）。

### edgar_8k.csv — TSLA 8-K 重大事件
- 同一 submissions API，零额外请求；时间戳同 acceptanceDateTime。
- type = `8k_items_<item列表>`；payload.items 存 item 编号（2.02 业绩、5.02 高管变动、
  1.01 重大协议、7.01 RegFD、8.01 其他…）。136 张（2018-01 起）。

### fomc.csv — FOMC 决议
- 采集：`python -m intel.fomc`。联储官网新闻稿 JSON 流
  https://www.federalreserve.gov/json/ne-press.json ，筛 "Federal Reserve issues FOMC
  statement"，`d` 字段（美东时刻）转 UTC。常规会 14:00 ET，2020-03 两次紧急决议为
  真实非常规时刻（3/3 10:00、3/15 17:00），比硬编码日历准。70 条（2018-01 起）。

### musk_tweets.csv — Musk 发帖流（best-effort）
- 采集：`python -m intel.musk_tweets`。HuggingFace 公开数据集 fdaudens/musk-tweets
  （Sprinklr 导出，原始文件缓存在 _musk_tweets_raw.csv）。
- **覆盖 2018-01 → 2025-05-08，此后无数据**（X API 收费后无免费全量归档）；
  完整性不可证（可能漏帖/含已删推），逐年密度与公开报道量级一致（2018 ~7 帖/日 →
  2024 ~80 帖/日）。72,743 条。
- 时间戳 = 发帖时刻（发帖即公开）。type：musk_post/musk_reply/musk_repost。
  本轮零 LLM 判断，payload 只存原文（截 2000 字符），下游做关键词字符串匹配。

## 复采
```
.venv/bin/python -m intel.edgar && .venv/bin/python -m intel.fomc && .venv/bin/python -m intel.musk_tweets
```
