"""哨兵（Sentinel）情报采集层 — N 系列实验的数据源模块.

两套管线（详见 intel/README.md）：
1. 哨兵 v0 前向流：store.py + collectors/ + run_sentinel.py →
   data/intel/sentinel.sqlite（双时间戳 event_time_utc / observed_time_utc）
2. 历史批量（本包根下 edgar/fomc/musk_tweets 模块）→ CSV：

统一输出 schema（data/intel/*.csv）：
    event_time_utc  事件的**公开披露时刻**（UTC, ISO8601）——防前视的关键字段，
                    一律用披露/接收时刻，绝不用交易发生时刻
    source          数据源标识（edgar_form4 / edgar_8k / fomc / musk_tweets）
    type            源内事件分类（如 insider_buy / 8-K item / fomc_decision）
    payload         JSON 字符串，源特有字段全部塞这里
"""
