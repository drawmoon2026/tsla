# 哨兵 launchd 部署（模板，未加载）

节奏实现：launchd 每 5 分钟触发一次 `run_sentinel --once --auto`；
`--auto` 在盘中（美东工作日 09:30-16:00）每次都真跑，盘外只有分钟落在
[0,5) 或 [30,35) 时才跑（≈每 30 分钟一轮），其余触发直接退出（零开销）。
慢渠道（fed/uspto，日级 poll_interval_s）由 run_sentinel 自身节流。

启用：

```bash
cp intel/deploy/com.tsla.sentinel.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tsla.sentinel.plist
```

停用：

```bash
launchctl unload ~/Library/LaunchAgents/com.tsla.sentinel.plist
```

健康检查：

```bash
.venv/bin/python -m intel.run_sentinel --status   # 各渠道最近一次轮询与事件量
tail -f outputs/sentinel/launchd.log
```

与既有 `com.tsla.shadow`（每交易日 21:00 UTC 一次的 shadow 交易）label 独立、
日志目录独立（outputs/sentinel/），互不影响。

其余模板（2026-08-01 起已安装并 load）：

- `com.tsla.backup.plist`：每日 08:30 本地时间跑 `intel.backup`（不可再生数据
  备份到 backups/，热备+校验+30 份滚动，见 intel/README.md「备份与恢复」）。
- `com.tsla.serve.plist`：常驻 KeepAlive 跑 `intel.serve`（0.0.0.0:8765 局域网
  只读静态服务，仅 data/intel/ 的 html/json 白名单）。
