# TSLA 5 分钟数据下载与统计

使用 `yfinance` 抓取特斯拉（TSLA）近 5 天的 5 分钟级别数据，输出基础统计信息。

## 环境要求
- [uv](https://docs.astral.sh/uv/)（自动安装 Python 3.12 并管理 `.venv`）
- macOS / Linux / WSL 均可

## 快速开始
```bash
make run
```
首次运行会自动创建 `.venv` 并安装依赖，然后执行脚本。

### 一键跑全流程
```bash
python run_all.py        # 如需强制重新下载，加 --refresh
```
流程：下载/读取 5m 数据 → 统计 V 事件 (5m/15m/1h 全参数网格) → 生成图表 → 输出 `outputs/summary.md`。

## 单独安装依赖
```bash
make install
```

## 项目结构
- `src/main.py`：下载数据并打印统计结果。
- `src/data_fetch.py`：抓取并缓存 5m 数据到 `data/TSLA_5m_60d.csv`。
- `src/v_stats.py`：计算 V 事件，产出 `v_events_{tf}.csv`、`v_summary.csv`、`summary.md`。
- `src/v_plots.py`：基于统计结果生成 `figures/*.png` 和 `daily_v_count.csv`。
- `run_all.py`：串联上述步骤。
- `src/hourly_signal_backtest.py`：1H 信号（上一根触发、下一根执行）回测，默认网格搜参。
- `src/hourly_signal_backtest_1m.py`：1m 重播执行，避免同根高低顺序前视。
- `live_trading/`：实时/半实盘脚手架（配置、信号、执行模拟、示例 `run_sim.py`）。
- `requirements.txt`：项目依赖（yfinance, pandas, numpy, matplotlib, pytz）。
- `Makefile`：一键安装与运行。

## 输出文件
- `data/TSLA_5m_60d.csv`：价格数据缓存
- `v_events_{5m,15m,1h}.csv`：各时间尺度的 V 事件明细
- `v_summary.csv`：参数网格汇总
- `summary.md` / `outputs/summary.md`：Markdown 摘要（top 10 等）
- `figures/*.png`：统计图
- `daily_v_count.csv`：按交易日的 V 计数
- `outputs/hourly_signal/`：1H 信号回测结果（网格、最佳交易、报告）
- `outputs/hourly_signal_1m/`：1m 重播结果
- `outputs/live_sim/`：实时脚手架在历史 5m 上的模拟输出

## 调整参数网格
- 默认阈值在 `src/v_stats.py` 的 `DEFAULT_X / DEFAULT_Y / DEFAULT_T`，修改后重新运行 `python run_all.py`。
- 仅测试单组参数可直接运行：`python src/v_stats.py --tf 5m --x 0.01 --y 0.005 --tbars 24`

## 输出内容
- 行数、列名、时间范围（美东时区）
- 关键列的描述统计（均值、标准差、极值等）
- 最新收盘价
- 每日成交量汇总
