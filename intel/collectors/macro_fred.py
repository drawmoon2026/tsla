"""FRED 宏观周期序列采集器（T2，日频低频轮询）——N10 宏观相位层数据源.

来源：FRED fredgraph.csv 公开端点（免费、无需 API key）：
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

序列（N10 预登记，嵌套周期操作化）：
    FEDFUNDS  联邦基金有效利率（月）——利率周期相位（6 个月变化方向：加息/平台/降息）
    M2SL      M2 存量（月）——流动性周期（同比增速方向）
    INDPRO    工业生产指数（月）——库存周期 proxy（同比方向）。
              ⚠️ NAPM/ISM PMI 已从 FRED 下架（ISM 授权收回，实测 404），
              按 N10 预案降级为 INDPRO 同比方向替代口径，如实注明
    MANEMP    制造业就业人数（月）——库存周期第二 proxy（6 个月变化方向）
    DGS10     10 年期国债收益率（日）——长端方向
    T10Y2Y    10Y-2Y 期限利差（日）——曲线状态（倒挂/正常）

双时间戳口径（vintage 近似，如实声明局限）：
    observation_date   = 数据所属期（FRED 原始列）
    available_from_utc = 可知时刻近似——月度序列取 observation_date + 1 个月
                         （发布滞后 2-6 周不等，统一保守取 1 个月）；日度序列取
                         observation_date + 1 天。这是滞后近似而非真 ALFRED vintage：
                         月度宏观值事后会被修订，本口径用的是最终修订值+滞后时间戳，
                         非当时真实可见值——结论解读时须记得这层局限。

落盘：
    data/intel/macro/<SERIES>.csv  全历史（observation_date,value,available_from_utc）
    哨兵事件表：每序列近 400 天内的观测各一条（type=macro_obs，增量去重入库）

用法： .venv/bin/python -m intel.collectors.macro_fred --once
"""

from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import pandas as pd

from intel.collectors.base import Collector, cli, http_get

ROOT = Path(__file__).resolve().parents[2]
MACRO_DIR = ROOT / "data" / "intel" / "macro"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# series -> (freq, 中文说明)
SERIES: dict[str, tuple[str, str]] = {
    "FEDFUNDS": ("monthly", "联邦基金有效利率（利率周期相位）"),
    "M2SL": ("monthly", "M2 存量（流动性周期）"),
    "INDPRO": ("monthly", "工业生产指数（库存周期 proxy，NAPM 不可用的替代口径）"),
    "MANEMP": ("monthly", "制造业就业（库存周期第二 proxy）"),
    "DGS10": ("daily", "10 年期国债收益率（长端方向）"),
    "T10Y2Y": ("daily", "10Y-2Y 期限利差（曲线状态）"),
}

EVENT_WINDOW_DAYS = 400  # 只把近 400 天的观测发成哨兵事件（CSV 落全历史）


def _available_from(obs: pd.Timestamp, freq: str) -> pd.Timestamp:
    """可知时刻近似：月度 +1 个月、日度 +1 天，取 12:00 UTC."""
    if freq == "monthly":
        t = obs + pd.DateOffset(months=1)
    else:
        t = obs + pd.DateOffset(days=1)
    return t.tz_localize("UTC") + pd.Timedelta(hours=12)


def load_series(sid: str) -> pd.DataFrame:
    """研究侧读取接口：返回 observation_date(index) / value / available_from_utc."""
    df = pd.read_csv(MACRO_DIR / f"{sid}.csv", parse_dates=["observation_date"])
    df["available_from_utc"] = pd.to_datetime(df["available_from_utc"], utc=True)
    return df.set_index("observation_date").sort_index()


class MacroFredCollector(Collector):
    SOURCE = {
        "source_id": "macro_fred",
        "name": "FRED macro cycle series (fredgraph.csv)",
        "tier": "T2",
        "method": "csv",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.5,
        "notes": ("N10 宏观相位层：FEDFUNDS/M2SL/INDPRO/MANEMP/DGS10/T10Y2Y；"
                  "vintage 近似=月度滞后 1 个月、日度滞后 1 天（非真实时 vintage，"
                  "月度值含事后修订）；NAPM/ISM 已从 FRED 下架，INDPRO+MANEMP 替代"),
    }

    def fetch(self):
        raw: dict[str, pd.DataFrame] = {}
        for sid in SERIES:
            resp = http_get(FRED_URL.format(sid=sid))
            df = pd.read_csv(io.StringIO(resp.text))
            if "observation_date" not in df.columns or sid not in df.columns:
                raise ValueError(f"FRED {sid}: 响应结构异常（列 {list(df.columns)[:4]}）")
            raw[sid] = df
        return raw

    def normalize(self, raw: dict[str, pd.DataFrame]) -> list[dict]:
        MACRO_DIR.mkdir(parents=True, exist_ok=True)
        events: list[dict] = []
        cutoff = pd.Timestamp.now("UTC").tz_localize(None) - timedelta(days=EVENT_WINDOW_DAYS)
        for sid, df in raw.items():
            freq, desc = SERIES[sid]
            df = df.copy()
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df["value"] = pd.to_numeric(df[sid], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("observation_date")
            df["available_from_utc"] = [
                _available_from(o, freq).isoformat(timespec="seconds")
                for o in df["observation_date"]
            ]
            # 全历史落盘（研究侧唯一读取口径）
            df[["observation_date", "value", "available_from_utc"]].to_csv(
                MACRO_DIR / f"{sid}.csv", index=False)
            # 近窗观测入哨兵事件表（dedupe_key 保证增量）
            for _, r in df[df["observation_date"] >= cutoff].iterrows():
                obs = r["observation_date"].date()
                events.append({
                    "dedupe_key": f"{sid}_{obs}",
                    "event_time_utc": r["available_from_utc"],
                    "symbol": None,
                    "type": "macro_obs",
                    "title": f"{sid} {obs} = {r['value']:.2f}（{desc}）",
                    "url": FRED_URL.format(sid=sid),
                    "payload": {"series": sid, "freq": freq,
                                "observation_date": str(obs),
                                "value": float(r["value"])},
                })
        return events


if __name__ == "__main__":
    cli(MacroFredCollector)
