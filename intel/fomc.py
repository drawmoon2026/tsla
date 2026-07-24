"""FOMC 决议事件：联储官网新闻稿 JSON 流（机读，带精确发布时刻）.

来源 https://www.federalreserve.gov/json/ne-press.json —— 官网新闻列表的数据后端，
覆盖 2016 起全部新闻稿。筛 title == "Federal Reserve issues FOMC statement"，
其 `d` 字段即发布时刻（美东时间，常规会 14:00 ET；2020-03 两次紧急决议为真实
非常规时刻，比硬编码日历表更准）。event_time_utc = 该时刻转 UTC。

用法： .venv/bin/python -m intel.fomc
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://www.federalreserve.gov/json/ne-press.json"
UA = {"User-Agent": "TSLA-research tom (drawmoon2026@gmail.com)"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "intel"
SINCE = "2018-01-01"


def main() -> None:
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers=UA), timeout=30
    ).read()
    items = [x for x in json.loads(raw.decode("utf-8-sig")) if "d" in x and "t" in x]
    rows = []
    for x in items:
        if "FOMC statement" not in x["t"]:
            continue
        et = pd.Timestamp(x["d"], tz="America/New_York")
        if et.strftime("%Y-%m-%d") < SINCE:
            continue
        rows.append(
            {
                "event_time_utc": et.tz_convert("UTC").isoformat(),
                "source": "fomc",
                "type": "fomc_statement",
                "payload": json.dumps(
                    {"title": x["t"], "link": x.get("l", ""), "et_time": x["d"]}
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("event_time_utc")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "fomc.csv", index=False)
    print(f"FOMC statements since {SINCE}: {len(df)} -> {DATA_DIR / 'fomc.csv'}")


if __name__ == "__main__":
    main()
