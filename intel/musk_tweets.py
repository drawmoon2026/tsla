"""Musk 发帖归档：HuggingFace fdaudens/musk-tweets（Sprinklr 导出，公开数据集）.

覆盖与口径（诚实标注）：
- 归档覆盖 2013-03 → 2025-05-08；**2025-05 之后无数据**（X API 关闭后无免费全量归档）。
- event_time_utc = CreatedTime（带 UTC 偏移的发帖时刻）。发帖即公开，无披露时延。
- 只保留 SenderScreenName == elonmusk（大小写不敏感）的 X 平台条目
  （X Update=原创, X Reply=回复, X Repost=转发）；Facebook/YouTube 条目丢弃。
- 本轮不做任何 LLM 内容判断；payload 存原文，下游只做关键词字符串匹配。
- 完整性不可证：Sprinklr 归档可能漏帖/含已删推，密度逐年核对与公开报道量级一致
  （2018 ~7 帖/日 → 2024 ~80 帖/日），作 best-effort 源使用。

用法： .venv/bin/python -m intel.musk_tweets
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://huggingface.co/datasets/fdaudens/musk-tweets/resolve/main/"
    "elonmusk_data_fixed%20-%20Sheet1.csv"
)
UA = {"User-Agent": "TSLA-research tom (drawmoon2026@gmail.com)"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "intel"
SINCE = "2018-01-01"
KEEP_TYPES = {"X Update": "post", "X Reply": "reply", "X Repost": "repost"}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "_musk_tweets_raw.csv"
    if not raw_path.exists():
        req = urllib.request.Request(URL, headers=UA)
        raw_path.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    df = pd.read_csv(raw_path, engine="python", on_bad_lines="skip")
    df = df[df["SenderScreenName"].astype(str).str.strip().str.lower() == "elonmusk"]
    df = df[df["MessageType"].isin(KEEP_TYPES)]
    t = pd.to_datetime(df["CreatedTime"], errors="coerce", utc=True, format="mixed")
    df = df[t.notna() & (t >= pd.Timestamp(SINCE, tz="UTC"))]
    t = t[df.index]

    rows = []
    for (_, r), ts in zip(df.iterrows(), t):
        rows.append(
            {
                "event_time_utc": ts.isoformat(),
                "source": "musk_tweets",
                "type": f"musk_{KEEP_TYPES[r['MessageType']]}",
                "payload": json.dumps(
                    {"text": str(r.get("Message", "") or "")[:2000]},
                    ensure_ascii=False,
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("event_time_utc")
    out.to_csv(DATA_DIR / "musk_tweets.csv", index=False)
    print(
        f"musk tweets since {SINCE}: {len(out)} "
        f"({out['event_time_utc'].min()} -> {out['event_time_utc'].max()}) "
        f"-> {DATA_DIR / 'musk_tweets.csv'}"
    )


if __name__ == "__main__":
    main()
