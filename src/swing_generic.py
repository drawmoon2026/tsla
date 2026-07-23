"""
ATR 自适应 ZigZag 波段检测与统计
目标：用通用算法尽可能覆盖历史波段（例如此前的 95 笔）。
方法：
- 动态阈值：以 N=14 的 ATR 百分比作为最小摆动幅度（atr_mult 可调）。
- 峰谷检测：单序列遍历，按累积涨跌幅超阈值反转确定 pivot。
- 去重：相邻 pivot 方向必须交替，突出度不足的点会被替换。
输出：
- swings.csv：所有 pivot（时间、价格、类型）
- segments.csv：峰->谷->峰 三点形成的波段，给出跌幅/反弹幅度
- summary.txt：统计信息
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import load_bars


def zigzag_atr(close: pd.Series, thresholds: pd.Series) -> List[Tuple[pd.Timestamp, float, str]]:
    """
    ATR 自适应 ZigZag：极值 (extreme) 与已确认 pivot 分离追踪。
    thresholds 是逐 bar 的反转阈值（百分比），NaN 处沿用趋势不反转。
    返回 [(ts, price, 'H'/'L'), ...]
    """
    prices = close.to_numpy()
    thr_arr = thresholds.to_numpy()
    idx = close.index
    pivots: List[Tuple[pd.Timestamp, float, str]] = []
    trend = 1  # start assuming up; first confirmed reversal fixes it
    ext_idx = 0
    ext_price = prices[0]

    for i in range(1, len(prices)):
        p = prices[i]
        thr = thr_arr[i]
        if trend >= 0:
            if p > ext_price:
                ext_idx, ext_price = i, p
            elif not np.isnan(thr) and (ext_price - p) / ext_price >= thr:
                pivots.append((idx[ext_idx], ext_price, "H"))
                trend = -1
                ext_idx, ext_price = i, p
        else:
            if p < ext_price:
                ext_idx, ext_price = i, p
            elif not np.isnan(thr) and (p - ext_price) / ext_price >= thr:
                pivots.append((idx[ext_idx], ext_price, "L"))
                trend = 1
                ext_idx, ext_price = i, p

    pivots.append((idx[ext_idx], ext_price, "H" if trend >= 0 else "L"))
    # 相邻同类 pivot 合并，保留更极端者
    cleaned: List[Tuple[pd.Timestamp, float, str]] = []
    for ts, p, k in pivots:
        if cleaned and cleaned[-1][2] == k:
            if (k == "H" and p > cleaned[-1][1]) or (k == "L" and p < cleaned[-1][1]):
                cleaned[-1] = (ts, p, k)
            continue
        cleaned.append((ts, p, k))
    return cleaned


def build_segments(pivots: List[Tuple[pd.Timestamp, float, str]]) -> pd.DataFrame:
    rows = []
    for i in range(len(pivots) - 2):
        a, b, c = pivots[i], pivots[i + 1], pivots[i + 2]
        if a[2] == "H" and b[2] == "L" and c[2] == "H":
            drop = (a[1] - b[1]) / a[1]
            rebound = (c[1] - b[1]) / b[1]
            rows.append(
                {
                    "peak_t": a[0],
                    "peak_p": a[1],
                    "trough_t": b[0],
                    "trough_p": b[1],
                    "rebound_t": c[0],
                    "rebound_p": c[1],
                    "drop_pct": drop,
                    "rebound_pct": rebound,
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/TSLA_5m_60d.csv")
    ap.add_argument("--atr_window", type=int, default=14)
    ap.add_argument("--atr_mult", type=float, default=1.2, help="ATR multiplier to set swing threshold")
    ap.add_argument("--output_dir", default="outputs/swing_generic")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_bars(args.input)
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(args.atr_window).mean()

    close = df["Close"]
    # 逐 bar 滚动阈值，shift(1) 保证 bar i 的阈值只用 i-1 及之前的信息
    thresholds = (atr / close).shift(1) * args.atr_mult

    pivots = zigzag_atr(close, thresholds)
    atr_pct = float(thresholds.mean())  # 仅用于 summary 展示
    piv_df = pd.DataFrame(pivots, columns=["ts", "price", "kind"])
    seg_df = build_segments(pivots)

    piv_df.to_csv(outdir / "swings.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    seg_df.to_csv(outdir / "segments.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    summary = (
        f"ATR window={args.atr_window}, atr_mult={args.atr_mult}, atr_pct={atr_pct:.4%}\n"
        f"Swings: {len(piv_df)}\n"
        f"Segments (H-L-H): {len(seg_df)}\n"
        f"Mean drop: {seg_df['drop_pct'].mean():.4%}, Mean rebound: {seg_df['rebound_pct'].mean():.4%}\n"
    )
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
