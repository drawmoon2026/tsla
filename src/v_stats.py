import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TF_RULES = {"5m": None, "15m": "15min", "1h": "1h"}

DEFAULT_X = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%
DEFAULT_Y = [0.003, 0.005, 0.01]  # 0.3%, 0.5%, 1%
DEFAULT_T = {
    "5m": [12, 24, 48, 78],
    "15m": [4, 8, 16, 26],
    "1h": [2, 4, 8],
}


@dataclass
class VEvent:
    tf: str
    x: float
    y: float
    T: int
    peak_t: pd.Timestamp
    peak_p: float
    trough_t: pd.Timestamp
    trough_p: float
    rebound_t: pd.Timestamp
    rebound_p: float
    drop_pct: float
    rebound_pct: float
    bars_to_trough: int
    bars_to_rebound: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect V-shaped events across multiple timeframes.")
    parser.add_argument("--tf", choices=["5m", "15m", "1h", "all"], default="all", help="Timeframe to run (default: all)")
    parser.add_argument("--x", type=float, help="Drop threshold (fraction). If omitted, use defaults.")
    parser.add_argument("--y", type=float, help="Rebound threshold (fraction). If omitted, use defaults.")
    parser.add_argument("--tbars", type=int, help="Max bars from trough to rebound. If omitted, use defaults per tf.")
    parser.add_argument("--input_csv", default="data/TSLA_5m_60d.csv", help="Input CSV produced by data_fetch.py")
    parser.add_argument("--lookback", type=int, default=3, help="Bars to look back for local extrema (default: 3)")
    parser.add_argument("--lookforward", type=int, default=3, help="Bars to look forward for local extrema (default: 3)")
    return parser.parse_args()


def load_base_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.set_index("Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Standardize columns.
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    return df[cols].sort_index()


def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = TF_RULES[tf]
    if rule is None:
        return df.copy()
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def find_local_extrema(series: pd.Series, lb: int, lf: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for local peaks and troughs."""
    vals = series.to_numpy()
    n = len(vals)
    peaks = np.zeros(n, dtype=bool)
    troughs = np.zeros(n, dtype=bool)
    for i in range(lb, n - lf):
        window = vals[i - lb : i + lf + 1]
        val = vals[i]
        if val >= window.max():
            peaks[i] = True
        if val <= window.min():
            troughs[i] = True
    return peaks, troughs


def detect_v_events(
    df: pd.DataFrame,
    tf: str,
    x: float,
    y: float,
    tbars: int,
    lb: int,
    lf: int,
) -> List[VEvent]:
    close = df["Close"]
    peaks_mask, troughs_mask = find_local_extrema(close, lb, lf)
    vals = close.to_numpy()
    idx = close.index
    n = len(close)

    events: List[VEvent] = []
    i = lb
    while i < n - lf:
        if not peaks_mask[i]:
            i += 1
            continue

        peak_i = i
        peak_p = vals[i]

        # search for trough after peak
        trough_i: Optional[int] = None
        j = i + 1
        while j < n - lf:
            if troughs_mask[j]:
                drop = (peak_p - vals[j]) / peak_p
                if drop >= x:
                    trough_i = j
                    trough_p = vals[j]
                    break
            j += 1

        if trough_i is None:
            i += 1
            continue

        # search for rebound within tbars after trough
        rebound_i: Optional[int] = None
        k = trough_i + 1
        limit = min(n - 1, trough_i + tbars)
        while k <= limit:
            if (vals[k] - trough_p) / trough_p >= y:
                rebound_i = k
                rebound_p = vals[k]
                break
            k += 1

        if rebound_i is not None:
            events.append(
                VEvent(
                    tf=tf,
                    x=x,
                    y=y,
                    T=tbars,
                    peak_t=idx[peak_i],
                    peak_p=peak_p,
                    trough_t=idx[trough_i],
                    trough_p=trough_p,
                    rebound_t=idx[rebound_i],
                    rebound_p=rebound_p,
                    drop_pct=(peak_p - trough_p) / peak_p,
                    rebound_pct=(rebound_p - trough_p) / trough_p,
                    bars_to_trough=trough_i - peak_i,
                    bars_to_rebound=rebound_i - trough_i,
                )
            )
            i = rebound_i + 1  # move past rebound to avoid overlap
        else:
            # timeout: resume search after window
            i = limit + 1

    return events


def run_grid_for_tf(
    df_tf: pd.DataFrame,
    tf: str,
    x_list: Iterable[float],
    y_list: Iterable[float],
    t_list: Iterable[int],
    lb: int,
    lf: int,
) -> Tuple[pd.DataFrame, List[Dict]]:
    all_events: List[VEvent] = []
    summary_rows: List[Dict] = []

    for x in x_list:
        for y in y_list:
            for T in t_list:
                events = detect_v_events(df_tf, tf, x, y, T, lb, lf)
                all_events.extend(events)

                if events:
                    drops = [e.drop_pct for e in events]
                    rebounds = [e.rebound_pct for e in events]
                    times = [e.bars_to_rebound for e in events]
                    summary_rows.append(
                        {
                            "tf": tf,
                            "X": x,
                            "Y": y,
                            "T": T,
                            "v_count": len(events),
                            "avg_drop": float(np.mean(drops)),
                            "avg_rebound": float(np.mean(rebounds)),
                            "median_time_to_rebound": float(np.median(times)),
                            "p90_time_to_rebound": float(np.percentile(times, 90)),
                        }
                    )
                else:
                    summary_rows.append(
                        {
                            "tf": tf,
                            "X": x,
                            "Y": y,
                            "T": T,
                            "v_count": 0,
                            "avg_drop": np.nan,
                            "avg_rebound": np.nan,
                            "median_time_to_rebound": np.nan,
                            "p90_time_to_rebound": np.nan,
                        }
                    )

    events_df = pd.DataFrame(
        [
            {
                "tf": e.tf,
                "X": e.x,
                "Y": e.y,
                "T": e.T,
                "peak_t": e.peak_t,
                "peak_p": e.peak_p,
                "trough_t": e.trough_t,
                "trough_p": e.trough_p,
                "rebound_t": e.rebound_t,
                "rebound_p": e.rebound_p,
                "drop_pct": e.drop_pct,
                "rebound_pct": e.rebound_pct,
                "bars_to_trough": e.bars_to_trough,
                "bars_to_rebound": e.bars_to_rebound,
            }
            for e in all_events
        ]
    )

    return events_df, summary_rows


def save_events(tf: str, events_df: pd.DataFrame) -> None:
    out_path = Path(f"v_events_{tf}.csv")
    events_df.to_csv(out_path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")


def write_summary_md(all_events: Dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("# V 事件统计摘要")
    lines.append("")
    lines.append("V 定义：从局部峰值开始，价格下跌超过阈值 X 至谷值，随后在窗口 T 内反弹超过阈值 Y；谷值确认后直到反弹成功或超时前不再寻找下一次 V，以避免重叠。局部峰/谷使用对称窗口 (lookback=lookforward=3) 检测。")
    lines.append("")

    for tf, df in all_events.items():
        lines.append(f"## 时间尺度：{tf}")
        if df.empty:
            lines.append("无事件。")
            lines.append("")
            continue

        top = df.sort_values(["drop_pct", "rebound_pct"], ascending=False).head(10)
        lines.append("Top 10 按跌幅排序：")
        lines.append("")
        lines.append("| peak_t | drop_pct | rebound_pct | bars_to_rebound | X | Y | T |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['peak_t']} | {row['drop_pct']:.4f} | {row['rebound_pct']:.4f} | {int(row['bars_to_rebound'])} | {row['X']} | {row['Y']} | {int(row['T'])} |"
            )
        lines.append("")

    Path("summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_df = load_base_df(args.input_csv)

    if args.tf == "all":
        tf_list = ["5m", "15m", "1h"]
    else:
        tf_list = [args.tf]

    all_summary: List[Dict] = []
    events_by_tf: Dict[str, pd.DataFrame] = {}

    for tf in tf_list:
        df_tf = resample_tf(base_df, tf)
        x_list = [args.x] if args.x is not None else DEFAULT_X
        y_list = [args.y] if args.y is not None else DEFAULT_Y
        t_list = [args.tbars] if args.tbars is not None else DEFAULT_T[tf]

        events_df, summary_rows = run_grid_for_tf(df_tf, tf, x_list, y_list, t_list, args.lookback, args.lookforward)
        events_by_tf[tf] = events_df
        all_summary.extend(summary_rows)

        save_events(tf, events_df)
        print(f"[{tf}] events: {len(events_df)} -> v_events_{tf}.csv")

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv("v_summary.csv", index=False)
    print(f"Summary rows: {len(summary_df)} -> v_summary.csv")

    write_summary_md(events_by_tf)
    print("summary.md generated.")


if __name__ == "__main__":
    main()
