import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data_io import ET, load_bars, resample_bars  # noqa: E402


TF_MINUTES = {"5m": None, "15m": 15, "1h": 60}

DEFAULT_X = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%
DEFAULT_Y = [0.003, 0.005, 0.01]  # 0.3%, 0.5%, 1%
DEFAULT_T = {
    "5m": [12, 24, 48, 78],
    "15m": [4, 8, 16, 26],
    "1h": [2, 4, 8],
}

UNIQUE_KEY = ["peak_t", "trough_t"]  # one physical V event == one (peak_t, trough_t) pair


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
    trough_confirm_t: pd.Timestamp
    rebound_t: Optional[pd.Timestamp]  # NaT when outcome == "timeout"
    rebound_p: float                   # NaN when outcome == "timeout"
    drop_pct: float
    rebound_pct: float                 # NaN when outcome == "timeout"
    bars_to_trough: int
    bars_to_rebound: float             # NaN when outcome == "timeout"
    outcome: str                       # "rebounded" | "timeout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect V-shaped events across multiple timeframes.")
    parser.add_argument("--tf", choices=["5m", "15m", "1h", "all"], default="all", help="Timeframe to run (default: all)")
    parser.add_argument("--x", type=float, help="Drop threshold (fraction). If omitted, use defaults.")
    parser.add_argument("--y", type=float, help="Rebound threshold (fraction). If omitted, use defaults.")
    parser.add_argument("--tbars", type=int, help="Max bars from trough to rebound. If omitted, use defaults per tf.")
    parser.add_argument("--input_csv", default="data/TSLA_5m_60d.csv", help="Input CSV produced by data_fetch.py")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for v_events_*.csv, v_summary.csv and summary.md (default: project root, legacy layout)",
    )
    parser.add_argument("--lookback", type=int, default=3, help="Bars to look back for local extrema (default: 3)")
    parser.add_argument("--lookforward", type=int, default=3, help="Bars to look forward for local extrema (default: 3)")
    parser.add_argument(
        "--max_drop_bars",
        type=int,
        default=None,
        help="Max bars allowed between peak and trough (default: one trading day of bars for the timeframe)",
    )
    return parser.parse_args()


def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    minutes = TF_MINUTES[tf]
    if minutes is None:
        return df.copy()
    return resample_bars(df, minutes)


def bars_per_trading_day(index: pd.DatetimeIndex) -> int:
    """Median number of bars per ET trading day."""
    counts = pd.Series(1, index=index).groupby(index.tz_convert(ET).date).size()
    return int(counts.median())


def find_local_extrema(series: pd.Series, lb: int, lf: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for local peaks and troughs.

    Depends only on (lb, lf) — compute once per timeframe and reuse across
    the whole (x, y, T) parameter grid.
    """
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
    peaks_mask: np.ndarray,
    troughs_mask: np.ndarray,
    et_dates: np.ndarray,
    max_drop_bars: int,
) -> List[VEvent]:
    """Detect V events.

    Event definition (post-review fixes):
    - peak -> trough search is capped at `max_drop_bars` bars and never crosses
      an ET trading day;
    - if a higher close shows up while searching for the trough, the peak
      anchor is re-based to it, so drop_pct is always relative to the most
      recent peak;
    - `trough_confirm_t` is the timestamp of the bar `lf` bars after the
      trough — the earliest moment the trough is identifiable in real time;
    - on rebound timeout the scan resumes at the bar after the original peak,
      so other peaks inside the window are not skipped.

    Survivorship-bias fix: triggers whose rebound times out are NO longer
    dropped — they are emitted with outcome="timeout" and NaN/NaT rebound
    fields, so the event table reflects every trigger a live system would
    have acted on. A rebound must also be fully observable (trough + tbars
    within the data) for a timeout to be declared; if the data ends
    mid-window the trigger is skipped as unresolved.
    """
    close = df["Close"]
    vals = close.to_numpy()
    idx = close.index
    n = len(close)

    events: List[VEvent] = []
    seen_timeouts: set = set()  # (peak_i, trough_i) already emitted as timeout
    i = lb
    while i < n - lf:
        if not peaks_mask[i]:
            i += 1
            continue

        orig_i = i
        peak_i = i
        peak_p = vals[i]

        # search for trough after peak (bounded, same ET trading day, re-anchoring)
        trough_i: Optional[int] = None
        trough_p = np.nan
        j = i + 1
        while j < n - lf:
            if et_dates[j] != et_dates[peak_i] or (j - peak_i) > max_drop_bars:
                break
            if vals[j] > peak_p:
                # higher close: re-base the peak anchor
                peak_i = j
                peak_p = vals[j]
            elif troughs_mask[j]:
                drop = (peak_p - vals[j]) / peak_p
                if drop >= x:
                    trough_i = j
                    trough_p = vals[j]
                    break
            j += 1

        if trough_i is None:
            i = orig_i + 1
            continue

        # search for rebound within tbars after trough
        rebound_i: Optional[int] = None
        rebound_p = np.nan
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
                    trough_confirm_t=idx[min(trough_i + lf, n - 1)],
                    rebound_t=idx[rebound_i],
                    rebound_p=rebound_p,
                    drop_pct=(peak_p - trough_p) / peak_p,
                    rebound_pct=(rebound_p - trough_p) / trough_p,
                    bars_to_trough=trough_i - peak_i,
                    bars_to_rebound=float(rebound_i - trough_i),
                    outcome="rebounded",
                )
            )
            i = rebound_i + 1  # move past rebound to avoid overlap
        else:
            # Timeout: record the failed trigger too (survivorship-bias fix),
            # but only when the full T-bar window is inside the data —
            # otherwise the outcome is unresolved, not a timeout.
            window_complete = (trough_i + tbars) <= (n - 1)
            if window_complete and (peak_i, trough_i) not in seen_timeouts:
                seen_timeouts.add((peak_i, trough_i))
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
                        trough_confirm_t=idx[min(trough_i + lf, n - 1)],
                        rebound_t=pd.NaT,
                        rebound_p=np.nan,
                        drop_pct=(peak_p - trough_p) / peak_p,
                        rebound_pct=np.nan,
                        bars_to_trough=trough_i - peak_i,
                        bars_to_rebound=np.nan,
                        outcome="timeout",
                    )
                )
            # resume right after the original peak so other peaks
            # inside the window are still examined (was: i = limit + 1)
            i = orig_i + 1

    return events


def run_grid_for_tf(
    df_tf: pd.DataFrame,
    tf: str,
    x_list: Iterable[float],
    y_list: Iterable[float],
    t_list: Iterable[int],
    lb: int,
    lf: int,
    max_drop_bars: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Run the parameter grid for one timeframe.

    Returns (full grid events_df, deduplicated events_df, summary rows).
    The dedup table keeps one row per physical event, keyed on (peak_t,
    trough_t), keeping the loosest (smallest X/Y/T) parameter combo that
    detected it, plus `n_param_combos` = how many combos detected it.
    """
    # extrema depend only on (lb, lf): compute once, reuse across the grid
    peaks_mask, troughs_mask = find_local_extrema(df_tf["Close"], lb, lf)
    et_dates = np.asarray(df_tf.index.tz_convert(ET).date)
    if max_drop_bars is None:
        max_drop_bars = bars_per_trading_day(df_tf.index)

    all_events: List[VEvent] = []
    summary_rows: List[Dict] = []

    for x in x_list:
        for y in y_list:
            for T in t_list:
                events = detect_v_events(
                    df_tf, tf, x, y, T, lb, lf, peaks_mask, troughs_mask, et_dates, max_drop_bars
                )
                all_events.extend(events)

                # v_count keeps its legacy meaning: SUCCESSFUL (rebounded)
                # events only. Timeout triggers are reported separately so
                # per-combo rebound stats keep their original denominator.
                ok = [e for e in events if e.outcome == "rebounded"]
                to = [e for e in events if e.outcome == "timeout"]
                row = {
                    "tf": tf,
                    "X": x,
                    "Y": y,
                    "T": T,
                    "v_count": len(ok),
                    "timeout_count": len(to),
                    "trigger_count": len(events),
                    "timeout_rate": len(to) / len(events) if events else np.nan,
                    # avg_drop covers ALL triggers (drop is known either way)
                    "avg_drop": float(np.mean([e.drop_pct for e in events])) if events else np.nan,
                    # rebound stats are defined on rebounded events only
                    "avg_rebound": float(np.mean([e.rebound_pct for e in ok])) if ok else np.nan,
                    "median_time_to_rebound": float(np.median([e.bars_to_rebound for e in ok])) if ok else np.nan,
                    "p90_time_to_rebound": float(np.percentile([e.bars_to_rebound for e in ok], 90)) if ok else np.nan,
                }
                summary_rows.append(row)

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
                "trough_confirm_t": e.trough_confirm_t,
                "rebound_t": e.rebound_t,
                "rebound_p": e.rebound_p,
                "drop_pct": e.drop_pct,
                "rebound_pct": e.rebound_pct,
                "bars_to_trough": e.bars_to_trough,
                "bars_to_rebound": e.bars_to_rebound,
                "outcome": e.outcome,
            }
            for e in all_events
        ]
    )

    unique_df = dedup_events(events_df)
    if not events_df.empty:
        # mark the representative row of each physical event in the full table
        rep_keys = set(map(tuple, unique_df[UNIQUE_KEY + ["X", "Y", "T"]].to_numpy().tolist()))
        events_df["is_unique_rep"] = [
            tuple(row) in rep_keys for row in events_df[UNIQUE_KEY + ["X", "Y", "T"]].to_numpy().tolist()
        ]

    return events_df, unique_df, summary_rows


def dedup_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per physical event, keyed on (peak_t, trough_t).

    Contains BOTH outcomes. If any parameter combo saw the event rebound, the
    representative row is a rebounded one (loosest X/Y/T among those);
    an event is outcome="timeout" only when it timed out under every combo
    that triggered on it. n_param_combos counts all combos that triggered.
    """
    if events_df.empty:
        return events_df.copy()
    df = events_df.copy()
    # rebounded rows sort before timeout rows within a (peak_t, trough_t) key
    df["_outcome_rank"] = (df["outcome"] != "rebounded").astype(int)
    df = df.sort_values(UNIQUE_KEY + ["_outcome_rank", "X", "Y", "T"])
    n_combos = df.groupby(UNIQUE_KEY).size().rename("n_param_combos")
    df = df.drop_duplicates(subset=UNIQUE_KEY, keep="first")
    df = df.merge(n_combos, left_on=UNIQUE_KEY, right_index=True)
    df = df.drop(columns=["_outcome_rank"])
    return df.sort_values("peak_t").reset_index(drop=True)


def save_events(tf: str, events_df: pd.DataFrame, unique_df: pd.DataFrame, out_dir: Path) -> None:
    date_fmt = "%Y-%m-%dT%H:%M:%S%z"
    events_df.to_csv(out_dir / f"v_events_{tf}.csv", index=False, date_format=date_fmt)
    unique_df.to_csv(out_dir / f"v_events_{tf}_unique.csv", index=False, date_format=date_fmt)


def write_summary_md(unique_events: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    lines = []
    lines.append("# V 事件统计摘要")
    lines.append("")
    lines.append("V 定义：从局部峰值开始，价格下跌超过阈值 X 至谷值，随后在窗口 T 内反弹超过阈值 Y；谷值确认后直到反弹成功或超时前不再寻找下一次 V，以避免重叠。局部峰/谷使用对称窗口 (lookback=lookforward=3) 检测。峰→谷搜索限制在同一 ET 交易日内且不超过一个交易日的 bar 数；搜索途中出现更高收盘价时峰值锚点会重置。")
    lines.append("")
    lines.append("口径：下表基于按 (peak_t, trough_t) 去重后的**物理事件**（同一事件被多套参数检出只计一次，X/Y/T 为检出该事件的最宽松组合，n_param_combos 为检出它的参数组数）。逐参数组计数见 v_summary.csv。")
    lines.append("")
    lines.append("幸存者偏差修正：事件表现在**同时包含**反弹成功 (outcome=rebounded) 与反弹超时 (outcome=timeout) 的触发。timeout 事件的 rebound_t / rebound_pct / bars_to_rebound 为 NaN。去重时若任一参数组内该事件反弹成功，则代表行取成功行（最宽松组合）；只有在所有触发它的参数组下都超时的事件才标记为 timeout。")
    lines.append("")

    for tf, df in unique_events.items():
        lines.append(f"## 时间尺度：{tf}")
        if df.empty:
            lines.append("无事件。")
            lines.append("")
            continue

        n_ok = int((df["outcome"] == "rebounded").sum())
        n_to = int((df["outcome"] == "timeout").sum())
        lines.append(
            f"物理事件数（去重后，含 timeout）：{len(df)}"
            f"（rebounded: {n_ok}, timeout: {n_to}, timeout 占比: {n_to / len(df):.1%}）"
        )
        lines.append("")

        top = df.sort_values(["drop_pct", "rebound_pct"], ascending=False).head(10)
        lines.append("Top 10 按跌幅排序（去重后，**含 timeout 事件**；timeout 行的 rebound 列为 n/a）：")
        lines.append("")
        lines.append("| peak_t | outcome | drop_pct | rebound_pct | bars_to_rebound | X | Y | T | n_param_combos |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in top.iterrows():
            reb = f"{row['rebound_pct']:.4f}" if np.isfinite(row["rebound_pct"]) else "n/a"
            btr = f"{int(row['bars_to_rebound'])}" if np.isfinite(row["bars_to_rebound"]) else "n/a"
            lines.append(
                f"| {row['peak_t']} | {row['outcome']} | {row['drop_pct']:.4f} | {reb} | {btr} | {row['X']} | {row['Y']} | {int(row['T'])} | {int(row['n_param_combos'])} |"
            )
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_df = load_bars(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tf == "all":
        tf_list = ["5m", "15m", "1h"]
    else:
        tf_list = [args.tf]

    all_summary: List[Dict] = []
    unique_by_tf: Dict[str, pd.DataFrame] = {}

    for tf in tf_list:
        df_tf = resample_tf(base_df, tf)
        x_list = [args.x] if args.x is not None else DEFAULT_X
        y_list = [args.y] if args.y is not None else DEFAULT_Y
        t_list = [args.tbars] if args.tbars is not None else DEFAULT_T[tf]

        events_df, unique_df, summary_rows = run_grid_for_tf(
            df_tf, tf, x_list, y_list, t_list, args.lookback, args.lookforward, args.max_drop_bars
        )
        unique_by_tf[tf] = unique_df
        all_summary.extend(summary_rows)

        save_events(tf, events_df, unique_df, out_dir)
        if not unique_df.empty:
            n_ok = int((unique_df["outcome"] == "rebounded").sum())
            n_to = int((unique_df["outcome"] == "timeout").sum())
            outcome_note = f" (rebounded: {n_ok}, timeout: {n_to})"
        else:
            outcome_note = ""
        print(
            f"[{tf}] grid rows: {len(events_df)} -> v_events_{tf}.csv | "
            f"physical (deduped): {len(unique_df)}{outcome_note} -> v_events_{tf}_unique.csv"
        )

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out_dir / "v_summary.csv", index=False)
    print(f"Summary rows: {len(summary_df)} -> v_summary.csv")

    write_summary_md(unique_by_tf, out_dir)
    print("summary.md generated (Top10 based on deduped physical events, incl. timeout).")


if __name__ == "__main__":
    main()
