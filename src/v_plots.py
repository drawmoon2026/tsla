import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


TFS = ["5m", "15m", "1h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from V-event statistics.")
    parser.add_argument("--summary", default="v_summary.csv", help="Path to v_summary.csv")
    parser.add_argument("--events_dir", default=".", help="Directory containing v_events_{tf}.csv")
    parser.add_argument("--output_dir", default="figures", help="Directory to save figures")
    return parser.parse_args()


def load_events(tf: str, events_dir: Path) -> pd.DataFrame:
    path = events_dir / f"v_events_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing events file: {path}")
    df = pd.read_csv(
        path,
        parse_dates=["peak_t", "trough_t", "rebound_t"],
    )
    # Ensure timezone-awareness
    for col in ["peak_t", "trough_t", "rebound_t"]:
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")
        else:
            df[col] = df[col].dt.tz_convert("UTC")
    return df


def plot_count_bar(summary_df: pd.DataFrame, tf: str, outdir: Path) -> None:
    sdf = summary_df[summary_df["tf"] == tf].copy()
    if sdf.empty:
        return
    sdf["combo"] = sdf.apply(lambda r: f"X={r['X']:.3f}\nY={r['Y']:.3f}\nT={int(r['T'])}", axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(sdf["combo"], sdf["v_count"])
    plt.title(f"V count by X/Y/T — {tf}")
    plt.ylabel("V count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"{tf}_counts.png", dpi=150)
    plt.close()


def plot_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 40) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def compute_daily_counts(events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict] = []
    ny = "America/New_York"
    for tf, df in events.items():
        if df.empty:
            continue
        # Use rebound time to tag the completion day.
        dt_et = df["rebound_t"].dt.tz_convert(ny).dt.date
        counts = dt_et.value_counts().sort_index()
        for date, cnt in counts.items():
            rows.append({"date": pd.to_datetime(date), "tf": tf, "v_count": int(cnt)})
    daily = pd.DataFrame(rows).sort_values(["date", "tf"])
    return daily


def plot_daily_counts(daily: pd.DataFrame, outdir: Path) -> None:
    if daily.empty:
        return
    plt.figure(figsize=(10, 5))
    for tf in TFS:
        df_tf = daily[daily["tf"] == tf]
        if df_tf.empty:
            continue
        plt.plot(df_tf["date"], df_tf["v_count"], label=tf)
    plt.title("Daily V count (by rebound date, ET)")
    plt.xlabel("Date")
    plt.ylabel("V count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "daily_v_count.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    events_dir = Path(args.events_dir)

    summary_df = pd.read_csv(args.summary)
    events_by_tf: Dict[str, pd.DataFrame] = {}

    for tf in TFS:
        try:
            df_tf = load_events(tf, events_dir)
        except FileNotFoundError:
            continue
        events_by_tf[tf] = df_tf

        plot_count_bar(summary_df, tf, outdir)
        if not df_tf.empty:
            plot_hist(df_tf["drop_pct"], f"drop_pct distribution — {tf}", "drop_pct", outdir / f"{tf}_drop_hist.png")
            plot_hist(
                df_tf["bars_to_rebound"],
                f"bars_to_rebound distribution — {tf}",
                "bars_to_rebound",
                outdir / f"{tf}_rebound_hist.png",
                bins=30,
            )

    daily = compute_daily_counts(events_by_tf)
    if not daily.empty:
        daily.to_csv("daily_v_count.csv", index=False, date_format="%Y-%m-%d")
        plot_daily_counts(daily, outdir)
        print(f"Daily counts saved to daily_v_count.csv and plotted.")
    else:
        print("No daily counts to plot.")

    print(f"Figures saved to {outdir}")


if __name__ == "__main__":
    main()
