import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline: fetch data, compute V stats, plot outputs.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of price data.")
    args = parser.parse_args()

    # 1) Fetch data
    fetch_cmd = [sys.executable, "src/data_fetch.py"]
    if args.refresh:
        fetch_cmd.append("--refresh")
    run(fetch_cmd)

    # 2) V stats for all timeframes (full grid)
    run([sys.executable, "src/v_stats.py"])

    # 3) Plots
    run([sys.executable, "src/v_plots.py"])

    # 4) Copy summary to outputs/
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    summary_src = Path("summary.md")
    summary_dst = outputs_dir / "summary.md"
    if summary_src.exists():
        shutil.copy(summary_src, summary_dst)
        print(f"summary copied to {summary_dst}")
    else:
        print("summary.md not found; skipping copy.")


if __name__ == "__main__":
    main()
