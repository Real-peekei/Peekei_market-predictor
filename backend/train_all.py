"""
train_all.py
Trains a model for every timeframe you've fetched data for, one command.
Skips any timeframe whose data file doesn't exist yet (with a note) rather
than failing the whole run. Covers all 12 of Twelve Data's intervals plus
yfinance's 10 (they overlap) - whichever you've actually fetched data for.

Works for any asset(s) - not gold-specific. If you don't pass --asset, it
auto-discovers every asset you've fetched data for (by scanning data/ for
{ASSET}_{interval}.csv files) and trains all of them.

Usage:
    python train_all.py                       # trains every asset you have data for
    python train_all.py --asset GC_F          # just gold futures
    python train_all.py --asset AAPL --intervals 1d 1h 15m
    python train_all.py --fast                 # quick rough pass on everything
"""

import argparse
import subprocess
import sys
from pathlib import Path

from intervals import interval_sort_key

ALL_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "45m", "1h", "90m", "2h", "4h", "8h", "1d", "1wk", "1mo"]
DATA_DIR = Path("data")


def discover_assets() -> dict[str, list[str]]:
    """Scans data/ for {ASSET}_{interval}.csv files and returns
    {asset_key: [intervals available]} for everything found."""
    found: dict[str, list[str]] = {}
    for csv_path in DATA_DIR.glob("*.csv"):
        stem = csv_path.stem
        for interval in ALL_INTERVALS:
            suffix = f"_{interval}"
            if stem.endswith(suffix):
                asset_key = stem[: -len(suffix)]
                found.setdefault(asset_key, []).append(interval)
                break
    for asset_key in found:
        found[asset_key] = sorted(found[asset_key], key=interval_sort_key)
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None,
                         help="Train only this asset key (e.g. GC_F, AAPL). "
                              "Omit to auto-discover and train every asset you have data for.")
    parser.add_argument("--intervals", nargs="+", default=None, choices=ALL_INTERVALS,
                         help="Restrict to these intervals (default: all available per asset)")
    parser.add_argument("--fast", action="store_true", help="pass --fast through to train_model.py for every run")
    args = parser.parse_args()

    discovered = discover_assets()
    if not discovered:
        raise SystemExit(f"No data files found in {DATA_DIR}/ - run fetch_all.py or data_fetch.py first.")

    if args.asset:
        if args.asset not in discovered:
            raise SystemExit(f"No data found for asset '{args.asset}'. "
                              f"Assets with data available: {sorted(discovered.keys())}")
        assets_to_train = {args.asset: discovered[args.asset]}
    else:
        assets_to_train = discovered
        print(f"No --asset given - training every discovered asset: {sorted(discovered.keys())}\n")

    trained, skipped, failed = [], [], []
    for asset_key, available_intervals in assets_to_train.items():
        intervals_to_run = args.intervals or available_intervals
        for interval in intervals_to_run:
            data_path = DATA_DIR / f"{asset_key}_{interval}.csv"
            if not data_path.exists():
                print(f"--- {asset_key} {interval}: SKIPPED (no data file at {data_path})\n")
                skipped.append(f"{asset_key}/{interval}")
                continue

            print(f"--- training {asset_key} {interval} ---")
            cmd = [sys.executable, "train_model.py", "--data", str(data_path),
                   "--interval", interval, "--asset", asset_key]
            if args.fast:
                cmd.append("--fast")
            result = subprocess.run(cmd)
            (trained if result.returncode == 0 else failed).append(f"{asset_key}/{interval}")
            print()

    print(f"Trained: {trained or 'none'}")
    if skipped:
        print(f"Skipped (no data): {skipped}")
    if failed:
        print(f"Failed: {failed}")
