"""
predict_all.py
Prints a live prediction for every trained timeframe of one asset at once,
in chronological order (shortest timeframe first), with a plain-language
trust read for each based on that timeframe's own walk-forward backtest.

Usage:
    python predict_all.py                  # prints available assets if you don't pick one
    python predict_all.py --asset GC_F
    python predict_all.py --asset AAPL
"""

import argparse
import re
from pathlib import Path

import joblib

from model import predict_next
from trust import trust_assessment, load_backtest  # same logic the API uses
from intervals import sort_intervals

MODELS_DIR = Path(__file__).parent / "models"
STATE_FILE_RE = re.compile(r"^(?P<asset>.+)_state_(?P<interval>[A-Za-z0-9]+)\.pkl$")


def discover_assets() -> set[str]:
    assets = set()
    for p in MODELS_DIR.glob("*_state_*.pkl"):
        m = STATE_FILE_RE.match(p.name)
        if m:
            assets.add(m.group("asset"))
    return assets


def available_intervals(asset_key: str) -> list[str]:
    prefix = f"{asset_key}_clf_"
    found = {p.stem.replace(prefix, "", 1) for p in MODELS_DIR.glob(f"{prefix}*.pkl")}
    return sort_intervals(list(found))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="Asset key to predict, e.g. GC_F, AAPL")
    args = parser.parse_args()

    assets = discover_assets()
    if not assets:
        print("No trained models found. Run train_model.py (or train_all.py) first.")
        raise SystemExit(1)

    if args.asset is None:
        print(f"No --asset given. Assets with trained models: {sorted(assets)}")
        print(f"Example: python predict_all.py --asset {sorted(assets)[0]}")
        raise SystemExit(0)

    if args.asset not in assets:
        print(f"No trained models for asset='{args.asset}'. Available: {sorted(assets)}")
        raise SystemExit(1)

    intervals = available_intervals(args.asset)
    prefix = f"{args.asset}_"

    print(f"Asset: {args.asset}\n")
    print(f"{'Interval':10s} {'Direction':10s} {'Conf':>6s} {'Range':>24s}  Trust")
    print("-" * 95)

    for interval in intervals:  # already chronologically sorted
        clf = joblib.load(MODELS_DIR / f"{prefix}clf_{interval}.pkl")
        reg_low = joblib.load(MODELS_DIR / f"{prefix}reg_low_{interval}.pkl")
        reg_high = joblib.load(MODELS_DIR / f"{prefix}reg_high_{interval}.pkl")
        state = joblib.load(MODELS_DIR / f"{prefix}state_{interval}.pkl")

        p_up, ret_low, ret_high = predict_next(clf, reg_low, reg_high, state["latest_row"])
        last_close = state["last_close"]
        low_price = last_close * (1 + ret_low)
        high_price = last_close * (1 + ret_high)
        direction = "UP" if p_up > 0.5 else "DOWN"
        confidence = abs(p_up - 0.5) * 2

        bt = load_backtest(args.asset, interval)
        trust = trust_assessment(bt["metrics"] if bt else None)

        range_str = f"{low_price:,.2f} - {high_price:,.2f}"
        print(f"{interval:10s} {direction:10s} {confidence:>5.0%}  {range_str:>24s}  [{trust['level']}] {trust['headline']}")
