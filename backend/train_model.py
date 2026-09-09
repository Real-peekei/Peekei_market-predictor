"""
train_model.py
Builds features, trains the live model, runs the walk-forward backtest,
and saves everything for a given asset + timeframe. Works for any tradable
product - gold, another commodity, a stock, forex, crypto, whatever you
fetched with data_fetch.py / twelvedata_fetch.py.

Usage:
    python train_model.py --data data/GC_F_1d.csv --interval 1d
    python train_model.py --data data/AAPL_1h.csv --interval 1h

The asset key (e.g. "GC_F", "AAPL") is normally inferred automatically from
the data filename - data_fetch.py and twelvedata_fetch.py always name files
as {ASSET}_{interval}.csv, so this "just works" if you used those. Pass
--asset explicitly to override (e.g. if you renamed the file).

Speed: the walk-forward backtest is the slow part (it retrains many times
through history). Defaults below cap it to a bounded amount of work
regardless of how much history you feed in, so it shouldn't blow up on any
timeframe. Use --fast for a quicker, rougher pass, or tune the individual
flags below directly.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from indicators import build_features
from model import train_full, walk_forward_backtest, compare_candidates

MODELS_DIR = Path(__file__).parent / "models"


def infer_asset_key(data_path: str, interval: str) -> str:
    """
    data_fetch.py / twelvedata_fetch.py always name files {ASSET}_{interval}.csv,
    so stripping the known interval suffix from the filename stem recovers
    the asset key unambiguously - robust regardless of what the asset key
    itself looks like, since we know the exact interval string to remove.
    """
    stem = Path(data_path).stem
    suffix = f"_{interval}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    # nonstandard filename - use the whole stem and let the user know
    print(f"  NOTE: '{data_path}' doesn't match the expected {{ASSET}}_{interval}.csv naming - "
          f"using '{stem}' as the asset key. Pass --asset to override.")
    return stem


def load_symbol_label(asset_key: str, data_path: str) -> str:
    """Looks for a {ASSET}_meta.json next to the data file (written by the
    fetch scripts) to get the original human-readable symbol. Falls back to
    the asset key itself if no meta file exists."""
    meta_path = Path(data_path).parent / f"{asset_key}_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f).get("symbol", asset_key)
        except (json.JSONDecodeError, OSError):
            pass
    return asset_key


def run(data_path: str, interval: str, asset_key: str, retrain_every: int,
        max_train_window: int | None, max_test_bars: int | None):
    symbol_label = load_symbol_label(asset_key, data_path)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    print(f"Loaded {len(df)} bars from {data_path}  (asset: {asset_key}, symbol: {symbol_label})")

    feat_df = build_features(df)

    print("Comparing candidate direction models on chronological hold-out...")
    comparison = compare_candidates(feat_df)
    for r in comparison["results"]:
        print(f"  {r['name']:20s} accuracy: {r['accuracy']:.3f}   log-loss: {r['log_loss']:.3f}  (lower is better)")
    model_name = comparison["best"]
    print(f"  -> best model: {model_name}")
    if comparison["gap_to_runner_up"] is not None and comparison["gap_to_runner_up"] < 0.02:
        print(f"  NOTE: gap to next-best is only {comparison['gap_to_runner_up']:.3f} log-loss - treat as a near tie.")

    print(f"Training live model on full history using '{model_name}'...")
    clf, reg_low, reg_high = train_full(feat_df, model_name=model_name)

    print("Running walk-forward backtest (this is the honest number)...")
    t0 = time.time()
    try:
        backtest = walk_forward_backtest(
            feat_df, interval=interval, model_name=model_name,
            retrain_every=retrain_every, max_train_window=max_train_window,
            max_test_bars=max_test_bars,
        )
        elapsed = time.time() - t0
        m = backtest["metrics"]
        print(f"  done in {elapsed:.1f}s")
        print(f"  strategy total return: {m['strategy_total_return']:.2%}  "
              f"vs buy&hold: {m['buyhold_total_return']:.2%}")
        print(f"  strategy Sharpe: {m['strategy_sharpe']}  vs buy&hold Sharpe: {m['buyhold_sharpe']}")
        print(f"  direction hit rate: {m['direction_hit_rate']:.1%}  "
              f"(50% = coin flip, so compare against that, not against 100%)")
        print(f"  max drawdown: {m['strategy_max_drawdown']:.2%}  "
              f"vs buy&hold: {m['buyhold_max_drawdown']:.2%}")
    except ValueError as e:
        print(f"  Skipped backtest: {e}")
        backtest = None

    MODELS_DIR.mkdir(exist_ok=True)
    prefix = f"{asset_key}_"
    joblib.dump(clf, MODELS_DIR / f"{prefix}clf_{interval}.pkl")
    joblib.dump(reg_low, MODELS_DIR / f"{prefix}reg_low_{interval}.pkl")
    joblib.dump(reg_high, MODELS_DIR / f"{prefix}reg_high_{interval}.pkl")

    latest_row = feat_df.dropna(subset=["ret_1"]).iloc[[-1]]
    joblib.dump({"latest_row": latest_row, "last_close": float(df["Close"].iloc[-1]),
                 "last_date": str(df["Date"].iloc[-1]), "symbol": symbol_label, "asset_key": asset_key},
                MODELS_DIR / f"{prefix}state_{interval}.pkl")

    if backtest:
        with open(MODELS_DIR / f"{prefix}backtest_{interval}.json", "w") as f:
            json.dump(backtest, f)

    with open(MODELS_DIR / f"{prefix}model_selection_{interval}.json", "w") as f:
        json.dump({"chosen_model": model_name, "candidates": comparison["results"]}, f)

    print(f"Saved models for asset={asset_key} interval={interval} -> {MODELS_DIR}/  (direction model = {model_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--interval", required=True,
                         choices=["1m", "2m", "5m", "15m", "30m", "45m", "60m", "90m", "1h", "2h", "4h", "8h", "1d", "1wk", "1mo"])
    parser.add_argument("--asset", default=None,
                         help="Asset key for namespacing model files. Auto-inferred from --data filename if omitted.")
    parser.add_argument("--retrain-every", type=int, default=40,
                         help="retrain the backtest model every N bars (default: 40; higher = faster, less precise)")
    parser.add_argument("--max-train-window", type=int, default=1500,
                         help="cap each retrain's training set to the most recent N bars (default: 1500; "
                              "use 0 to disable and use full expanding history - much slower on long histories)")
    parser.add_argument("--max-test-bars", type=int, default=2500,
                         help="only walk forward through the most recent N bars (default: 2500; "
                              "use 0 to backtest the full history - much slower)")
    parser.add_argument("--fast", action="store_true",
                         help="quick rough pass: retrain-every=100, max-train-window=800, max-test-bars=1000")
    args = parser.parse_args()

    if args.fast:
        retrain_every, max_train_window, max_test_bars = 100, 800, 1000
    else:
        retrain_every = args.retrain_every
        max_train_window = args.max_train_window or None
        max_test_bars = args.max_test_bars or None

    asset_key = args.asset or infer_asset_key(args.data, args.interval)

    run(args.data, args.interval, asset_key, retrain_every, max_train_window, max_test_bars)
