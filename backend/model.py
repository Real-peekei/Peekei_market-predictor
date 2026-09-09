"""
model.py
Two models per timeframe:
  1. GradientBoostingClassifier -> P(next bar closes up)
  2. Two GradientBoostingRegressors with quantile loss -> 20th/80th
     percentile of next-bar return, giving an expected price RANGE
     rather than a single point guess (which would be false precision).

Also implements a walk-forward backtest: instead of one lucky train/test
split, the model is periodically retrained using only data available up
to that point in time, and evaluated bar-by-bar against what actually
happened next. This is slower but it's the only honest way to estimate
how a model like this would have performed if traded live.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from indicators import FEATURE_COLS

# If you re-enable n_jobs on any estimator below, note that joblib's
# parallel workers run in separate processes and won't inherit this
# warnings filter - the harmless RandomForest parallel warning would
# reappear even with this filter in place. Single-threaded training is
# fast enough here given the capped training window, so we avoid it.
warnings.filterwarnings(
    "ignore",
    message=".*should be used with.*Parallel.*to make it possible to propagate.*",
    category=UserWarning,
)

# approx bars/year per interval, used to annualize Sharpe ratio
PERIODS_PER_YEAR = {
    "1m": 252 * 24 * 60, "2m": 252 * 24 * 30, "5m": 252 * 24 * 12, "15m": 252 * 24 * 4,
    "30m": 252 * 24 * 2, "45m": int(252 * 24 * 60 / 45), "90m": int(252 * 24 * 60 / 90),
    "60m": 252 * 24, "1h": 252 * 24, "2h": 252 * 12, "4h": 252 * 6, "8h": 252 * 3,
    "1d": 252, "1wk": 52, "1mo": 12,
}

CANDIDATE_MODELS = {
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.8,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42,
    ),
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=1.0),
    ),
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURE_COLS + ["target_return", "target_direction"]).reset_index(drop=True)


def compare_candidates(df: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """
    Trains each candidate direction-classifier on the first (1-test_frac)
    of the data (chronological, no shuffling) and scores it on the rest.
    Returns the comparison table and the name of the best model by log-loss.

    NOTE: with a single hold-out split, a small log-loss gap between two
    candidates is noise, not a real edge - the printed gap-to-runner-up
    tells you how seriously to take "best".
    """
    data = _clean(df)
    n = len(data)
    split = int(n * (1 - test_frac))
    X_train, X_test = data[FEATURE_COLS].iloc[:split], data[FEATURE_COLS].iloc[split:]
    y_train, y_test = data["target_direction"].iloc[:split], data["target_direction"].iloc[split:]

    results = []
    for name, make_model in CANDIDATE_MODELS.items():
        model = make_model()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs, labels=[0, 1])
        results.append({"name": name, "accuracy": float(acc), "log_loss": float(ll)})

    best = min(results, key=lambda r: r["log_loss"])
    sorted_ll = sorted(r["log_loss"] for r in results)
    gap = sorted_ll[1] - sorted_ll[0] if len(sorted_ll) > 1 else None

    return {"results": results, "best": best["name"], "gap_to_runner_up": gap}


def train_full(df: pd.DataFrame, model_name: str = "gradient_boosting"):
    """Train on ALL available data using the given model type - this is
    what's actually used for live /predict calls (no future data to hold
    out at inference time)."""
    data = _clean(df)
    X = data[FEATURE_COLS]

    clf = CANDIDATE_MODELS[model_name]()
    clf.fit(X, data["target_direction"])

    reg_low = GradientBoostingRegressor(loss="quantile", alpha=0.2, n_estimators=150, max_depth=3, learning_rate=0.05)
    reg_high = GradientBoostingRegressor(loss="quantile", alpha=0.8, n_estimators=150, max_depth=3, learning_rate=0.05)
    reg_low.fit(X, data["target_return"])
    reg_high.fit(X, data["target_return"])

    return clf, reg_low, reg_high


def predict_next(clf, reg_low, reg_high, latest_row: pd.DataFrame):
    X = latest_row[FEATURE_COLS]
    p_up = float(clf.predict_proba(X)[0][1])
    ret_low = float(reg_low.predict(X)[0])
    ret_high = float(reg_high.predict(X)[0])
    return p_up, ret_low, ret_high


def walk_forward_backtest(df: pd.DataFrame, interval: str, model_name: str = "gradient_boosting",
                           min_train_bars: int = 250, retrain_every: int = 20,
                           max_train_window: int | None = None, max_test_bars: int | None = None,
                           verbose: bool = True):
    """
    Expanding-window (or sliding-window, if max_train_window is set)
    walk-forward test:
      - Train on bars [0 : i]  (or [i-max_train_window : i] if capped)
      - Predict bar i+1's direction
      - Go long (hold the asset) if predicted P(up) > 0.5, else stay in cash
      - Compare cumulative strategy return to plain buy-and-hold over the
        same stretch
      - Retrain every `retrain_every` bars (not every single bar) to keep
        runtime reasonable

    Performance knobs (matter a lot on large intraday datasets like 1h/730d,
    which can be 10-15x more bars than 15m/60d):
      - max_train_window: caps how much history each retrain uses. Without
        this, training sets grow for the whole backtest (expanding window),
        so cost grows roughly with n^2. With it capped (e.g. 2000 bars),
        each retrain costs roughly the same regardless of how much total
        history there is - the backtest scales linearly instead.
      - max_test_bars: only walk forward through the most recent N bars
        instead of the entire history. A recent 2000-3000 bar window is
        usually more relevant anyway (market regimes shift), and it directly
        caps the number of retrain cycles.
    """
    data = _clean(df)
    n = len(data)
    if n < min_train_bars + 20:
        raise ValueError(f"Not enough data for a walk-forward backtest (have {n} usable bars, need >= {min_train_bars + 20}).")

    X_all = data[FEATURE_COLS].values
    y_dir_all = data["target_direction"].values
    actual_ret_all = data["target_return"].values  # next bar's realized return

    start_i = min_train_bars
    if max_test_bars is not None:
        start_i = max(min_train_bars, n - 1 - max_test_bars)

    total_steps = (n - 1) - start_i
    expected_retrains = max(1, total_steps // retrain_every)
    if verbose:
        print(f"  walk-forward: {total_steps} bars to test, ~{expected_retrains} retrains "
              f"(train window: {'last ' + str(max_train_window) + ' bars' if max_train_window else 'all history (expanding)'})")

    dates, strat_returns, buyhold_returns, predicted_probs = [], [], [], []

    clf = None
    for step, i in enumerate(range(start_i, n - 1)):
        if clf is None or (i - start_i) % retrain_every == 0:
            train_start = max(0, i - max_train_window) if max_train_window else 0
            clf = CANDIDATE_MODELS[model_name]()
            clf.fit(X_all[train_start:i], y_dir_all[train_start:i])
            if verbose and (i - start_i) % (retrain_every * 20) == 0 and step > 0:
                print(f"    ...{step}/{total_steps} bars tested "
                      f"({step / total_steps:.0%})")

        p_up = clf.predict_proba(X_all[i:i + 1])[0][1]
        actual_ret = actual_ret_all[i]

        position = 1 if p_up > 0.5 else 0
        strat_returns.append(position * actual_ret)
        buyhold_returns.append(actual_ret)
        predicted_probs.append(p_up)
        dates.append(data.iloc[i]["Date"])

    strat_returns = np.array(strat_returns)
    buyhold_returns = np.array(buyhold_returns)
    predicted_probs = np.array(predicted_probs)

    strat_equity = (1 + strat_returns).cumprod()
    buyhold_equity = (1 + buyhold_returns).cumprod()

    periods_per_year = PERIODS_PER_YEAR.get(interval, 252)

    def sharpe(returns):
        if returns.std() == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())

    def max_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min())

    predicted_dir = (predicted_probs > 0.5).astype(int)
    actual_dir = (buyhold_returns > 0).astype(int)
    hit_rate = float((predicted_dir == actual_dir).mean())

    return {
        "dates": [str(d) for d in dates],
        "strategy_equity": strat_equity.tolist(),
        "buyhold_equity": buyhold_equity.tolist(),
        "metrics": {
            "strategy_total_return": round(float(strat_equity[-1] - 1), 4),
            "buyhold_total_return": round(float(buyhold_equity[-1] - 1), 4),
            "strategy_sharpe": round(sharpe(strat_returns), 3),
            "buyhold_sharpe": round(sharpe(buyhold_returns), 3),
            "strategy_max_drawdown": round(max_drawdown(strat_equity), 4),
            "buyhold_max_drawdown": round(max_drawdown(buyhold_equity), 4),
            "direction_hit_rate": round(hit_rate, 4),
            "n_bars_tested": int(len(strat_returns)),
        },
    }
