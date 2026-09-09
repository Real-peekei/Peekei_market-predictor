"""
indicators.py
Technical indicator feature engineering for the gold model. Implemented
from scratch on pandas/numpy (no ta-lib / pandas-ta dependency) so it runs
anywhere without extra native libraries.

All indicators are computed using ONLY past bars (rolling windows), so
there's no lookahead leakage - each row's features are known at that bar's
close, before the *next* bar's move is predicted.
"""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _atr(high, low, close, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns Date, Open, High, Low, Close, Volume
    Returns df with feature columns appended, plus the forward-looking
    target columns (next-bar direction and next-bar return) used for training.
    """
    out = df.sort_values("Date").reset_index(drop=True).copy()
    close, high, low, vol = out["Close"], out["High"], out["Low"], out["Volume"]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    for w in (5, 10, 20, 50):
        sma = close.rolling(w).mean()
        out[f"sma_{w}_dist"] = (close - sma) / sma
    for w in (12, 26):
        out[f"ema_{w}_dist"] = (close - close.ewm(span=w, adjust=False).mean()) / close

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / close
    out["macd_hist"] = (macd - signal) / close

    out["rsi_14"] = _rsi(close, 14)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_pct"] = (close - (sma20 - 2 * std20)) / ((sma20 + 2 * std20) - (sma20 - 2 * std20))
    out["bb_width"] = (4 * std20) / sma20

    out["atr_14"] = _atr(high, low, close, 14) / close
    out["volatility_10"] = out["ret_1"].rolling(10).std()
    out["volatility_20"] = out["ret_1"].rolling(20).std()

    vol_sma = vol.rolling(20).mean()
    if vol.abs().sum() == 0:
        # No real volume data available (e.g. spot gold via Twelve Data,
        # which has no single central exchange to report volume from).
        # A naive vol/vol_sma here would be 0/0 = NaN for EVERY row, which
        # then wipes out the entire dataset once rows with any missing
        # feature get dropped downstream. Use a neutral constant instead -
        # this correctly makes the feature uninformative (no real volume
        # signal exists) without destroying the rest of the data.
        out["vol_ratio"] = 1.0
    else:
        out["vol_ratio"] = vol / vol_sma.replace(0, np.nan)

    out["high_low_range"] = (high - low) / close
    out["close_position"] = (close - low) / (high - low).replace(0, np.nan)

    # ---- forward-looking targets (what we're trying to predict) ----
    out["target_return"] = close.pct_change().shift(-1)          # next bar's % return
    out["target_direction"] = (out["target_return"] > 0).astype(int)  # 1 = up, 0 = down/flat

    return out


FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "sma_5_dist", "sma_10_dist", "sma_20_dist", "sma_50_dist",
    "ema_12_dist", "ema_26_dist",
    "macd", "macd_hist", "rsi_14",
    "bb_pct", "bb_width",
    "atr_14", "volatility_10", "volatility_20",
    "vol_ratio", "high_low_range", "close_position",
]
