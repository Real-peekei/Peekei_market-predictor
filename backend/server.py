"""
server.py
Live API for the predictor - works for any asset you've trained models
for (gold, another commodity, a stock, forex, crypto, whatever), not just
gold. Run train_model.py / train_all.py first for at least one asset.

Run:
    uvicorn server:app --reload --port 8001

(Using 8001 so it can run alongside the football predictor's 8000 if you want.)
"""

import json
import re
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import predict_next
from trust import trust_assessment, load_backtest
from intervals import sort_intervals

MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(title="Market Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# matches e.g. "GC_F_clf_1d.pkl" -> asset="GC_F", interval="1d"
STATE_FILE_RE = re.compile(r"^(?P<asset>.+)_state_(?P<interval>[A-Za-z0-9]+)\.pkl$")


def available_assets() -> dict[str, dict]:
    """Scans models/ for trained assets, returning {asset_key: {symbol, intervals}}."""
    assets: dict[str, dict] = {}
    for p in MODELS_DIR.glob("*_state_*.pkl"):
        m = STATE_FILE_RE.match(p.name)
        if not m:
            continue
        asset_key, interval = m.group("asset"), m.group("interval")
        if asset_key not in assets:
            try:
                state = joblib.load(p)
                symbol = state.get("symbol", asset_key)
            except Exception:
                symbol = asset_key
            assets[asset_key] = {"symbol": symbol, "intervals": []}
        assets[asset_key]["intervals"].append(interval)

    for asset_key in assets:
        assets[asset_key]["intervals"] = sort_intervals(assets[asset_key]["intervals"])
    return assets


def available_intervals(asset_key: str) -> list[str]:
    prefix = f"{asset_key}_clf_"
    found = {p.stem.replace(prefix, "", 1) for p in MODELS_DIR.glob(f"{prefix}*.pkl")}
    return sort_intervals(list(found))


@app.get("/")
def root():
    return {"status": "ok", "assets": list(available_assets().keys())}


@app.get("/api/assets")
def get_assets():
    """Every asset with at least one trained timeframe, with a friendly
    symbol label and its available timeframes in chronological order."""
    assets = available_assets()
    if not assets:
        raise HTTPException(503, "No trained models found - run train_model.py or train_all.py first")
    return {
        "assets": [
            {"asset_key": key, "symbol": info["symbol"], "intervals": info["intervals"]}
            for key, info in sorted(assets.items())
        ]
    }


@app.get("/api/intervals")
def get_intervals(asset: str):
    intervals = available_intervals(asset)
    if not intervals:
        raise HTTPException(404, f"No trained models for asset='{asset}'. "
                                  f"Check /api/assets for what's available.")
    return {"asset": asset, "intervals": intervals}


def _predict_one(asset_key: str, interval: str) -> dict:
    prefix = f"{asset_key}_"
    paths = {
        "clf": MODELS_DIR / f"{prefix}clf_{interval}.pkl",
        "reg_low": MODELS_DIR / f"{prefix}reg_low_{interval}.pkl",
        "reg_high": MODELS_DIR / f"{prefix}reg_high_{interval}.pkl",
        "state": MODELS_DIR / f"{prefix}state_{interval}.pkl",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"No trained model for asset='{asset_key}' interval='{interval}'. "
                                     f"Run: python train_model.py --asset {asset_key} --interval {interval} --data ...")

    clf = joblib.load(paths["clf"])
    reg_low = joblib.load(paths["reg_low"])
    reg_high = joblib.load(paths["reg_high"])
    state = joblib.load(paths["state"])

    p_up, ret_low, ret_high = predict_next(clf, reg_low, reg_high, state["latest_row"])
    last_close = state["last_close"]

    backtest = load_backtest(asset_key, interval)
    trust = trust_assessment(backtest["metrics"] if backtest else None)

    return {
        "asset": asset_key,
        "symbol": state.get("symbol", asset_key),
        "interval": interval,
        "as_of": state["last_date"],
        "last_close": last_close,
        "prob_up": round(p_up, 4),
        "prob_down": round(1 - p_up, 4),
        "direction": "up" if p_up > 0.5 else "down",
        "confidence": round(abs(p_up - 0.5) * 2, 4),  # 0 = coin flip, 1 = maximally confident
        "expected_range": {
            "low_price": round(last_close * (1 + ret_low), 2),
            "high_price": round(last_close * (1 + ret_high), 2),
            "low_return": round(ret_low, 5),
            "high_return": round(ret_high, 5),
        },
        "trust": trust,
    }


@app.get("/api/predict")
def predict(asset: str, interval: str):
    try:
        return _predict_one(asset, interval)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/api/predict-all")
def predict_all(asset: str):
    """Runs a prediction for every trained timeframe of one asset at once,
    returned in ascending chronological order (shortest timeframe first)."""
    intervals = available_intervals(asset)
    if not intervals:
        raise HTTPException(404, f"No trained models for asset='{asset}'. Check /api/assets.")

    predictions, errors = [], []
    for interval in intervals:  # already chronologically sorted
        try:
            predictions.append(_predict_one(asset, interval))
        except Exception as e:
            # any single bad/corrupt/incompatible model shouldn't take down
            # the whole batch - skip it and report why, keep going
            errors.append({"interval": interval, "error": f"{type(e).__name__}: {e}"})

    return {"asset": asset, "predictions": predictions, "errors": errors}


@app.get("/api/backtest")
def backtest(asset: str, interval: str):
    data = load_backtest(asset, interval)
    if data is None:
        raise HTTPException(404, f"No backtest saved for asset='{asset}' interval='{interval}'.")
    return data
