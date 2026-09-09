"""
generate_sample_data.py
Creates a synthetic OHLCV series (geometric Brownian motion with mild
drift + volatility clustering, loosely gold-like in price scale) purely so
the pipeline can be tested before you point it at real data via
data_fetch.py / twelvedata_fetch.py, for any asset. NOT for real trading
decisions - it has no real market structure, just plausible-looking noise.

Writes to data/SAMPLE_1d.csv, following the same {ASSET}_{interval}.csv
naming convention as the real fetch scripts, so you can test the full
pipeline (train_model.py, train_all.py, the API, the frontend) with
--asset SAMPLE before ever touching a real data source.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)

N = 1500  # trading days (~6 years)
start_price = 1800.0
mu = 0.0002        # slight upward drift, roughly gold's long-run behavior
base_vol = 0.008

# volatility clustering: vol itself follows a slow random walk
vol = np.zeros(N)
vol[0] = base_vol
for i in range(1, N):
    vol[i] = np.clip(vol[i - 1] + rng.normal(0, 0.0006), 0.004, 0.02)

returns = rng.normal(mu, 1.0, N) * vol
close = start_price * np.cumprod(1 + returns)

open_ = np.empty(N)
high = np.empty(N)
low = np.empty(N)
open_[0] = start_price
for i in range(N):
    prev_close = close[i - 1] if i > 0 else start_price
    open_[i] = prev_close * (1 + rng.normal(0, vol[i] * 0.3))
    intraday_range = abs(rng.normal(0, vol[i])) * close[i]
    high[i] = max(open_[i], close[i]) + intraday_range * rng.uniform(0.1, 0.6)
    low[i] = min(open_[i], close[i]) - intraday_range * rng.uniform(0.1, 0.6)

volume = rng.integers(80_000, 250_000, N)
dates = pd.bdate_range("2019-01-02", periods=N)

df = pd.DataFrame({
    "Date": dates, "Open": open_, "High": high, "Low": low,
    "Close": close, "Volume": volume,
})

out = Path(__file__).parent / "data" / "SAMPLE_1d.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

meta_path = out.parent / "SAMPLE_meta.json"
with open(meta_path, "w") as f:
    json.dump({"symbol": "SAMPLE (synthetic test data)", "source": "generate_sample_data.py"}, f)

print(f"Wrote {len(df)} synthetic daily bars -> {out}")
print(f"Test the pipeline with: python train_model.py --data {out} --interval 1d --fast")
