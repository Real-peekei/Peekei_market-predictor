"""
twelvedata_fetch.py
Downloads price history from Twelve Data at any of its 12 supported
intervals, for ANY symbol Twelve Data supports - stocks, forex, crypto,
indices, and commodities (spot gold, silver, oil, etc. - see their docs
for the full catalog). Not limited to gold.

IMPORTANT - about futures specifically:
Twelve Data's dedicated Commodities product (which XAU/USD belongs to) is
explicitly SPOT data only - there is no documented "Gold Futures" product
alongside it, and spot has no real trading volume (see indicators.py for
how that's handled). Twelve Data's marketing does list "futures" as a
broader supported asset class, but no stable, documented symbol format for
COMEX gold futures (e.g. continuous GC) is published for the time_series
endpoint - guessing one risks silently fetching the wrong instrument or a
symbol that doesn't exist. Use twelvedata_symbol_search.py first to look
up the exact, correct symbol Twelve Data recognizes before trusting any
--symbol you pass here for futures. It may also require a paid plan tier -
the free Basic plan's asset coverage should be checked at twelvedata.com/pricing.

Setup:
    1. Sign up free at https://twelvedata.com (no credit card required)
    2. Copy your API key from the dashboard
    3. Set it as an environment variable so it's never hardcoded/committed:
         Windows (PowerShell):  $env:TWELVEDATA_API_KEY = "your_key_here"
         Windows (cmd):         set TWELVEDATA_API_KEY=your_key_here
         Mac/Linux:              export TWELVEDATA_API_KEY="your_key_here"
       Or pass it directly with --api-key each time (less convenient, but works).

Free tier limits: 8 requests/minute, 800/day (resets midnight UTC). Fetching
every timeframe for one symbol is nowhere near this - each fetch is a single
request. This script backs off and retries automatically if you do hit the
per-minute cap (HTTP 429).

Output files are automatically namespaced by symbol, e.g. --symbol XAU/USD
--interval 1d writes data/XAU_USD_1d.csv, so multiple products' data can
coexist without colliding.

Usage:
    python twelvedata_fetch.py --symbol XAU/USD --interval 1d  --period 10y
    python twelvedata_fetch.py --symbol AAPL    --interval 1h  --period 730d
    python twelvedata_fetch.py --symbol EUR/USD --interval 15m --period 60d
    python twelvedata_fetch.py --symbol "GC1!" --interval 4h --period 730d
        (only after confirming "GC1!" - or whatever symbol - is correct via
        twelvedata_symbol_search.py; this is illustrative, not a verified default)
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from assets import slugify_symbol

API_URL = "https://api.twelvedata.com/time_series"
DEFAULT_SYMBOL = "XAU/USD"  # example default (spot gold) - pass any --symbol you like

# All 12 intervals Twelve Data's time_series endpoint supports, mapped from
# our project's interval codes to Twelve Data's own naming.
INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "45m": "45min",
    "60m": "1h", "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h",
    "1d": "1day", "1wk": "1week", "1mo": "1month",
}

# every interval EXCEPT "60m" (which is an alias for "1h" - same data)
ALL_TWELVEDATA_INTERVALS = ["1m", "5m", "15m", "30m", "45m", "1h", "2h", "4h", "8h", "1d", "1wk", "1mo"]

# sensible default lookback per interval. Twelve Data's own intraday history
# generally spans a couple of years for most symbols (see their docs), a bit
# more generous than Yahoo's yfinance limits - these are reasonable starting
# points, not hard API limits, so feel free to override with --period.
DEFAULT_PERIOD = {
    "1m": "60d", "5m": "365d", "15m": "365d", "30m": "365d", "45m": "365d",
    "60m": "730d", "1h": "730d", "2h": "730d", "4h": "730d", "8h": "730d",
    "1d": "10y", "1wk": "max", "1mo": "max",
}

MAX_OUTPUTSIZE = 5000  # Twelve Data's per-request cap


def _period_to_start_date(period: str) -> str | None:
    """Converts '730d' / '10y' / '60d' / 'max' into a start_date string."""
    if period == "max":
        return "1990-01-01"  # effectively "give me everything you have"
    now = datetime.now(timezone.utc)
    if period.endswith("d"):
        delta = timedelta(days=int(period[:-1]))
    elif period.endswith("y"):
        delta = timedelta(days=int(period[:-1]) * 365)
    else:
        raise ValueError(f"Unrecognized period format: '{period}' (expected e.g. '60d', '10y', 'max')")
    return (now - delta).strftime("%Y-%m-%d")


def fetch(symbol: str, interval: str, period: str, api_key: str, max_retries: int = 5) -> pd.DataFrame:
    td_interval = INTERVAL_MAP.get(interval)
    if td_interval is None:
        raise ValueError(f"Interval '{interval}' has no Twelve Data mapping - check INTERVAL_MAP.")

    start_date = _period_to_start_date(period)
    params = {
        "symbol": symbol,
        "interval": td_interval,
        "start_date": start_date,
        "outputsize": MAX_OUTPUTSIZE,
        "apikey": api_key,
        "order": "ASC",
    }

    for attempt in range(1, max_retries + 1):
        resp = requests.get(API_URL, params=params, timeout=30)
        data = resp.json()

        if data.get("status") == "error":
            code = data.get("code")
            message = data.get("message", "unknown error")
            if code == 429:
                wait = 15
                print(f"  rate limited (free tier: 8/min) - waiting {wait}s and retrying "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Twelve Data error ({code}): {message}. "
                f"Check your API key and that '{symbol}' is a valid Twelve Data symbol."
            )

        values = data.get("values")
        if not values:
            raise RuntimeError(
                f"No data returned for {symbol} @ {interval}/{period}. "
                f"Check the symbol name (try 'XAU/USD') and your API key."
            )

        df = pd.DataFrame(values)
        df = df.rename(columns={"datetime": "Date", "open": "Open", "high": "High",
                                 "low": "Low", "close": "Close", "volume": "Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        # Twelve Data's spot commodity/forex feeds often don't report volume
        # (spot gold has no single central exchange volume figure) - the
        # "volume" key may be missing from the response entirely, or present
        # with null values. Handle both: fill with 0 either way. This makes
        # the model's volume-based feature (vol_ratio) uninformative for
        # spot data, which is a real but minor limitation - one feature out
        # of ~20.
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        else:
            df["Volume"] = 0.0
        df = df.sort_values("Date").reset_index(drop=True)

        if len(df) >= MAX_OUTPUTSIZE:
            print(f"  NOTE: hit the {MAX_OUTPUTSIZE}-bar per-request cap - your actual history "
                  f"may be longer than what was returned. Consider a shorter --period for "
                  f"intraday intervals if you need the full requested range.")

        return df

    raise RuntimeError(f"Gave up after {max_retries} attempts due to repeated rate limiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Twelve Data symbol, e.g. XAU/USD")
    parser.add_argument("--interval", default="1d",
                         choices=["1m", "5m", "15m", "30m", "45m", "60m", "1h", "2h", "4h", "8h", "1d", "1wk", "1mo"])
    parser.add_argument("--period", default=None, help="e.g. 5y, 730d, 60d, max - defaults per interval if omitted")
    parser.add_argument("--out", default=None)
    parser.add_argument("--api-key", default=None,
                         help="Twelve Data API key. Defaults to the TWELVEDATA_API_KEY env var if not passed.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set the TWELVEDATA_API_KEY environment variable, "
            "or pass --api-key directly. Sign up free at https://twelvedata.com"
        )

    period = args.period or DEFAULT_PERIOD.get(args.interval, "5y")
    asset_key = slugify_symbol(args.symbol)
    out_path = Path(args.out or f"data/{asset_key}_{args.interval}.csv")

    print(f"Fetching {args.symbol} @ interval={args.interval} period={period} (Twelve Data)...")
    df = fetch(args.symbol, args.interval, period, api_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # record the original symbol so downstream steps (training, the API)
    # can show a friendly label instead of just the asset key
    meta_path = out_path.parent / f"{asset_key}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"symbol": args.symbol, "source": "twelvedata"}, f)

    print(f"Saved {len(df)} bars -> {out_path}")
