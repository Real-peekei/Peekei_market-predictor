"""
data_fetch.py
Downloads price history via yfinance, at whatever timeframe you ask for,
for ANY symbol yfinance supports - stocks (AAPL), commodities/futures
(GC=F gold, CL=F crude oil), forex (EURUSD=X), crypto (BTC-USD), indices,
ETFs, and more. Not limited to gold - --symbol accepts anything yfinance
recognizes.

Yahoo Finance limits how far back intraday data goes:
    1m                 -> last 7 days only
    2m/5m/15m/30m/90m  -> last 60 days only
    60m / 1h           -> last 730 days
    1d / 1wk / 1mo     -> full history

Usage:
    python data_fetch.py --symbol GC=F     --interval 1d  --period 5y   # gold futures
    python data_fetch.py --symbol AAPL     --interval 1h  --period 730d # a stock
    python data_fetch.py --symbol BTC-USD  --interval 15m --period 60d  # crypto
    python data_fetch.py --symbol EURUSD=X --interval 1d  --period 10y  # forex

Output files are automatically namespaced by symbol, e.g. --symbol GC=F
--interval 1d writes data/GC_F_1d.csv, so multiple products' data can
coexist without colliding. Override with --out if you want a specific path.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from assets import slugify_symbol

DEFAULT_TICKER = "GC=F"  # example default (COMEX gold futures) - pass any --symbol you like

# sensible default lookback per interval, used if --period isn't given
DEFAULT_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "90m": "60d",
    "60m": "730d", "1h": "730d",
    "1d": "10y", "1wk": "max", "1mo": "max",
}


def fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} @ {interval}/{period}. "
            f"Check the symbol/interval, or that intraday history isn't outside Yahoo's lookback limit."
        )

    # yfinance sometimes returns MultiIndex columns for a single ticker - flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={date_col: "Date"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", "--ticker", dest="symbol", default=DEFAULT_TICKER,
                         help="Any yfinance symbol, e.g. GC=F, AAPL, BTC-USD, EURUSD=X")
    parser.add_argument("--interval", default="1d",
                         choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"])
    parser.add_argument("--period", default=None, help="e.g. 5y, 730d, 60d - defaults per interval if omitted")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    period = args.period or DEFAULT_PERIOD.get(args.interval, "5y")
    asset_key = slugify_symbol(args.symbol)
    out_path = Path(args.out or f"data/{asset_key}_{args.interval}.csv")

    print(f"Fetching {args.symbol} @ interval={args.interval} period={period} ...")
    df = fetch(args.symbol, args.interval, period)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # record the original symbol so downstream steps (training, the API)
    # can show a friendly label instead of just the asset key
    meta_path = out_path.parent / f"{asset_key}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"symbol": args.symbol, "source": "yfinance"}, f)

    print(f"Saved {len(df)} bars -> {out_path}")
