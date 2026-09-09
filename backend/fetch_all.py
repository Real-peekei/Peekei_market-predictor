"""
fetch_all.py
Downloads data for every common timeframe of ANY tradable symbol in one
go, using sensible default lookback periods per interval. Equivalent to
running data_fetch.py (or twelvedata_fetch.py) once per interval, but you
only have to type one command. Not limited to gold - pass any --symbol
your chosen source supports.

Two sources, with different interval coverage:
    --source yfinance   (default) - no API key needed, reliable for
                         exchange-traded instruments (stocks, futures like
                         GC=F, crypto, indices). Supports 10 intervals
                         (yfinance's own limit). Also works for OTC/forex
                         spot tickers (e.g. XAUUSD=X, EURUSD=X) but those
                         can be less reliable on Yahoo's end.
    --source twelvedata - dedicated spot/forex/stocks/crypto endpoint,
                         generally more reliable for spot/OTC instruments
                         specifically. Supports all 12 of Twelve Data's
                         intervals (adds 45m/2h/4h/8h on top of yfinance's
                         set). Needs a free API key: see twelvedata_fetch.py
                         for setup. For futures via Twelve Data (real trade
                         volume, unlike spot) run twelvedata_symbol_search.py
                         first to find the correct symbol - see that file
                         for why.

Usage:
    python fetch_all.py                                       # GC=F gold futures via yfinance, 10 intervals
    python fetch_all.py --symbol AAPL                          # a stock via yfinance
    python fetch_all.py --symbol BTC-USD                       # crypto via yfinance
    python fetch_all.py --source twelvedata --symbol XAU/USD   # spot gold via Twelve Data, all 12 intervals
    python fetch_all.py --source twelvedata --symbol EUR/USD   # forex via Twelve Data
    python fetch_all.py --intervals 1d 1h 15m                  # only fetch a subset

Output files are automatically namespaced by symbol (e.g. --symbol AAPL
--interval 1d writes data/AAPL_1d.csv), so multiple products' data can
coexist without colliding - fetch and train as many symbols as you want.
"""

import argparse
import subprocess
import sys

from assets import slugify_symbol
from data_fetch import DEFAULT_PERIOD, DEFAULT_TICKER
from twelvedata_fetch import (
    DEFAULT_SYMBOL as TD_DEFAULT_SYMBOL,
    ALL_TWELVEDATA_INTERVALS,
    DEFAULT_PERIOD as TD_DEFAULT_PERIOD,
)

YFINANCE_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "90m", "1h", "1d", "1wk", "1mo"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yfinance", "twelvedata"], default="yfinance")
    parser.add_argument("--symbol", "--ticker", dest="symbol", default=None,
                         help="Any symbol your chosen source supports, e.g. GC=F, AAPL, BTC-USD, XAU/USD, EUR/USD. "
                              f"Defaults to {DEFAULT_TICKER} for yfinance or {TD_DEFAULT_SYMBOL} for twelvedata.")
    parser.add_argument("--intervals", nargs="+", default=None,
                         choices=sorted(set(YFINANCE_INTERVALS) | set(ALL_TWELVEDATA_INTERVALS)),
                         help="defaults to all intervals the chosen --source supports")
    args = parser.parse_args()

    # per-source default symbol, only applied if the user didn't pass one
    if args.symbol is None:
        args.symbol = DEFAULT_TICKER if args.source == "yfinance" else TD_DEFAULT_SYMBOL

    # default interval set depends on source: yfinance supports 10,
    # Twelve Data supports all 12
    if args.intervals is None:
        args.intervals = YFINANCE_INTERVALS if args.source == "yfinance" else ALL_TWELVEDATA_INTERVALS
    elif args.source == "yfinance":
        unsupported = [i for i in args.intervals if i not in YFINANCE_INTERVALS]
        if unsupported:
            raise SystemExit(f"yfinance doesn't support interval(s) {unsupported} - "
                              f"those need --source twelvedata instead.")

    asset_key = slugify_symbol(args.symbol)
    print(f"Fetching {len(args.intervals)} timeframes for {args.symbol} (asset key: {asset_key}) via {args.source}...\n")

    period_lookup = DEFAULT_PERIOD if args.source == "yfinance" else TD_DEFAULT_PERIOD

    failures = []
    for interval in args.intervals:
        period = period_lookup.get(interval, "5y")
        # both sources write to the same asset-namespaced filename
        # convention so train_all.py finds them regardless of source
        out = f"data/{asset_key}_{interval}.csv"
        print(f"--- {interval} (period={period}) ---")

        if args.source == "yfinance":
            cmd = [sys.executable, "data_fetch.py",
                   "--symbol", args.symbol, "--interval", interval, "--period", period, "--out", out]
        else:
            cmd = [sys.executable, "twelvedata_fetch.py",
                   "--symbol", args.symbol, "--interval", interval, "--period", period, "--out", out]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(interval)
        print()

    if failures:
        hint = ("often a Yahoo intraday lookback limit - see README" if args.source == "yfinance"
                 else "check your TWELVEDATA_API_KEY, free-tier rate limits, and that --symbol is correct - see README")
        print(f"Done, with failures on: {', '.join(failures)} ({hint})")
    else:
        print(f"Done. All timeframes fetched successfully for {args.symbol} (asset key: {asset_key}).")
