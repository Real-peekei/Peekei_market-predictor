"""
twelvedata_symbol_search.py
Looks up the EXACT symbol Twelve Data recognizes for a given search term,
using their own symbol_search reference endpoint. Use this before trying
to fetch gold futures - Twelve Data's dedicated Commodities product
(XAU/USD etc.) is documented as spot-only, and no stable, published symbol
format for COMEX gold futures exists in their public docs. Rather than
guess a symbol and risk silently fetching the wrong instrument (or one
that doesn't exist), this asks Twelve Data directly what it actually has.

Usage:
    python twelvedata_symbol_search.py --query gold
    python twelvedata_symbol_search.py --query "gold futures"
    python twelvedata_symbol_search.py --query GC
"""

import argparse
import os

import requests

SEARCH_URL = "https://api.twelvedata.com/symbol_search"


def search(query: str, api_key: str) -> list[dict]:
    resp = requests.get(SEARCH_URL, params={"symbol": query, "apikey": api_key}, timeout=30)
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error ({data.get('code')}): {data.get('message', 'unknown error')}")

    return data.get("data", [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search term, e.g. 'gold', 'gold futures', 'GC'")
    parser.add_argument("--api-key", default=None,
                         help="Twelve Data API key. Defaults to the TWELVEDATA_API_KEY env var if not passed.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set the TWELVEDATA_API_KEY environment variable, "
            "or pass --api-key directly. Sign up free at https://twelvedata.com"
        )

    results = search(args.query, api_key)

    if not results:
        print(f"No matches for '{args.query}'. Try a broader term, e.g. 'gold' instead of 'gold futures continuous'.")
        raise SystemExit(0)

    print(f"{len(results)} match(es) for '{args.query}':\n")
    print(f"{'Symbol':15s} {'Type':12s} {'Exchange':20s} {'Currency':10s} Name")
    print("-" * 95)
    for r in results:
        print(f"{r.get('symbol', '?'):15s} {r.get('instrument_type', '?'):12s} "
              f"{r.get('exchange', '?'):20s} {r.get('currency', '?'):10s} {r.get('instrument_name', '?')}")

    print(
        "\nLook for an entry with instrument_type containing 'Futures' or an exchange like "
        "COMEX/CME/NYMEX. Copy its exact 'Symbol' value and use it with:\n"
        "  python twelvedata_fetch.py --symbol \"<symbol from above>\" --interval 1d --out data/gold_1d.csv\n"
        "If nothing futures-related shows up, gold futures likely aren't available on your current "
        "Twelve Data plan - check https://twelvedata.com/pricing for which tier includes futures data."
    )
