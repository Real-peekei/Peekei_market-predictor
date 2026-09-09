"""
assets.py
Shared utility for namespacing data and model files by tradable product.
This is what makes the predictor generic (any yfinance/Twelve Data symbol -
gold, another commodity, a stock, a forex pair, crypto, whatever) instead
of hardcoded to one asset: every symbol gets its own filesystem-safe key,
so multiple products' data/models can coexist side by side without
colliding or overwriting each other.
"""

import re


def slugify_symbol(symbol: str) -> str:
    """
    Turns a raw ticker/symbol into a consistent, filesystem-safe asset key.
    Examples:
        'GC=F'     -> 'GC_F'
        'XAU/USD'  -> 'XAU_USD'
        'AAPL'     -> 'AAPL'
        'BTC-USD'  -> 'BTC_USD'
        'EURUSD=X' -> 'EURUSD_X'
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").upper()
    if not slug:
        raise ValueError(f"Symbol '{symbol}' produced an empty asset key after slugifying.")
    return slug
