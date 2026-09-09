"""
intervals.py
Shared utility for sorting timeframe intervals chronologically (1m before
5m before 1h before 1d ...) instead of alphabetically, which is wrong -
plain string sort puts "1d" before "1h" before "1m" before "1wk", mixing
up minutes/hours/days/weeks/months in a meaningless order. Used by the API
(so /api/intervals and /api/predict-all return ascending order) and the
CLI scripts. The frontend has its own equivalent in script.js.
"""

# duration of one bar, in minutes, for every interval code used anywhere
# in this project (yfinance's set, Twelve Data's set, and their union).
# "60m" and "1h" are the same duration on purpose - they're aliases.
INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
    "60m": 60, "1h": 60, "90m": 90, "2h": 120, "4h": 240, "8h": 480,
    "1d": 1440, "1wk": 1440 * 7, "1mo": 1440 * 30,  # 1mo is approximate, fine for sorting
}


def interval_sort_key(interval: str) -> int:
    """Unknown/future interval codes sort last rather than erroring."""
    return INTERVAL_MINUTES.get(interval, 10 ** 9)


def sort_intervals(intervals: list[str]) -> list[str]:
    return sorted(intervals, key=interval_sort_key)
