"""
trust.py
Plain-language trust assessment for a trained timeframe's model, based on
its own walk-forward backtest. Shared by server.py (API) and predict_all.py
(CLI) so both give the same read.
"""

import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


def load_backtest(asset_key: str, interval: str):
    path = MODELS_DIR / f"{asset_key}_backtest_{interval}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def trust_assessment(metrics: dict | None) -> dict:
    """
    When to trust this model's direction call and price range, in short:

    - "worth_watching": hit rate clearly above a coin flip AND it beat
      buy-and-hold on both return and risk-adjusted return in the backtest.
      Worth weighing as one input - still not a reason to trade on its own.
    - "mixed": some signs of edge, some signs of none. Treat the call as
      a minor input, not something to act on by itself.
    - "low": no real edge shown - hit rate near 50% and it didn't beat
      buy-and-hold. Treat any single prediction as noise.
    - "skip": not enough backtested bars to judge either way yet.
    - "unknown": no backtest has been run for this timeframe at all.
    """
    if metrics is None:
        return {
            "level": "unknown",
            "headline": "No backtest available yet.",
            "explanation": "Retrain this timeframe to generate a walk-forward backtest before trusting any prediction from it.",
        }

    n = metrics["n_bars_tested"]
    if n < 100:
        return {
            "level": "skip",
            "headline": "Too little history to judge.",
            "explanation": f"Only {n} bars were tested in the backtest. That's too few to tell skill from luck - "
                            f"fetch more history for this timeframe and retrain before trusting it either way.",
        }

    hit_rate = metrics["direction_hit_rate"]
    sharpe_edge = metrics["strategy_sharpe"] - metrics["buyhold_sharpe"]
    return_edge = metrics["strategy_total_return"] - metrics["buyhold_total_return"]

    beats_coinflip = hit_rate > 0.53
    beats_sharpe = sharpe_edge > 0
    beats_return = return_edge > 0

    if beats_coinflip and beats_sharpe and beats_return:
        return {
            "level": "worth_watching",
            "headline": "This model has shown a real edge on past data.",
            "explanation": f"Direction hit rate was {hit_rate:.0%} (above the 50% coin-flip line), and it beat plain "
                            f"buy-and-hold on both return and risk-adjusted return (Sharpe) in the backtest. "
                            f"That's a genuine signal worth weighing - but past edge doesn't guarantee future edge, "
                            f"so treat this as one input, not a green light to trade on its own.",
        }
    if not beats_coinflip and not beats_sharpe and not beats_return:
        return {
            "level": "low",
            "headline": "No real edge shown at this timeframe.",
            "explanation": f"Direction hit rate was {hit_rate:.0%} (essentially a coin flip), and the model "
                            f"underperformed simple buy-and-hold on both return and Sharpe. Treat any single "
                            f"prediction from this model as noise, not signal, until retraining on more/fresher "
                            f"data changes this picture.",
        }
    return {
        "level": "mixed",
        "headline": "Mixed results - proceed with caution.",
        "explanation": f"Direction hit rate {hit_rate:.0%}, some metrics beat buy-and-hold and some didn't. "
                        f"This is the most common outcome and the hardest to act on - don't treat the direction "
                        f"call as reliable on its own; use it as a minor input alongside your own judgment.",
    }
