"""Sell‑window scanner for the Cerebrum Trader Bot.

Same statistical‑edge + sentiment engine as `src.pipeline.find_next_window`,
but with two changes:

  1. The edge sign is *inverted* when scoring (we want negative expected
     return → good for a short / sell).
  2. The threshold is applied to the *negated* probability so the same
     confidence bar (0.55, 0.65, …) picks a real SELL window.

Usage:
    from src.sell_window import find_next_sell_window
    result = find_next_sell_window(edge, threshold=0.55, lam=0.3)
"""

import datetime as dt
import pytz
import numpy as np
from . import pipeline as pl

# ---------------------------------------------------------------------------
# Inverted scoring
# ---------------------------------------------------------------------------
def _sell_probability(edge_row, lam: float, sentiment: float) -> float:
    """Probability that the hour has *negative* expected return.

    We feed the negated mean_ret through the same logistic so the resulting
    probability represents confidence in a downward move.
    """
    flipped = edge_row.copy()
    flipped["mean_ret"] = -flipped["mean_ret"]  # invert
    return pl.combine_probability(flipped, lam=lam, sentiment=sentiment)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_next_sell_window(edge, threshold: float = 0.55, lam: float = 0.3,
                          future_only: bool = True) -> dict:
    """Scan forward for the first hour where the *downside* probability ≥ threshold."""
    utc = pytz.UTC
    now = dt.datetime.now(utc)
    for offset in range(24 if future_only else 0, 48):
        target = (now + dt.timedelta(hours=offset)).hour
        match = edge[edge["hour"] == target]
        if match.empty:
            continue
        sent = pl.load_sentiment_score(target)
        prob = _sell_probability(match.iloc[0], lam=lam, sentiment=sent)
        if prob >= threshold:
            nxt = (now + dt.timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
            return {
                "window_start_utc": nxt.isoformat(),
                "hour_local": target,
                "probability": prob,
                "edge_mean_ret": match.iloc[0]["mean_ret"],   # original (negative) value
                "sentiment": sent,
                "side": "SELL",
            }
    return {"none": None}

# ---------------------------------------------------------------------------
# CLI self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pl.load_hourly_parquet()
    edge = pl.compute_hourly_edge(df)
    res = find_next_sell_window(edge, threshold=0.50, lam=0.3)
    if "none" in res:
        print("No SELL window above threshold in next 48h.")
    else:
        when = dt.datetime.fromisoformat(res["window_start_utc"])
        print(f"🔴 Next SELL window: {when.strftime('%H:%M')} UTC")
        print(
            f"   Probability: {res['probability']:.3f}  "
            f"(edge={res['edge_mean_ret']:+.2%})  "
            f"sentiment={res['sentiment']:+.2f}"
        )