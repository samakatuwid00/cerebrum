"""Sentiment overlay for the Cerebrum Trader Bot.

Provides a `sentiment_score(hour_utc)` function that returns a float in [-1, +1].

Prototype implementation:
  • VADER lexicon on headlines fetched from NewsAPI (or any RSS you prefer).
  • If a high‑impact economic event is scheduled for the given hour, add +0.3.
  • Otherwise return 0.0 (neutral).

Replace the `fetch_headlines()` stub with a real API call for production.
"""

# Lazy nltk import so the module loads even when nltk is not installed.
try:
    import nltk
    try:
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
        _NLTK_OK = True
    except Exception:
        _SIA = None
        _NLTK_OK = False
except Exception:
    _SIA = None
    _NLTK_OK = False
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _hour_utc(dt: datetime) -> int:
    return dt.astimezone(timezone.utc).hour

# ---------------------------------------------------------------------------
# News‑headline fetcher (stub – replace with NewsAPI / RSS)
# ---------------------------------------------------------------------------
def fetch_headlines() -> list[str]:
    """Return a list of recent EUR/USD‑related headlines (UTC today)."""
    # ---- PROTOTYPE STUB ----
    # In a real deployment you would:
    #   1. Call NewsAPI:   https://newsapi.org/v2/everything?domains=bloomberg.com,reuters.com
    #   2. Filter by keyword: "EUR" OR "USD" OR "ECB" OR "CPI"
    #   3. Return only titles from the last 24 h, UTC.
    # ----
    return [
        "ECB keeps rates steady, markets eye inflation",
        "US dollar steadies ahead of jobs data",
        # add more as you like for demo purposes
    ]  # ← replace with real fetch
    # -------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core sentiment scorer
# ---------------------------------------------------------------------------
def sentiment_score(for_hour: int | None = None) -> float:
    """Return a sentiment delta in [-1, +1] for *for_hour* (UTC hour 0‑23).

    The score is a simple combination of:
      1️⃣ VADER average sentiment of recent EUR/USD headlines  (range -1..+1)
      2️⃣ +0.3 if a high‑impact economic event is scheduled for that hour
    """
    # ------------------------------------------------------------------
    # 1️⃣ VADER part (graceful fallback if nltk is unavailable)
    # ------------------------------------------------------------------
    if not _NLTK_OK:
        vader_avg = 0.0
    else:
        headlines = fetch_headlines()
        if not headlines:
            vader_avg = 0.0
        else:
            vs = [_SIA.polarity_scores(h) for h in headlines]
            vader_avg = sum(v["compound"] for v in vs) / len(vs)

    # ------------------------------------------------------------------
    # 2️⃣ Economic‑calendar bump (placeholder)
    # ------------------------------------------------------------------
    # TODO: load a real calendar (ForexFactory/Investing.com) and check if any
    # high‑impact event falls at `for_hour`. If yes, add +0.3 (or -0.3 for
    # negative‑impact events).
    # ------------------------------------------------------------------
    calendar_bonus = 0.0  # prototype: neutral

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    total = vader_avg + calendar_bonus
    # Clip to the allowed [-1, +1] range just in case
    return max(-1.0, min(1.0, total))

# ---------------------------------------------------------------------------
# Quick self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Print sentiment for every UTC hour of today
    now = _now_utc()
    for h in range(24):
        s = sentiment_score(h)
        print(f"Hour {h:02d} UTC → sentiment = {s:+.2f}")