"""Cerebrum Trader Bot configuration.

Edit these values to tweak the signal engine.
"""

# Probability‑score weight for news sentiment (0 = pure technical edge)
LAMBDA_NEWS: float = 0.3

# Confidence threshold [0,1] – above this the engine recommends a trade.
#   0.55  → more signals (risk‑tolerant)
#   0.65  → balanced (default)
#   0.75  → conservative, fewer but higher‑confidence signals
CONFIDENCE_THRESHOLD: float = 0.65

# If True, the pipeline will output a Telegram alert (requires TELEGRAM_TOKEN & CHAT_ID)
ENABLE_TELEGRAM: bool = False

# ----------------------------------------------------------------------
# Hybrid execution mode
# ----------------------------------------------------------------------
# AUTO_EXECUTE controls whether alerts trigger automatic trade placement
#   False (default) -> alerts only, user must manually place on IQ Option
#   True            -> bot attempts to execute via IQ Option API
# (Manual override via Telegram reply is always available)
AUTO_EXECUTE: bool = False

# Primary timeframe for signal generation
#   "30m" -> 30-minute bars (more signals, more noise)
#   "4h"  -> 4-hour bars   (fewer signals, more reliable)
TIMEFRAME: str = "30m"

# ----------------------------------------------------------------------
# API / data paths (keep relative to repo root)
# ----------------------------------------------------------------------
# Path to the pre‑aggregated 1‑hour USD/EUR OHLCV parquet
DATA_PATH: str = "data/usd_eur_hourly.csv"

# Economic‑calendar CSV (optional – used for news‑sentiment overlay)
CALENDAR_CSV: str = "data/forexfactory_calendar.csv"

# ----------------------------------------------------------------------
# IQ Option broker (for live trading)
# ----------------------------------------------------------------------
# Get this from your IQ Option profile page (SSID field).
# Leave empty string to disable the IQ Option feed.
IQOPTION_SSID: str = ""

# How many 1‑minute bars to keep in the rolling window for live‑edge
# computation.  1440 minutes = 24 hours.
IQOPTION_ROLLING_BARS: int = 1440