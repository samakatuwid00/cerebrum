"""Generate a tiny synthetic 2‑year USD/EUR 1‑hour OHLCV dataset.

Writes both formats:
  • data/usd_eur_hourly.parquet  (if pyarrow is installed)
  • data/usd_eur_hourly.csv      (always — stdlib‑compatible)

The CSV is what the stdlib pipeline consumes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv

DATA = Path(__file__).parent / "data"
OUT_PARQUET = DATA / "usd_eur_hourly.parquet"
OUT_CSV = DATA / "usd_eur_hourly.csv"

# ---------------------------------------------------------------------------
# Parameters (make them configurable if you wish)
# ---------------------------------------------------------------------------
START = pd.Timestamp("2023-01-01", tz="UTC")
END   = pd.Timestamp("2025-12-31", tz="UTC")
FREQ  = "1h"                                                   # 1‑hour candles (lowercase 'h' for pandas 3.x)
MEAN_RET = 0.0002                                              # ~0.02% mean drift per hour
STD_RET  = 0.015                                               # 1.5% hourly volatility
SEASON_STRENGTH = 0.0001                                       # tiny diurnal pattern
NOISE_SCALE = 0.005                                            # extra jitter

# ---------------------------------------------------------------------------
# Build the hourly index
# ---------------------------------------------------------------------------
rng = pd.date_range(start=START, end=END - pd.Timedelta(hours=1), freq=FREQ)
n = len(rng)

# ---------------------------------------------------------------------------
# Random‑walk with tiny seasonal tilt and jitter
# ---------------------------------------------------------------------------
np.random.seed(42)  # reproducible
drift = np.random.normal(MEAN_RET, STD_RET, size=n)

# tiny diurnal component: strongest around 02:00‑04:00 UTC (European open) and 13:00‑15:00 UTC (US open)
hour = rng.hour
seasonal = SEASON_STRENGTH * np.sin(2 * np.pi * (hour - 2) / 24)   # simple sine wave

# optional volatility clustering (optional – omitted for simplicity)

returns = drift + seasonal + np.random.normal(0, NOISE_SCALE, size=n)

# Price path (cumulative product of (1+ret))
close = 1.0000 * np.cumprod(1 + returns)

# Build OHLC from the close + tiny random spread
high = close + np.abs(np.random.normal(0, 0.001, size=n))
low  = close - np.abs(np.random.normal(0, 0.001, size=n))
# ensure high >= close >= low and high >= low
high = np.maximum.accumulate(np.maximum(high, close))
low  = np.minimum.accumulate(np.minimum(low, close))

volume = np.random.randint(1000, 5000, size=n)   # dummy volume

# ---------------------------------------------------------------------------
# Assemble DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "open":  close * (1 - np.random.uniform(0, 0.0005, size=n)),
    "high": high,
    "low":  low,
    "close": close,
    "volume": volume,
}, index=rng)

# ---------------------------------------------------------------------------
# Write outputs (CSV always, parquet if pyarrow is present)
# ---------------------------------------------------------------------------
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index_label="datetime")
print(f"✅ CSV written: {OUT_CSV}  ({len(df)} rows, {START.date()} → {END.date()})")

try:
    df.to_parquet(OUT_PARQUET)
    print(f"✅ Parquet written: {OUT_PARQUET}")
except Exception as e:
    print(f"⚠️  Skipped parquet (pyarrow not installed or build mismatch): {e}")