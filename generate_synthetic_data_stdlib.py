"""Stdlib-only synthetic data generator.

Produces `data/usd_eur_hourly.csv` (datetime, open, high, low, close, volume)
covering 2023‑01‑01 → 2025‑12‑31 at 1‑hour granularity.

Uses only the Python standard library so the pipeline can be exercised
even without numpy / pandas.
"""

import csv
import math
import random
import datetime as dt
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "usd_eur_hourly.csv"

START = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
END   = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

MEAN_RET = 0.0002      # +0.02% per hour baseline drift
STD_RET  = 0.015       # 1.5% hourly volatility
SEASON_AMP = 0.0006    # amplified diurnal tilt (so one hour stands out clearly)
NOISE     = 0.005

random.seed(42)

# ---------------------------------------------------------------------------
# Build series
# ---------------------------------------------------------------------------
timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
price = 1.0000

cur = START
while cur < END:
    # mean‑return + tiny diurnal tilt
    hour = cur.hour
    seasonal = SEASON_AMP * math.sin(2 * math.pi * (hour - 2) / 24)
    ret = random.gauss(MEAN_RET, STD_RET) + seasonal + random.gauss(0, NOISE)
    open_p = price
    close_p = open_p * (1 + ret)
    high_p = max(open_p, close_p) + abs(random.gauss(0, 0.001))
    low_p  = min(open_p, close_p) - abs(random.gauss(0, 0.001))
    vol    = random.randint(1000, 5000)

    timestamps.append(cur.isoformat())
    opens.append(open_p)
    highs.append(high_p)
    lows.append(low_p)
    closes.append(close_p)
    volumes.append(vol)

    price = close_p
    cur += dt.timedelta(hours=1)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "open", "high", "low", "close", "volume"])
    for row in zip(timestamps, opens, highs, lows, closes, volumes):
        w.writerow([f"{row[0]}", f"{row[1]:.6f}", f"{row[2]:.6f}",
                    f"{row[3]:.6f}", f"{row[4]:.6f}", row[5]])

print(f"✅ Synthetic data written to {OUT}  ({len(timestamps)} rows, "
      f"{START.date()} → {END.date()})")