"""Cerebrum Trader Bot – real-time USD/EUR feed via exchangerate.host.

Public API, no signup, no key required.
URL: https://api.exchangerate.host/latest?base=USD&symbols=EUR

Returns a JSON like:
  {"motd":"...","success":true,"base":"USD",
   "date":"2026-08-31","rates":{"EUR":0.92}}

The feed polls every POLL_SECONDS and appends each new rate (with UTC
timestamp) to `data/usd_eur_live.csv` so the Cerebrum pipeline can
re-read the parquet/CSV on the next cycle.

If the network is unreachable, the feed stays silent (does not raise) so
the surrounding bot keeps running on cached data.
"""

import csv
import json
import time
import urllib.request
import urllib.error
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV_PATH = DATA / "usd_eur_live.csv"

API_URL = "https://open.er-api.com/v6/latest/USD"
POLL_SECONDS = 30  # 30s keeps us well under any free-tier rate limit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _fetch_rate() -> float | None:
    """Hit the public API, return USD->EUR rate, or None on failure."""
    try:
        with urllib.request.urlopen(API_URL, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
        rate = payload.get("rates", {}).get("EUR")
        if isinstance(rate, (int, float)) and rate > 0:
            return float(rate)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        print(f"[fx_feed] fetch failed: {e}", file=sys.stderr)
        return None

def _append_row(rate: float) -> None:
    """Append a single row to the live CSV. Creates header if file is new."""
    new_file = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["datetime", "rate"])
        w.writerow([_now_utc().isoformat(), f"{rate:.6f}"])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def poll_once() -> float | None:
    """Fetch one tick, append to CSV, return the rate (or None on failure)."""
    rate = _fetch_rate()
    if rate is not None:
        _append_row(rate)
    return rate

def run_forever(poll_seconds: int = POLL_SECONDS) -> None:
    """Long-running loop.  Use Ctrl-C to stop."""
    print(f"[fx_feed] starting live USD/EUR feed "
          f"(poll every {poll_seconds}s, CSV: {CSV_PATH})", flush=True)
    try:
        while True:
            r = poll_once()
            if r is not None:
                print(f"[fx_feed] {_now_utc().isoformat()}  USD/EUR = {r:.6f}",
                      flush=True)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[fx_feed] stopped", flush=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_forever()