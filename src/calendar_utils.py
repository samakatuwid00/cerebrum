"""Economic‑calendar loader for the Cerebrum Trader Bot.

Reads a ForexFactory / Investing.com CSV and returns a dict mapping
UTC hour → list of events with impact level.

Expected CSV columns (header row):
    time, event, impact, forecast, actual

- `time` is in UTC (or can be parsed to UTC).
- `impact` is one of: "high", "medium", "low".

The function returns:
    { hour: [{"event":..., "impact":...}, ...] }
"""

import csv
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Helper: parse a time string like "14:30" or "08:00 GMT" into a UTC hour
# ---------------------------------------------------------------------------
def _parse_time_to_utc_hour(time_str: str) -> int:
    """Convert various time formats to a UTC hour (0‑23)."""
    # Strip any non‑digit chars except colon
    t = re.sub(r"[^0-9:]", "", time_str)
    try:
        h, m = t.split(":")
        h, m = int(h) % 24, int(m)
        return h  # we just need the hour; ignore minute for our bucketing
    except Exception:
        return -1  # unscheduled

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_calendar(csv_filename: str = "forexfactory_calendar.csv") -> Dict[int, List[Dict[str, Any]]]:
    """Load the economic calendar and return a dict: hour_utc → events."""
    path = DATA / csv_filename
    if not path.exists():
        raise FileNotFoundError(f"Calendar file not found: {path}")

    events_by_hour: Dict[int, List[Dict[str, Any]]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise time
            time_str = row.get("time", "").strip()
            if not time_str:
                continue
            hour = _parse_time_to_utc_hour(time_str)
            if hour < 0:
                continue  # couldn't parse

            impact = row.get("impact", "low").strip().lower()
            event_name = row.get("event", "").strip()

            events_by_hour.setdefault(hour, []).append({
                "event": event_name,
                "impact": impact,
                "forecast": row.get("forecast", "").strip(),
                "actual": row.get("actual", "").strip(),
            })

    return events_by_hour

# ---------------------------------------------------------------------------
# Quick self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cal = load_calendar()
    for h, evts in sorted(cal.items()):
        print(f"UTC hour {h}: {len(evts)} event(s)")
        for e in evts:
            print(f"  - {e['event']} (impact={e['impact']})")