"""Live edge from rolling IQ-Option / FX 1-minute bars.

Reads `data/iq_live_bars.csv` (or `data/usd_eur_live.csv` as a fallback),
aggregates the last ROLLING_BARS 1-minute entries into 1-hour OHLCV
candles, then computes the same (hour, day_of_week) statistical edge used
by the synthetic baseline.  Returns an `edge` DataFrame compatible with
`pipeline.find_next_window`.
"""

import csv
import datetime as dt
import pathlib
from collections import defaultdict
from typing import Optional

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
IQ_CSV = ROOT / "data" / "iq_live_bars.csv"
FX_CSV = ROOT / "data" / "usd_eur_live.csv"


def _load_iq_minute_bars(path: pathlib.Path = IQ_CSV,
                          limit: int = 1440) -> list[dict]:
    """Read up to `limit` most-recent 1-min bars from the IQ CSV."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "datetime": dt.datetime.fromisoformat(r["datetime"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low":  float(r["low"]),
                    "close": float(r["close"]),
                    "volume": int(r.get("volume", 0) or 0),
                })
            except Exception:
                continue
    rows.sort(key=lambda x: x["datetime"])
    return rows[-limit:]


def _aggregate_to_1h(bars: list[dict]) -> pd.DataFrame:
    """Group 1-min bars into 1-hour OHLCV candles."""
    buckets: dict[dt.datetime, list[dict]] = defaultdict(list)
    for b in bars:
        hour_key = b["datetime"].replace(minute=0, second=0, microsecond=0)
        buckets[hour_key].append(b)

    rows = []
    for hour, group in sorted(buckets.items()):
        group.sort(key=lambda x: x["datetime"])
        rows.append({
            "datetime": hour,
            "open":   group[0]["open"],
            "high":   max(b["high"]   for b in group),
            "low":    min(b["low"]    for b in group),
            "close":  group[-1]["close"],
            "volume": sum(b["volume"] for b in group),
        })
    return pd.DataFrame(rows).set_index("datetime") if rows else pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )


def compute_live_edge(min_bars_required: int = 24) -> Optional[pd.DataFrame]:
    """Return a DataFrame with the (hour, day_of_week) statistical edge derived
    from the live rolling 1-min bars.  Returns None if we don't have enough data.

    Default 24 bars = 12 hours of live data, enough to compute a meaningful
    per-hour edge (previous default was 60 = 60h which was too conservative).
    """
    bars = _load_iq_minute_bars(limit=1440)
    if len(bars) < min_bars_required:
        return None
    df = _aggregate_to_1h(bars)
    if len(df) < 2:
        return None

    # Use the same (hour, day_of_week) grouping as pipeline.compute_edge()
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    ret = df["close"].pct_change().rename("ret")
    d = df.join(ret)
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    grp = d.groupby(["hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    return edge.reset_index(drop=True)


if __name__ == "__main__":
    e = compute_live_edge()
    if e is None:
        print("Not enough live 1-min bars yet (need >= 24).")
        print("   Start the IQ Option collector with your SSID to stream them.")
    else:
        print(f"Live (hour, day_of_week) edge  ({len(e)} slots, {e['count'].sum()} bars total):")
        for _, r in e.iterrows():
            print(f"  {int(r['hour']):02d}:00 UTC dow={int(r['day_of_week'])} | "
                  f"mean={r['mean_ret']:+.4%}  "
                  f"sigma={r['std_ret']:.2%}  win={r['win_rate']:.0%}  n={int(r['count'])}")
