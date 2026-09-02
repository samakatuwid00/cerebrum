"""Pipeline extensions for 30-minute timeframe.

This module adds 30-min bar aggregation and edge computation on top of
the existing live 1-minute data stream.

Key insight: 30-min bars give 48 slots per day (24 hours × 2), allowing
more granular signals than the 6 slots per day for 4-hour bars.
"""
from __future__ import annotations

import datetime as dt
import pathlib
from collections import defaultdict
from typing import Optional

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load_iq_minute_bars(path: pathlib.Path = DATA / "iq_live_bars.csv",
                          limit: int = 2880) -> list[dict]:
    """Read up to `limit` most-recent 1-min bars from the IQ CSV."""
    if not path.exists():
        return []
    rows = []
    import csv
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


def aggregate_to_30m(bars: list[dict]) -> pd.DataFrame:
    """Group 1-min bars into 30-minute OHLCV candles.

    30-minute boundaries: 00, 30 minutes past each hour.
    Returns DataFrame indexed by datetime (hour:minute = :00 or :30).
    """
    buckets: dict[dt.datetime, list[dict]] = defaultdict(list)
    for b in bars:
        # Round down to nearest 30-minute mark
        minute = b["datetime"].minute
        if minute < 30:
            bucket_minute = 0
        else:
            bucket_minute = 30
        key = b["datetime"].replace(minute=bucket_minute, second=0, microsecond=0)
        buckets[key].append(b)

    rows = []
    for period_start, group in sorted(buckets.items()):
        group.sort(key=lambda x: x["datetime"])
        rows.append({
            "datetime": period_start,
            "open":   group[0]["open"],
            "high":   max(b["high"]   for b in group),
            "low":    min(b["low"]    for b in group),
            "close":  group[-1]["close"],
            "volume": sum(b["volume"] for b in group),
        })
    return pd.DataFrame(rows).set_index("datetime") if rows else pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )


def compute_30m_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Compute (half_hour, day_of_week) statistical edge for 30-min bars.

    Output columns: half_hour (0 or 30), day_of_week, mean_ret, std_ret,
    win_rate, count. Slots with < 30 samples are dropped (too noisy for 30m).
    """
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    ret = df["close"].pct_change().rename("ret")
    d = df.join(ret)
    d["half_hour"] = df.index.minute // 30 * 30  # 0 or 30
    d["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
    grp = d.groupby(["half_hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    edge = edge[edge["count"] >= 30].copy()
    return edge.reset_index(drop=True)


def find_next_30m_window(edge: pd.DataFrame, threshold: float = 0.65,
                         lam: float = 0.3,
                         future_only: bool = True) -> dict:
    """Find the next 30-minute window above threshold probability.

    Returns dict with:
        window_start_utc      - ISO format start time
        half_hour             - 0 or 30 (minute mark)
        day_of_week           - 0=Monday, 6=Sunday
        probability           - combined probability (0-1)
        side                  - BUY or SELL
        edge_mean_ret         - historical mean return for this slot
        edge_std_ret          - historical std dev
        n_samples             - count of samples in this slot

    Scans up to 48 half-hour slots (next 24 hours).
    Returns {'none': None} if no window above threshold.
    """
    import numpy as np
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

    # Build ordered list of upcoming half-hour boundaries
    slots_to_check = []
    for hour_offset in range(24):
        for half in [0, 30]:
            target_hour = (now.hour + hour_offset) % 24
            day_offset = (now.hour + hour_offset) // 24
            if hour_offset == 0:
                # For current hour, only take half >= current minute
                if half < now.minute:
                    continue
            target_date = now.date() + dt.timedelta(days=day_offset)
            slot_dt = dt.datetime.combine(
                target_date, dt.time(hour=target_hour, minute=half, tzinfo=dt.timezone.utc)
            )
            slots_to_check.append((slot_dt, half, slot_dt.weekday()))

    for slot_dt, half_hour, dow in slots_to_check:
        # Look up edge row for this slot
        match = edge[(edge["half_hour"] == half_hour) & (edge["day_of_week"] == dow)]
        if match.empty:
            continue

        row = match.iloc[0]
        mean_ret = float(row["mean_ret"])
        std_ret = float(row["std_ret"]) if float(row["std_ret"]) > 1e-9 else 1e-9
        sharpe = mean_ret / std_ret

        # Add sentiment delta from config
        import os
        sentiment = float(os.getenv("CEREBRUM_SENTIMENT", "0.0"))
        raw = sharpe + lam * sentiment

        prob = 1.0 / (1.0 + np.exp(-8.0 * raw))

        if prob >= threshold:
            side = "BUY" if mean_ret > 0 else "SELL"
            return {
                "window_start_utc": slot_dt.isoformat(),
                "half_hour": half_hour,
                "day_of_week": dow,
                "probability": float(prob),
                "side": side,
                "edge_mean_ret": mean_ret,
                "edge_std_ret": std_ret,
                "n_samples": int(row["count"]),
            }

    return {"none": None}


def get_evidence_30m(df: pd.DataFrame, target_half_hour: int, target_dow: int,
                     n_similar: int = 3) -> dict:
    """Pull historical evidence for a 30-minute slot.

    Similar to pipeline.get_evidence but keyed by half_hour (0 or 30).
    """
    slot_mask = (df.index.minute // 30 * 30 == target_half_hour)
    if target_dow is not None:
        slot_mask &= (df.index.dayofweek == target_dow)
    sub = df[slot_mask].copy()

    if len(sub) < 2:
        return {
            "target_hit_rate": None,
            "target_avg_move": None,
            "target_count": len(sub),
            "best_similar": [],
        }

    moves = sub["close"].pct_change().dropna()
    if len(moves) < 2:
        return {
            "target_hit_rate": None,
            "target_avg_move": None,
            "target_count": len(sub),
            "best_similar": [],
        }

    hit_rate = float((moves > 0).mean())
    avg_move = float(moves.mean())

    # Find similar slots: same half_hour, neighboring half_hours
    candidates = []
    for dh in range(-1, 2):
        if dh == 0:
            continue
        other_half = (target_half_hour + dh) % 60  # will be 0 or 30
        if other_half not in (0, 30):
            other_half = 0  # normalize edge case
        mask2 = (df.index.minute // 30 * 30 == other_half)
        if target_dow is not None:
            mask2 &= (df.index.dayofweek == target_dow)
        sub2 = df[mask2]
        if len(sub2) < 5:
            continue
        m2 = sub2["close"].pct_change().dropna()
        if len(m2) < 2:
            continue
        candidates.append({
            "half_hour": other_half,
            "day_of_week": target_dow,
            "hit_rate": float((m2 > 0).mean()),
            "avg_move": float(m2.mean()),
            "count": int(len(m2)),
        })

    candidates.sort(key=lambda c: c["hit_rate"], reverse=True)
    return {
        "target_hit_rate": hit_rate,
        "target_avg_move": avg_move,
        "target_count": int(len(sub)),
        "best_similar": candidates[:n_similar],
    }


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------
def main():
    bars = _load_iq_minute_bars()
    if not bars:
        print("[pipeline_30m] No minute bars found - start IQ feed first")
        return

    df = aggregate_to_30m(bars)
    print(f"[pipeline_30m] {len(bars)} minute bars -> {len(df)} thirty-minute bars")
    print(f"                         from {df.index.min()} to {df.index.max()}")

    edge = compute_30m_edge(df)
    print(f"[pipeline_30m] {len(edge)} half-hour slots with >=30 samples")

    best = find_next_30m_window(edge, threshold=0.65)
    if "none" in best:
        print("[pipeline_30m] No 30m window above 0.65 in next 24h")
    else:
        when = pd.Timestamp(best["window_start_utc"])
        side = best["side"]
        prob = best["probability"]
        print(f"[pipeline_30m] RECOMMENDATION: {side} at {when} UTC")
        print(f"                         prob={prob:.3f}  edge={best['edge_mean_ret']:+.4%}  n={best['n_samples']}")


if __name__ == "__main__":
    main()