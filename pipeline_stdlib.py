"""Cerebrum Trader Bot — stdlib-only fallback.

Runs the *same* end-to-end flow as `src/pipeline.py`, but without
pandas / numpy / pyarrow.  Useful when the full scientific stack
isn't available.

Data source : data/usd_eur_hourly.csv  (pre-generated, see generate_synthetic_data_stdlib.py)
Output      : prints the next buy window (HH:MM UTC) + probability
"""

import csv
import math
import datetime as dt
import pytz
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
CSV_PATH = DATA / "usd_eur_hourly.csv"

# ---------------------------------------------------------------------------
# 1️⃣ Load CSV  (datetime, open, high, low, close, volume)
# ---------------------------------------------------------------------------
def load_ohlcv(path: pathlib.Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "dt": dt.datetime.fromisoformat(r["datetime"]).replace(tzinfo=dt.timezone.utc),
                "close": float(r["close"]),
            })
    rows.sort(key=lambda x: x["dt"])
    return rows

# ---------------------------------------------------------------------------
# 2️⃣ Compute hourly statistical edge
# ---------------------------------------------------------------------------
def hourly_edge(rows):
    """Return dict hour -> (mean_ret, std_ret, win_rate, count)"""
    buckets = {h: [] for h in range(24)}
    for i in range(1, len(rows)):
        prev = rows[i - 1]["close"]
        cur = rows[i]["close"]
        ret = (cur - prev) / prev
        h = rows[i]["dt"].hour
        buckets[h].append(ret)
    edge = {}
    for h, rets in buckets.items():
        if not rets:
            edge[h] = (0.0, 0.0, 0.0, 0)
            continue
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        std = math.sqrt(var)
        win = sum(1 for r in rets if r > 0) / n
        edge[h] = (mean, std, win, n)
    return edge

# ---------------------------------------------------------------------------
# 3️⃣ Sentiment overlay (placeholder; in real deployment use VADER or a model)
# ---------------------------------------------------------------------------
def sentiment_score(for_hour: int) -> float:
    # Prototype: 0.0 (neutral).  Plug in `src.sentiment.sentiment_score(h)` here.
    return 0.0

# ---------------------------------------------------------------------------
# 4️⃣ Combine → probability in [0,1] via logistic on (mean_ret + λ·sentiment)
# ---------------------------------------------------------------------------
def combine_probability(mean_ret: float, lam: float, sentiment: float) -> float:
    raw = mean_ret + lam * sentiment
    return 1.0 / (1.0 + math.exp(-10 * raw))   # logistic, scales ~[-.02,+.02] -> 0..1

# ---------------------------------------------------------------------------
# 5️⃣ Find next buy window
# ---------------------------------------------------------------------------
def find_next_window(edge, threshold=0.65, lam=0.3):
    now = dt.datetime.now(pytz.UTC)
    for offset in range(24, 48):
        target = (now + dt.timedelta(hours=offset)).hour
        mean, std, win, n = edge[target]
        if n == 0:
            continue
        sent = sentiment_score(target)
        prob = combine_probability(mean, lam, sent)
        if prob >= threshold:
            nxt = (now + dt.timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
            return {
                "window_start_utc": nxt.isoformat(),
                "hour_local": target,
                "probability": prob,
                "mean_ret": mean,
                "std_ret": std,
                "win_rate": win,
                "n": n,
                "sentiment": sent,
            }
    return {"none": None}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"❌ {CSV_PATH} not found.  Run `python generate_synthetic_data_stdlib.py` first."
        )
    rows = load_ohlcv(CSV_PATH)
    edge = hourly_edge(rows)
    # Print the full hourly edge table so you can see what the engine sees
    print("Hourly edge (mean return per hour, UTC):")
    for h in range(24):
        m, s, w, n = edge[h]
        print(f"  {h:02d} UTC | mean={m:+.4%}  σ={s:.2%}  win={w:.0%}  n={n}")
    print()
    # 0.50 picks any hour with positive mean return; raise to 0.55+ for stricter
    res = find_next_window(edge, threshold=0.50, lam=0.3)
    if "none" in res:
        print("⏳ No buy/sell window above threshold in the next 48h.")
    else:
        when = dt.datetime.fromisoformat(res["window_start_utc"])
        print(f"🕐 Next buy window: {when.strftime('%H:%M')} UTC")
        print(
            f"   Probability: {res['probability']:.3f}  "
            f"(edge={res['mean_ret']:+.2%}  σ={res['std_ret']:.2%}  "
            f"win={res['win_rate']:.0%}  n={res['n']})  "
            f"sentiment={res['sentiment']:+.2f}"
        )

if __name__ == "__main__":
    main()