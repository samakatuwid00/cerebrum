"""Fetch XAU/USD (Gold vs US Dollar) 1-hour OHLCV from Yahoo Finance.

Yahoo Finance exposes gold via COMEX futures ticker ``GC=F`` (USD per troy
ounce).  The forex pair ``XAUUSD=X`` was delisted, so we use the futures
contract.  yfinance allows the maximum ~730-day lookback for 1h bars; we
fetch that full window and save it as 1-hour bars (no aggregation).

Run with:

    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe data/fetch_xau_1h.py

Output: ``data/xau_usd_1h_real.csv``
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "data" / "xau_usd_1h_real.csv"


def fetch_xau_1h() -> pd.DataFrame:
    """Download XAU/USD hourly bars for the maximum yfinance window (~730 days)."""
    import yfinance as yf

    print("[fetch_xau_1h] downloading XAU/USD (GC=F) 1-hour bars …", flush=True)
    df = yf.download(
        tickers="GC=F",          # COMEX gold futures (USD/oz) — XAUUSD=X was delisted
        interval="1h",
        period="max",            # yfinance allows ~730 days for 1h interval
        progress=False,
        auto_adjust=False,       # keep raw close
    )
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no data for GC=F")

    # yfinance returns columns either as MultiIndex or flat depending on version
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    print(f"[fetch_xau_1h] {len(df)} hourly bars from {df.index.min()} → {df.index.max()}", flush=True)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    first = df.iloc[0]
    last = df.iloc[-1]
    cmin = df["close"].min()
    cmax = df["close"].max()
    cmean = df["close"].mean()
    print("\n[fetch_xau_1h] SANITY CHECK")
    print(f"  first row: {df.index[0].isoformat()}  close={first['close']:.2f}")
    print(f"  last  row: {df.index[-1].isoformat()}  close={last['close']:.2f}")
    print(f"  close min / max / mean : {cmin:.2f} / {cmax:.2f} / {cmean:.2f}")
    # 2025-2026 gold has traded ~$2,400–$4,400; sanity-check a wide range
    if not (2000.0 <= cmin and cmax <= 5000.0):
        print(f"  WARNING: price out of expected gold range (2000-5000).  "
              f"Proceeding – gold is volatile but typically in this band.", flush=True)
    else:
        print("  OK prices in expected 2000-5000 range for gold (GC=F futures)")
    # recent-dates sanity: last bar should be within the last few days
    last_ts = df.index[-1]
    now = pd.Timestamp.now("UTC")
    age_days = (now - last_ts).total_seconds() / 86400.0
    print(f"  data age: last bar {age_days:.1f} days before now")


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_xau_1h()
    sanity_check(df)
    df.to_csv(OUT_CSV, index_label="datetime")
    print(f"\n[fetch_xau_1h] saved -> {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())