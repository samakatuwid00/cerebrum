"""Fetch US10Y (10-Year Treasury Yield) 1-hour bars from Yahoo Finance.

Yahoo Finance exposes the 10-year Treasury yield via the CBOE ticker ``^TNX``
(yield in percent, e.g. 4.25 means 4.25%). 1-hour bars are limited to ~730 days
of history; we pull the maximum window, same as the DXY fetcher.

Run with:

    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe data/fetch_us10y.py

Output: ``data/us10y_1h_real.csv``

Sanity check: the 10Y yield has historically ranged roughly 0.5%–8.0% over the
last few decades (and traded inside that band recently). We print min/max/mean
and warn if the values fall outside the plausible band.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "data" / "us10y_1h_real.csv"


def fetch_us10y_1h() -> pd.DataFrame:
    """Download US10Y (^TNX) hourly bars for the max yfinance window."""
    import yfinance as yf

    print("[fetch_us10y] downloading US10Y (^TNX) 1-hour bars …", flush=True)
    df = yf.download(
        tickers="^TNX",          # CBOE 10-Year Treasury yield index (percent)
        interval="1h",
        period="max",            # yfinance allows ~730 days for 1h interval
        progress=False,
        auto_adjust=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no data for ^TNX")

    # yfinance may return MultiIndex columns or flat columns depending on version
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
    print(f"[fetch_us10y] {len(df)} hourly bars from {df.index.min()} → {df.index.max()}", flush=True)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    first = df.iloc[0]
    last = df.iloc[-1]
    cmin = df["close"].min()
    cmax = df["close"].max()
    cmean = df["close"].mean()
    print("\n[fetch_us10y] SANITY CHECK")
    print(f"  first row: {df.index[0].isoformat()}  close={first['close']:.3f}")
    print(f"  last  row: {df.index[-1].isoformat()}  close={last['close']:.3f}")
    print(f"  close min / max / mean : {cmin:.3f} / {cmax:.3f} / {cmean:.3f}")
    # 10Y yield historically in [0.5, 8.0] percent over the last few decades.
    if not (0.5 <= cmin and cmax <= 8.0):
        print(f"  WARNING: US10Y out of plausible band 0.5-8.0 percent.")
    else:
        print("  OK yield in plausible US10Y range (0.5-8.0 percent)")
    last_ts = df.index[-1]
    now = pd.Timestamp.now("UTC")
    age_days = (now - last_ts).total_seconds() / 86400.0
    print(f"  data age: last bar {age_days:.1f} days before now")


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_us10y_1h()
    sanity_check(df)
    df.to_csv(OUT_CSV, index_label="datetime")
    print(f"\n[fetch_us10y] saved -> {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
