"""Fetch DXY (US Dollar Index) 1-hour bars from Yahoo Finance.

Yahoo Finance exposes the DXY via the ICE-USD futures ticker ``DX-Y.NYB``.
1-hour bars are limited to ~730 days of history; we pull the maximum window.

Run with:

    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe data/fetch_dxy.py

Output: ``data/dxy_1h_real.csv``

Sanity check: DXY typically trades in the 90-115 range over the last decade,
with a recent trend depending on Fed policy.  We just print min/max/mean and
verify the index stays within plausible bounds.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "data" / "dxy_1h_real.csv"


def fetch_dxy_1h() -> pd.DataFrame:
    """Download DXY (US Dollar Index) hourly bars for the max yfinance window."""
    import yfinance as yf

    print("[fetch_dxy] downloading DXY (DX-Y.NYB) 1-hour bars …", flush=True)
    df = yf.download(
        tickers="DX-Y.NYB",      # ICE US Dollar Index futures
        interval="1h",
        period="max",            # yfinance allows ~730 days for 1h interval
        progress=False,
        auto_adjust=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no data for DX-Y.NYB")

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
    print(f"[fetch_dxy] {len(df)} hourly bars from {df.index.min()} → {df.index.max()}", flush=True)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    first = df.iloc[0]
    last = df.iloc[-1]
    cmin = df["close"].min()
    cmax = df["close"].max()
    cmean = df["close"].mean()
    print("\n[fetch_dxy] SANITY CHECK")
    print(f"  first row: {df.index[0].isoformat()}  close={first['close']:.3f}")
    print(f"  last  row: {df.index[-1].isoformat()}  close={last['close']:.3f}")
    print(f"  close min / max / mean : {cmin:.3f} / {cmax:.3f} / {cmean:.3f}")
    # DXY historically trades in [89, 115] over the last decade; widen slightly
    # for safety. If the values are wildly outside, log a warning.
    if not (80.0 <= cmin and cmax <= 130.0):
        print(f"  WARNING: DXY out of plausible band 80-130.")
    else:
        print("  OK prices in plausible DXY range (80-130)")
    last_ts = df.index[-1]
    now = pd.Timestamp.now("UTC")
    age_days = (now - last_ts).total_seconds() / 86400.0
    print(f"  data age: last bar {age_days:.1f} days before now")


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_dxy_1h()
    sanity_check(df)
    df.to_csv(OUT_CSV, index_label="datetime")
    print(f"\n[fetch_dxy] saved -> {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())