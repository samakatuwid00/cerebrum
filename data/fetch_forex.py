"""Fetch 1h forex pairs (less-efficient EM/volatile crosses) from Yahoo Finance.

Pulls 6 pairs that are widely believed to be less efficient than XAU/USD:
  USDTRY=X  - US Dollar / Turkish Lira (EM)
  GBPJPY=X  - British Pound / Japanese Yen (volatile cross)
  EURTRY=X  - Euro / Turkish Lira (EM)
  ZAR=X     - USD / South African Rand (EM)
  MXN=X     - USD / Mexican Peso (EM)
  BRL=X     - USD / Brazilian Real (EM)

yfinance tickers use the `=X` suffix for FX pairs and plain ticker for
cross-asset proxies.  For USD-based pairs (ZAR=X, MXN=X, BRL=X) the value is
"USD per unit of local currency" — high = strong USD.

The 1h interval has a ~730-day lookback in yfinance; ``period="max"`` is
the safe choice.

Run with:
    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe data/fetch_forex.py

Outputs ``data/{slug}_1h_real.csv`` for each pair plus a combined sanity
table printed to stdout.

If yfinance returns no data for a pair, the script keeps going and records
a row with row_count=0 so the operator can spot dead tickers.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data"

# (ticker, slug) — slug is the filename prefix and label in sanity output
PAIRS = [
    ("USDTRY=X", "usdtry"),
    ("GBPJPY=X", "gbpjpy"),
    ("EURTRY=X", "eurtry"),
    ("ZAR=X",    "usdzar"),
    ("MXN=X",    "usdmxn"),
    ("BRL=X",    "usdbrl"),
]


def fetch_one(ticker: str) -> pd.DataFrame:
    """Download one FX ticker 1h OHLCV for the maximum yfinance window."""
    import yfinance as yf

    print(f"[fetch_forex] downloading {ticker} 1h bars …", flush=True)
    df = yf.download(
        tickers=ticker,
        interval="1h",
        period="max",            # ~730 days for 1h interval
        progress=False,
        auto_adjust=False,
    )
    if df is None or len(df) == 0:
        print(f"[fetch_forex] {ticker}: NO DATA from yfinance", flush=True)
        return pd.DataFrame()

    # yfinance returns either MultiIndex or flat columns depending on version
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
    print(f"[fetch_forex] {ticker}: {len(df)} bars "
          f"({df.index.min().date()} → {df.index.max().date()})", flush=True)
    return df


def sanity_check(ticker: str, slug: str, df: pd.DataFrame) -> dict:
    """Return a per-asset sanity summary dict for printing."""
    if len(df) == 0:
        return {
            "ticker": ticker, "slug": slug, "rows": 0,
            "first": None, "last": None,
            "close_min": None, "close_max": None, "close_mean": None,
        }
    return {
        "ticker": ticker,
        "slug": slug,
        "rows": len(df),
        "first": df.index.min().isoformat(),
        "last": df.index.max().isoformat(),
        "close_min": float(df["close"].min()),
        "close_max": float(df["close"].max()),
        "close_mean": float(df["close"].mean()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker, slug in PAIRS:
        df = fetch_one(ticker)
        summary = sanity_check(ticker, slug, df)
        rows.append(summary)

        if len(df) > 0:
            out = OUT_DIR / f"{slug}_1h_real.csv"
            df.to_csv(out, index_label="datetime")
            print(f"[fetch_forex] saved -> {out}", flush=True)
        print()

    # Combined sanity table
    print("=" * 78)
    print("SANITY CHECK — FOREX 1H BARS")
    print("=" * 78)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    # Asset-class sanity hints (expected approximate ranges from Aug 2024 - Aug 2026)
    print("\n[fetch_forex] Rough sanity bands (Aug 2024 - Aug 2026):")
    print("  USDTRY   ~33   – 41")
    print("  GBPJPY   ~180  – 215")
    print("  EURTRY   ~36   – 46")
    print("  USDZAR   ~17   – 19")
    print("  USDMXN   ~17   – 21")
    print("  USDBRL   ~4.7  – 6.3")
    print("  (Anything wildly off → bad ticker / yfinance column quirk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())