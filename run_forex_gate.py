"""Vectorized backtest of the FULL-feature model on each forex asset.

Mirrors run_1h_gate_full.py but:
  * takes the asset from --asset arg (slug used in data/{slug}_1h_real.csv)
  * drops the hardcoded "xau_usd_1h_real.csv" path
  * prints a per-asset row summary table at the end

Features per bar (leak-safe at bar open):
  * RSI(14)         - Wilder RSI, oversold/overbought bias
  * Bollinger(20,2) - mean-reversion band position in [0, 1]
  * Momentum(10)    - simple N-bar percentage return (trend continuation)

Gate (per task spec): hit_rate >= 54% AND total_trades >= 100 at ANY threshold.

Run:
    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe run_forex_gate.py
    # or pick one:
    PYTHONPATH=. .venv_cerebrum\\Scripts\\python.exe run_forex_gate.py --asset usdtry
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent  # repo root = this script's dir
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.pipeline import (
    _load_csv,
    compute_rsi,
    compute_bollinger_position,
    compute_momentum,
    compute_edge_with_features,
    combine_probability_full,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GATE_HIT_RATE = 0.54
GATE_MIN_TRADES = 100
RSI_BOOST = 0.05
BB_BOOST = 0.05
MOM_BOOST = 0.05
WARMUP_BARS = 1000

DATA = ROOT / "data"

ASSETS = [
    ("USDTRY=X", "usdtry", "US Dollar / Turkish Lira"),
    ("GBPJPY=X", "gbpjpy", "British Pound / Japanese Yen"),
    ("EURTRY=X", "eurtry", "Euro / Turkish Lira"),
    ("ZAR=X",    "usdzar", "USD / South African Rand"),
    ("MXN=X",    "usdmxn", "USD / Mexican Peso"),
    ("BRL=X",    "usdbrl", "USD / Brazilian Real"),
]


def load_asset(slug: str) -> pd.DataFrame:
    """Load ``data/{slug}_1h_real.csv`` and return tz-aware UTC-indexed frame."""
    path = DATA / f"{slug}_1h_real.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run data/fetch_forex.py first."
        )
    # _load_csv() in src.pipeline resolves ROOT from its own __file__ (wrong
    # directory when imported by name from a different module), so we use
    # pd.read_csv directly and apply the same tz-aware index logic.
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def run_backtest_full_vectorized(df: pd.DataFrame, threshold: float,
                                  lam: float = 0.3,
                                  warmup_bars: int = WARMUP_BARS,
                                  rsi_boost: float = RSI_BOOST,
                                  bb_boost: float = BB_BOOST,
                                  mom_boost: float = MOM_BOOST):
    """Vectorized walk-forward with per-bar RSI + Bollinger + Momentum nudge."""
    edge = compute_edge_with_features(df, period=14, min_count=30,
                                       bb_period=20, bb_std=2.0,
                                       mom_period=10)

    rsi_series = compute_rsi(df["close"], period=14)
    bb_series = compute_bollinger_position(df["close"], period=20, std_dev=2.0)
    mom_series = compute_momentum(df["close"], period=10)
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["rsi"] = rsi_series
    df["bollinger_pos"] = bb_series
    df["momentum"] = mom_series

    edge_lookup = edge[["hour", "day_of_week", "mean_ret", "std_ret",
                         "count"]].copy()
    edge_lookup = edge_lookup.set_index(["hour", "day_of_week"])

    df_join = df.set_index(["hour", "day_of_week"], append=True)
    df = df_join.join(edge_lookup, how="left").reset_index(
        level=["hour", "day_of_week"]).sort_index()

    def _apply(row):
        return combine_probability_full(
            row,
            rsi=float(row["rsi"]) if pd.notna(row["rsi"]) else 50.0,
            bollinger_pos=float(row["bollinger_pos"]) if pd.notna(row["bollinger_pos"]) else 0.5,
            momentum=float(row["momentum"]) if pd.notna(row["momentum"]) else 0.0,
            lam=lam, sentiment=0.0,
            rsi_boost=rsi_boost, bb_boost=bb_boost, mom_boost=mom_boost,
        )

    df["prob_up_base"] = df.apply(_apply, axis=1)
    df["prob_dn"] = 1.0 - df["prob_up_base"]
    df["prob"] = df[["prob_up_base", "prob_dn"]].max(axis=1)
    df["side"] = np.where(df["prob_up_base"] >= df["prob_dn"], "BUY", "SELL")

    mask = (
        (df.index >= df.index[warmup_bars])
        & df["prob"].notna()
        & df["ret"].notna()
        & df["rsi"].notna()
        & df["bollinger_pos"].notna()
        & df["momentum"].notna()
        & (df["prob"] >= threshold)
    )
    trade_bars = df.loc[mask].copy()
    if len(trade_bars) == 0:
        return pd.DataFrame(), {
            "threshold": threshold,
            "total_trades": 0, "wins": 0, "losses": 0,
            "hit_rate": float("nan"),
            "avg_return_bps": float("nan"),
            "total_pnl_bps": float("nan"),
            "sharpe_annualized": float("nan"),
            "max_drawdown_bps": float("nan"),
        }

    trade_bars["next_ret"] = trade_bars["ret"].shift(-1)
    trade_bars = trade_bars[trade_bars["next_ret"].notna()].copy()
    trade_bars["win"] = np.where(
        trade_bars["side"] == "BUY",
        trade_bars["next_ret"] > 0,
        trade_bars["next_ret"] < 0,
    ).astype(int)

    trades_df = trade_bars[
        ["prob", "side", "mean_ret", "std_ret", "count", "rsi",
         "bollinger_pos", "momentum", "prob_up_base", "next_ret", "win"]
    ].copy()
    trades_df.index = trade_bars.index
    trades_df = trades_df.reset_index().rename(columns={"datetime": "timestamp"})
    trades_df["timestamp"] = trades_df["timestamp"].apply(lambda x: x.isoformat())

    total = len(trades_df)
    wins = int(trades_df["win"].sum())
    hit_rate = wins / total
    avg_ret = float(trades_df["next_ret"].mean())
    trades_df["pnl_bps"] = trades_df["next_ret"] * 10000
    total_pnl_bps = float(trades_df["pnl_bps"].sum())
    std_ret = float(trades_df["next_ret"].std())
    bars_per_year = 252 * 24
    sharpe = (avg_ret / std_ret * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0
    cumulative = trades_df["pnl_bps"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    return trades_df, {
        "threshold": threshold,
        "total_trades": int(total),
        "wins": wins,
        "losses": int(total - wins),
        "hit_rate": float(hit_rate),
        "avg_return_bps": avg_ret * 10000,
        "total_pnl_bps": total_pnl_bps,
        "sharpe_annualized": float(sharpe),
        "max_drawdown_bps": max_dd,
    }


def run_asset(ticker: str, slug: str, label: str) -> tuple[str, dict]:
    """Run the full threshold sweep for one asset, return (slug, summary_dict).

    The summary_dict has keys: asset, label, ticker, rows, first, last,
    plus one entry per threshold (`thr_{thr}` → row dict).
    """
    print()
    print("=" * 78)
    print(f"ASSET: {label} ({ticker})  file=data/{slug}_1h_real.csv")
    print("=" * 78)
    try:
        df = load_asset(slug)
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        return slug, {"asset": slug, "label": label, "ticker": ticker,
                      "rows": 0, "first": None, "last": None,
                      "error": str(e)}

    print(f"  bars: {len(df)}  range: {df.index.min()} -> {df.index.max()}")

    summary = {
        "asset": slug,
        "label": label,
        "ticker": ticker,
        "rows": len(df),
        "first": df.index.min().isoformat(),
        "last": df.index.max().isoformat(),
    }
    best_hit = None
    best_thr = None

    for thresh in THRESHOLDS:
        trades_df, s = run_backtest_full_vectorized(df, threshold=thresh)
        summary[f"thr_{thresh}"] = s
        if s["total_trades"] == 0:
            print(f"  threshold={thresh}: no trades")
            continue
        passed = (s["hit_rate"] >= GATE_HIT_RATE) and (
            s["total_trades"] >= GATE_MIN_TRADES)
        gate = "PASS" if passed else "FAIL"
        print(f"  threshold={thresh}: trades={s['total_trades']:5d}  "
                f"hit_rate={s['hit_rate']:.2%}  "
                f"avg={s['avg_return_bps']:+.2f}bps  "
                f"pnl={s['total_pnl_bps']:+.0f}bps  "
                f"sharpe={s['sharpe_annualized']:+.2f}  "
                f"maxDD={s['max_drawdown_bps']:.0f}bps  "
                f"-> {gate}")
        # Track best across thresholds that have >= 100 trades
        if s["total_trades"] >= GATE_MIN_TRADES:
            if best_hit is None or s["hit_rate"] > best_hit:
                best_hit = s["hit_rate"]
                best_thr = thresh
                summary["best_thr"] = thresh
                summary["best_hit_rate"] = s["hit_rate"]
                summary["best_trades"] = s["total_trades"]
                summary["best_pnl_bps"] = s["total_pnl_bps"]
                summary["best_sharpe"] = s["sharpe_annualized"]

    summary["verdict"] = (
        f"GATE PASS at thr={summary['best_thr']} "
        f"(hit={summary['best_hit_rate']:.2%}, n={summary['best_trades']})"
        if summary.get("best_hit_rate", 0) >= GATE_HIT_RATE
        else f"GATE FAIL — best thr={summary.get('best_thr')} "
             f"hit={summary.get('best_hit_rate', 0):.2%} "
             f"n={summary.get('best_trades', 0)}"
        if summary.get("best_thr") is not None
        else "GATE FAIL — no threshold produced >=100 trades"
    )
    return slug, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None,
                        help="Run only one slug (e.g. 'usdtry'). "
                             "Default = all six.")
    args = parser.parse_args()

    selected = ASSETS
    if args.asset:
        selected = [a for a in ASSETS if a[1] == args.asset]
        if not selected:
            print(f"Unknown asset '{args.asset}'. "
                  f"Valid: {[a[1] for a in ASSETS]}")
            return 2

    all_summaries: dict[str, dict] = {}
    for ticker, slug, label in selected:
        slug, summary = run_asset(ticker, slug, label)
        all_summaries[slug] = summary

    # Build comparison table — one row per (asset × threshold)
    rows = []
    for slug, summary in all_summaries.items():
        for thresh in THRESHOLDS:
            s = summary.get(f"thr_{thresh}", {})
            rows.append({
                "asset": slug,
                "label": summary.get("label", ""),
                "threshold": thresh,
                "trades": s.get("total_trades", 0),
                "wins": s.get("wins", 0),
                "losses": s.get("losses", 0),
                "hit_rate": s.get("hit_rate", float("nan")),
                "avg_bps": s.get("avg_return_bps", float("nan")),
                "pnl_bps": s.get("total_pnl_bps", float("nan")),
                "sharpe": s.get("sharpe_annualized", float("nan")),
                "max_dd_bps": s.get("max_drawdown_bps", float("nan")),
            })
    res = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("CROSS-ASSET COMPARISON — FULL FEATURE MODEL (RSI + BB + MOM)")
    print("=" * 78)
    print(res.to_string(
        index=False,
        float_format=lambda v: (f"{v:.4f}" if isinstance(v, float) and v == v
                                else str(v)),
    ))

    # Per-asset best summary
    print()
    print("=" * 78)
    print("PER-ASSET BEST (across thresholds with >= 100 trades)")
    print("=" * 78)
    best_rows = []
    for slug, summary in all_summaries.items():
        best_rows.append({
            "asset": slug,
            "label": summary.get("label", ""),
            "rows": summary.get("rows", 0),
            "best_thr": summary.get("best_thr"),
            "best_hit_rate": summary.get("best_hit_rate"),
            "best_trades": summary.get("best_trades"),
            "best_pnl_bps": summary.get("best_pnl_bps"),
            "best_sharpe": summary.get("best_sharpe"),
        })
    best_df = pd.DataFrame(best_rows)
    print(best_df.to_string(
        index=False,
        float_format=lambda v: (f"{v:.4f}" if isinstance(v, float) and v == v
                                else str(v)),
    ))

    # Verdict
    print()
    print("=" * 78)
    print("VERDICT — 54% / 100-TRADE GATE")
    print("=" * 78)
    winners = [r for r in best_rows
               if r.get("best_hit_rate") is not None
               and r["best_hit_rate"] >= GATE_HIT_RATE]
    if winners:
        # Pick winner with highest hit_rate * log(trades) tie-break
        winners.sort(key=lambda r: (r["best_hit_rate"], r["best_trades"]),
                     reverse=True)
        w = winners[0]
        print(f">>> WINNER: {w['asset']} ({w['label']})")
        print(f">>>   thr={w['best_thr']}  hit_rate={w['best_hit_rate']:.2%}  "
              f"trades={w['best_trades']}  pnl={w['best_pnl_bps']:+.0f}bps  "
              f"sharpe={w['best_sharpe']:+.2f}")
        print()
        print("All passing assets (>= 54% hit rate, >= 100 trades):")
        for w in winners:
            print(f"  {w['asset']:8s}  thr={w['best_thr']}  "
                  f"hit={w['best_hit_rate']:.2%}  "
                  f"n={w['best_trades']}  pnl={w['best_pnl_bps']:+.0f}bps")
    else:
        # None passed; report best asset
        valid = [r for r in best_rows if r.get("best_hit_rate") is not None]
        if valid:
            valid.sort(key=lambda r: r["best_hit_rate"], reverse=True)
            b = valid[0]
            print(">>> NO asset passed 54%/100-trade gate.")
            print(f">>> Best asset: {b['asset']} ({b['label']})")
            print(f">>>   hit_rate={b['best_hit_rate']:.2%}  "
                  f"trades={b['best_trades']}  "
                  f"thr={b['best_thr']}  "
                  f"pnl={b['best_pnl_bps']:+.0f}bps  "
                  f"sharpe={b['best_sharpe']:+.2f}")
        else:
            print(">>> NO asset produced >=100 trades at any threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())