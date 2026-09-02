"""Vectorized backtest of the DXY cross-asset model on XAU/USD 1h data.

Adds the US Dollar Index (DXY) as a CAUSAL cross-asset feature to the
Sharpe-logistic base. DXY moves inversely to gold; this should provide edge
that simple price-derived features (RSI/BB/momentum) cannot.

Features (per bar, leak-safe at bar open):
  * dxy_return_1h  — DXY 1h percentage return, scalar = 50
  * dxy_return_4h  — DXY 4h percentage return, scalar = 50
  * dxy_return_24h — DXY 24h percentage return, scalar = 50
  Each contributes +/- dxy_boost/3 ≈ +/-0.04, bounded by +/- dxy_boost=0.12.

Gate (per task spec): hit_rate >= 54% AND total_trades >= 100 at ANY threshold.

HARD GATE: if no threshold clears both conditions, the verdict is FAIL and
we report the best observed threshold honestly.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.pipeline import (
    load_history,
    load_dxy_history,
    merge_dxy_features,
    compute_edge_with_features,
    combine_probability_dxy,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GATE_HIT_RATE = 0.54
GATE_MIN_TRADES = 100
DXY_BOOST = 0.12          # total max nudge = +/- 0.12
DXY_SCALE = 50.0
WARMUP_BARS = 1000


def run_backtest_dxy_vectorized(threshold: float, lam: float = 0.3,
                                 warmup_bars: int = WARMUP_BARS,
                                 dxy_boost: float = DXY_BOOST,
                                 dxy_scale: float = DXY_SCALE):
    """Vectorized walk-forward with DXY cross-asset nudger.

    The base probability comes from compute_edge_with_features() (Sharpe-logistic
    on per-(hour, day_of_week) slot). We then add a DXY-return nudge that is
    signed negatively (DXY up → gold prob down).
    """
    df = load_history("xau_usd_1h_real.csv")
    print(f"[dxy-backtest] Gold: {len(df)} bars "
            f"({df.index.min()} -> {df.index.max()})")

    edge = compute_edge_with_features(df, period=14, min_count=30,
                                       bb_period=20, bb_std=2.0,
                                       mom_period=10)

    # Per-bar features
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Merge DXY (US Dollar Index) returns — leak-safe via asof backward
    dxy = load_dxy_history()
    df = merge_dxy_features(df, dxy)
    dxy_nonnull = int(df["dxy_return_1h"].notna().sum())
    print(f"[dxy-backtest] DXY merged: {dxy_nonnull} bars with dxy_return_1h")

    edge_lookup = edge[["hour", "day_of_week", "mean_ret", "std_ret",
                         "count"]].copy()
    edge_lookup = edge_lookup.set_index(["hour", "day_of_week"])

    df_join = df.set_index(["hour", "day_of_week"], append=True)
    df = df_join.join(edge_lookup, how="left").reset_index(
        level=["hour", "day_of_week"]).sort_index()

    # Apply DXY nudger row-wise (leak-safe: DXY return at bar t is observable
    # at the OPEN of bar t; we trade on gold return from t -> t+1)
    def _apply(row):
        return combine_probability_dxy(
            row,
            dxy_return_1h=row.get("dxy_return_1h", 0.0),
            dxy_return_4h=row.get("dxy_return_4h", 0.0),
            dxy_return_24h=row.get("dxy_return_24h", 0.0),
            lam=lam, sentiment=0.0,
            dxy_boost=dxy_boost, dxy_scale=dxy_scale,
        )

    df["prob_up_base"] = df.apply(_apply, axis=1)
    df["prob_dn"] = 1.0 - df["prob_up_base"]
    df["prob"] = df[["prob_up_base", "prob_dn"]].max(axis=1)
    df["side"] = np.where(df["prob_up_base"] >= df["prob_dn"], "BUY", "SELL")

    mask = (
        (df.index >= df.index[warmup_bars])
        & df["prob"].notna()
        & df["ret"].notna()
        & df["dxy_return_1h"].notna()    # require DXY feature
        & df["dxy_return_4h"].notna()
        & df["dxy_return_24h"].notna()
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
        ["prob", "side", "mean_ret", "std_ret", "count",
         "dxy_return_1h", "dxy_return_4h", "dxy_return_24h",
         "prob_up_base", "next_ret", "win"]
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


def main() -> int:
    print("=" * 78)
    print("CEREBRUM 1H BACKTEST — DXY CROSS-ASSET MODEL (XAU/USD vs DX-Y.NYB)")
    print(f"Nudge budget: +/- {DXY_BOOST} (3 horizons × +/- {DXY_BOOST/3:.4f})")
    print("=" * 78)

    rows = []
    gate_pass = None
    best = None
    for thresh in THRESHOLDS:
        trades_df, summary = run_backtest_dxy_vectorized(threshold=thresh)
        s = summary
        if s["total_trades"] == 0:
            rows.append({
                "threshold": thresh, "total_trades": 0, "wins": 0,
                "losses": 0, "hit_rate": np.nan, "avg_return_bps": np.nan,
                "total_pnl_bps": np.nan, "sharpe_annualized": np.nan,
                "max_drawdown_bps": np.nan, "gate": "FAIL (no trades)",
            })
            print(f"  threshold={thresh}: no trades")
            continue
        passed = (s["hit_rate"] >= GATE_HIT_RATE) and (
            s["total_trades"] >= GATE_MIN_TRADES)
        gate = "PASS" if passed else "FAIL"
        rows.append({
            "threshold": thresh,
            "total_trades": s["total_trades"],
            "wins": s["wins"],
            "losses": s["losses"],
            "hit_rate": s["hit_rate"],
            "avg_return_bps": s["avg_return_bps"],
            "total_pnl_bps": s["total_pnl_bps"],
            "sharpe_annualized": s["sharpe_annualized"],
            "max_drawdown_bps": s["max_drawdown_bps"],
            "gate": gate,
        })
        print(f"  threshold={thresh}: trades={s['total_trades']:5d}  "
                f"hit_rate={s['hit_rate']:.2%}  "
                f"avg={s['avg_return_bps']:+.2f}bps  "
                f"pnl={s['total_pnl_bps']:+.0f}bps  "
                f"sharpe={s['sharpe_annualized']:+.2f}  "
                f"maxDD={s['max_drawdown_bps']:.0f}bps  "
                f"-> {gate}")
        if passed and gate_pass is None:
            gate_pass = thresh
        # Track best (only thresholds that have >= 100 trades)
        if s["total_trades"] >= GATE_MIN_TRADES:
            if best is None or s["hit_rate"] > best["hit_rate"]:
                best = {"threshold": thresh, "hit_rate": s["hit_rate"],
                        "total_trades": s["total_trades"]}

    res = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("XAU/USD 1H BACKTEST — DXY CROSS-ASSET FEATURE — VECTORIZED")
    print("=" * 78)
    print(res.to_string(
        index=False,
        float_format=lambda v: (f"{v:.4f}" if isinstance(v, float) and v == v
                                else str(v)),
    ))
    print()
    print(f"Gate: hit_rate >= {GATE_HIT_RATE:.0%} AND trades >= {GATE_MIN_TRADES}")

    if gate_pass is not None:
        print(f">>> VERDICT: GATE PASSED at threshold={gate_pass}")
    else:
        if best is not None:
            print(">>> VERDICT: GATE FAILED — no threshold cleared "
                    "hit_rate>=54% with >=100 trades.")
            print(f">>> Best observed: threshold={best['threshold']}  "
                    f"hit_rate={best['hit_rate']:.2%}  "
                    f"trades={int(best['total_trades'])}")
        else:
            print(">>> VERDICT: GATE FAILED — no threshold produced >=100 trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())