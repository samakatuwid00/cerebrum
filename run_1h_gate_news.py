"""Vectorized backtest with FULL feature stack + economic-calendar proximity.

This is the FINAL attempt on Gold per the user's request:
  * Loads XAU/USD 1h real data (yfinance)
  * Loads DXY (US Dollar Index) 1h data — cross-asset causal feature
  * Loads the hardcoded economic calendar (FOMC/NFP/CPI/PPI 2024-2026)
  * Computes per-(hour, day_of_week) statistical edge
  * For each bar: probability = full feature stack
                   + DXY cross-asset nudge (inverse to gold)
                   + news proximity pullback toward 0.5 near high-impact events

The news pullback is the new feature:
    * Within ±4h of an event, prob is pulled 20% toward 0.5
    * 4-24h out, pulled 5% toward 0.5
    * Outside 24h, no effect

Goal: improve the hit_rate gate (>= 54% on >= 100 trades) by FILTERING OUT
the unpredictable event window rather than trying to predict direction
inside it.

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
    combine_probability_news,
    load_economic_calendar,
    compute_news_proximity,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GATE_HIT_RATE = 0.54
GATE_MIN_TRADES = 100
WARMUP_BARS = 1000

# Feature nudge magnitudes (kept from previous full-feature runner)
RSI_BOOST = 0.05
BB_BOOST = 0.05
MOM_BOOST = 0.05
DXY_BOOST = 0.12
DXY_SCALE = 50.0

# News proximity knobs
NEAR_H = 4.0
FAR_H = 24.0
NEAR_PULLBACK = 0.20
MID_PULLBACK = 0.05


def run_backtest_news_vectorized(threshold: float, lam: float = 0.3,
                                  warmup_bars: int = WARMUP_BARS):
    """Vectorized walk-forward with full stack + news pullback.

    Returns (trades_df, summary_dict). trades_df is empty if no trades fired.
    """
    df = load_history("xau_usd_1h_real.csv")
    print(f"  [news-backtest] Gold: {len(df)} bars "
          f"({df.index.min()} -> {df.index.max()})")

    edge = compute_edge_with_features(df, period=14, min_count=30,
                                       bb_period=20, bb_std=2.0,
                                       mom_period=10)

    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Merge DXY (US Dollar Index) returns
    dxy = load_dxy_history()
    df = merge_dxy_features(df, dxy)
    dxy_nonnull = int(df["dxy_return_1h"].notna().sum())
    print(f"  [news-backtest] DXY merged: {dxy_nonnull} bars with dxy_return_1h")

    # Merge economic calendar proximity
    cal = load_economic_calendar()
    prox = compute_news_proximity(df, cal)
    df["news_proximity_h"] = prox
    print(f"  [news-backtest] Calendar: {len(cal)} events, "
          f"prox range [{prox.min():.1f}h, {prox.max():.1f}h]")

    edge_lookup = edge[["hour", "day_of_week", "mean_ret", "std_ret",
                         "count"]].copy()
    edge_lookup = edge_lookup.set_index(["hour", "day_of_week"])

    df_join = df.set_index(["hour", "day_of_week"], append=True)
    df = df_join.join(edge_lookup, how="left").reset_index(
        level=["hour", "day_of_week"]).sort_index()

    # Apply news-aware combiner row-wise
    def _apply(row):
        return combine_probability_news(
            row,
            rsi=row.get("rsi", 50.0),
            bollinger_pos=row.get("bollinger_pos", 0.5),
            momentum=row.get("momentum", 0.0),
            dxy_return_1h=row.get("dxy_return_1h", 0.0),
            dxy_return_4h=row.get("dxy_return_4h", 0.0),
            dxy_return_24h=row.get("dxy_return_24h", 0.0),
            news_proximity_h=row.get("news_proximity_h", np.nan),
            lam=lam, sentiment=0.0,
            rsi_boost=RSI_BOOST, bb_boost=BB_BOOST, mom_boost=MOM_BOOST,
            dxy_boost=DXY_BOOST, dxy_scale=DXY_SCALE,
            near_h=NEAR_H, far_h=FAR_H,
            near_pullback=NEAR_PULLBACK, mid_pullback=MID_PULLBACK,
        )

    df["prob_up_base"] = df.apply(_apply, axis=1)
    df["prob_dn"] = 1.0 - df["prob_up_base"]
    df["prob"] = df[["prob_up_base", "prob_dn"]].max(axis=1)
    df["side"] = np.where(df["prob_up_base"] >= df["prob_dn"], "BUY", "SELL")

    mask = (
        (df.index >= df.index[warmup_bars])
        & df["prob"].notna()
        & df["ret"].notna()
        & df["dxy_return_1h"].notna()
        & df["dxy_return_4h"].notna()
        & df["dxy_return_24h"].notna()
        & df["news_proximity_h"].notna()
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
         "news_proximity_h", "prob_up_base", "next_ret", "win"]
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
    print("CEREBRUM 1H BACKTEST — FULL STACK + ECONOMIC CALENDAR (XAU/USD)")
    print(f"Features: hour + RSI({RSI_BOOST}) + BB({BB_BOOST}) + Mom({MOM_BOOST}) "
          f"+ DXY({DXY_BOOST}) + News pullback ({NEAR_PULLBACK} near, "
          f"{MID_PULLBACK} mid, ±{NEAR_H}h/{FAR_H}h windows)")
    print("=" * 78)

    rows = []
    gate_pass = None
    best = None
    for thresh in THRESHOLDS:
        trades_df, summary = run_backtest_news_vectorized(threshold=thresh)
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
        if s["total_trades"] >= GATE_MIN_TRADES:
            if best is None or s["hit_rate"] > best["hit_rate"]:
                best = {"threshold": thresh, "hit_rate": s["hit_rate"],
                        "total_trades": s["total_trades"]}

    res = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("XAU/USD 1H BACKTEST — FULL STACK + ECONOMIC CALENDAR — VECTORIZED")
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
        return 0
    else:
        if best is not None:
            print(">>> VERDICT: GATE FAILED — no threshold cleared "
                  "hit_rate>=54% with >=100 trades.")
            print(f">>> Best observed: threshold={best['threshold']}  "
                  f"hit_rate={best['hit_rate']:.2%}  "
                  f"trades={int(best['total_trades'])}")
        else:
            print(">>> VERDICT: GATE FAILED — no threshold produced >=100 trades.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())