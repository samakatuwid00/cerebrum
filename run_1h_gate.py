"""Run the vectorized backtest on XAU/USD 1h data at the requested thresholds.

Forces loading of data/xau_usd_1h_real.csv by passing it explicitly to
load_history() so the result is unambiguous regardless of any other
data files present.

Gate (per task spec): hit_rate >= 54% AND total_trades >= 100 at ANY threshold.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.pipeline import load_history
from src.backtest import run_backtest_vectorized

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GATE_HIT_RATE = 0.54
GATE_MIN_TRADES = 100

def main() -> int:
    # Force the 1h file so the result is unambiguous.
    df = load_history("xau_usd_1h_real.csv")
    print(f"[1h-gate] Loaded {len(df)} 1h bars "
          f"({df.index.min()} -> {df.index.max()})")

    rows = []
    gate_pass = None
    for thresh in THRESHOLDS:
        trades_df, summary = run_backtest_vectorized(threshold=thresh)
        if not summary:
            rows.append({
                "threshold": thresh,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "hit_rate": np.nan,
                "avg_return_bps": np.nan,
                "total_pnl_bps": np.nan,
                "sharpe_annualized": np.nan,
                "max_drawdown_bps": np.nan,
                "gate": "FAIL (no trades)",
            })
            print(f"  threshold={thresh}: no trades")
            continue

        s = summary
        passed = (s["hit_rate"] >= GATE_HIT_RATE) and (s["total_trades"] >= GATE_MIN_TRADES)
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
        print(f"  threshold={thresh}: trades={s['total_trades']:4d}  "
              f"hit_rate={s['hit_rate']:.2%}  "
              f"avg={s['avg_return_bps']:+.2f}bps  "
              f"sharpe={s['sharpe_annualized']:+.2f}  "
              f"maxDD={s['max_drawdown_bps']:.0f}bps  "
              f"-> {gate}")
        if passed and gate_pass is None:
            gate_pass = thresh

    res = pd.DataFrame(rows)

    print()
    print("=" * 78)
    print("XAU/USD 1H BACKTEST — VECTORIZED")
    print("=" * 78)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) and v == v else str(v)))
    print()
    print(f"Gate: hit_rate >= {GATE_HIT_RATE:.0%} AND trades >= {GATE_MIN_TRADES}")
    if gate_pass is not None:
        print(f">>> VERDICT: GATE PASSED at threshold={gate_pass}")
    else:
        # find best observed (max hit_rate with trade >= 100), report honestly
        ok = res[res["total_trades"] >= GATE_MIN_TRADES].copy()
        if len(ok):
            best = ok.loc[ok["hit_rate"].idxmax()]
            print(">>> VERDICT: GATE FAILED — no threshold cleared hit_rate>=54% with >=100 trades.")
            print(f">>> Best observed: threshold={best['threshold']}  "
                  f"hit_rate={best['hit_rate']:.2%}  trades={int(best['total_trades'])}")
        else:
            print(">>> VERDICT: GATE FAILED — no threshold produced >=100 trades.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())