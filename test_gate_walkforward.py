"""Test harness: verify improved hit rate + walk-forward for cerebrum bot.

Runs standard and MTF modes, vectorized and true walk-forward,
across the available XAU/USD datasets. Prints a compact scoreboard.
"""
import sys, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest_vectorized, run_backtest_walkforward

THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
GATE_HIT, GATE_TRADES = 0.53, 200


def report(tag, summary):
    if not summary:
        print(f"  {tag:<38}  NO TRADES")
        return None
    gate = "PASS" if (summary["hit_rate"] >= GATE_HIT and summary["total_trades"] >= GATE_TRADES) else "fail"
    print(f"  {tag:<38}  hit={summary['hit_rate']:.2%}  trades={summary['total_trades']:>5}  "
          f"pnl={summary['total_pnl_bps']:>8.0f}bps  sharpe={summary['sharpe_annualized']:>6.2f}  "
          f"maxDD={summary['max_drawdown_bps']:>7.0f}bps  [{gate}]")
    return summary


def sweep(fn, filename, use_mtf, label):
    best = None
    for thr in THRESHOLDS:
        t0 = time.time()
        trades, summary = fn(threshold=thr, filename=filename, use_mtf=use_mtf)
        dt = time.time() - t0
        s = report(f"{label} thr={thr:.2f} ({dt:.0f}s)", summary)
        if s and (best is None or s["hit_rate"] > best["hit_rate"]):
            best = s
    return best


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "vectorized"):
        print("=" * 100)
        print("VECTORIZED (in-sample edge, slight lookahead) — upper bound on hit rate")
        print("=" * 100)
        for fn_name, fname in [("xau_usd_1h_real.csv", "1h"), ("xau_usd_4h_real.csv", "4h")]:
            print(f"\n--- XAU/USD {fname} standard ---")
            sweep(run_backtest_vectorized, fn_name, False, f"vec-{fname}-std")
            print(f"\n--- XAU/USD {fname} MTF (weekly trend + ATR + MACD) ---")
            sweep(run_backtest_vectorized, fn_name, True, f"vec-{fname}-mtf")

    if mode in ("all", "walkforward"):
        print("\n" + "=" * 100)
        print("TRUE WALK-FORWARD (rolling 10k-bar window, zero lookahead) — the honest number")
        print("=" * 100)
        for fn_name, fname in [("xau_usd_1h_real.csv", "1h"), ("xau_usd_4h_real.csv", "4h")]:
            for thr in [0.55, 0.60, 0.65]:
                for mtf in (False, True):
                    tag = f"wf-{fname}-{'mtf' if mtf else 'std'} thr={thr:.2f}"
                    t0 = time.time()
                    trades, summary = run_backtest_walkforward(
                        threshold=thr, filename=fn_name, use_mtf=mtf)
                    dt = time.time() - t0
                    report(f"{tag} ({dt:.0f}s)", summary)
