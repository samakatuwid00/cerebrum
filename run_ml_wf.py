import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.ml_edge import walkforward_model

fname = "xau_usd_1h_real.csv"
horizon = 4
print(f"=== {fname} GBM walk-forward, horizon={horizon}, feature-pruned ===")
for mode in ["global", "regime", "meta"]:
    for thr in ([0.53, 0.56] if mode != "meta" else [0.50, 0.53]):
        t, s = walkforward_model(filename=fname, threshold=thr, horizon=horizon, mode=mode)
        if s:
            gate = "PASS" if s["hit_rate"] >= 0.53 and s["total_trades"] >= 200 else "fail"
            print(f"  [{mode:>6}] thr={thr}: hit={s['hit_rate']:.2%} trades={s['total_trades']} "
                  f"pnl={s['total_pnl_bps']:.0f}bps sharpe={s['sharpe_annualized']:.2f} "
                  f"maxDD={s['max_drawdown_bps']:.0f}bps [{gate}] | baseline "
                  f"hit={s['baseline_hit_rate']:.2%} pnl={s['baseline_pnl_bps']:.0f}bps")
        else:
            print(f"  [{mode:>6}] thr={thr}: no trades")

print("\n=== Horizon sweep, best mode ===")
horizon = None  # reset in loop
for horizon in [4, 8]:
    for mode in ["regime"]:
        t, s = walkforward_model(filename=fname, threshold=0.53,
                                 horizon=horizon, mode=mode)
        if s:
            gate = "PASS" if s["hit_rate"] >= 0.53 and s["total_trades"] >= 200 else "fail"
            print(f"  h={horizon} [{mode}] thr=0.53: hit={s['hit_rate']:.2%} "
                  f"trades={s['total_trades']} pnl={s['total_pnl_bps']:.0f}bps "
                  f"sharpe={s['sharpe_annualized']:.2f} "
                  f"maxDD={s['max_drawdown_bps']:.0f}bps [{gate}] | baseline "
                  f"hit={s['baseline_hit_rate']:.2%} pnl={s['baseline_pnl_bps']:.0f}bps")
