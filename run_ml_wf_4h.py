import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.ml_edge import walkforward_model

fname = "xau_usd_4h_real.csv"
print(f"=== {fname} meta-labeling walk-forward ===")
for horizon in [1, 2, 4]:
    for thr in [0.50, 0.53]:
        t, s = walkforward_model(filename=fname, threshold=thr,
                                 horizon=horizon, mode="meta", min_train=500)
        if s:
            gate = "PASS" if s["hit_rate"] >= 0.53 and s["total_trades"] >= 200 else "fail"
            print(f"  h={horizon} thr={thr}: hit={s['hit_rate']:.2%} trades={s['total_trades']} "
                  f"pnl={s['total_pnl_bps']:.0f}bps sharpe={s['sharpe_annualized']:.2f} "
                  f"maxDD={s['max_drawdown_bps']:.0f}bps [{gate}] | baseline "
                  f"hit={s['baseline_hit_rate']:.2%} pnl={s['baseline_pnl_bps']:.0f}bps")
        else:
            print(f"  h={horizon} thr={thr}: no trades")
