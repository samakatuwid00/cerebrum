"""Phase 1 — Cost-Aware Walk-Forward Gate.

Single function whose out-of-sample, cost-adjusted expectancy is evaluated
with TRUE expanding-window walk-forward (zero lookahead).

Contracts modelled:
  - IQ binary 1h: win = +80% of stake, loss = -100% of stake
  - Spot gold 1h: PnL = signed_ret - spread_cost (spread both ways)

Gate PASS requires:
  hit_rate >= threshold AND expectancy_after_cost > 0 AND trades >= min_trades
  AND walk-forward within 3pp of in-sample.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field
from typing import Optional

# Ensure project root is on sys.path for `from src.X` imports
_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

from src.pipeline import load_history
from src.ml_edge import build_features, _new_model
from src.backtest import deflated_sharpe_ratio


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #
@dataclass
class CostGateResult:
    contract: str
    total_trades: int
    hits: int
    hit_rate: float
    win_cost: float
    loss_cost: float
    expectancy_per_trade: float
    expectancy_per_trade_bps: float
    profit_factor: float
    sharpe_annualized: float
    max_drawdown_bps: float
    deflated_sharpe: float
    baseline_hit_rate: float
    baseline_expectancy_bps: float
    in_sample_hit_rate: float
    in_sample_expectancy_bps: float
    gate_pass: bool
    gate_details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Contract cost helpers
# --------------------------------------------------------------------------- #
def _contract_costs(contract: str, payout: float = 0.80,
                    spread_bps: float = 20) -> tuple[float, float]:
    """Return (win_reward, loss_cost) for the contract type.

    IQ binary:  win_reward = payout (e.g. 0.80), loss_cost = -1.00
    Spot gold:  win_reward = 1.0 - spread_cost, loss_cost = -1.0 - spread_cost
                (approximated as: PnL = signed_ret - spread_cost, so we track
                 spread_cost separately and the raw return drives the win/loss).
    """
    if contract == "iq_binary":
        return payout, -1.0
    elif contract == "spot_gold":
        spread = spread_bps / 10000.0
        return 1.0 - spread, -1.0 - spread
    else:
        raise ValueError(f"Unknown contract: {contract}")


def _trade_pnl(contract: str, side: str, actual_ret: float,
               payout: float = 0.80, spread_bps: float = 20) -> float:
    """Compute PnL for a single trade (as fraction of stake).

    For IQ binary: binary outcome — win pays payout, loss costs 1.
    For spot gold: signed return minus spread cost.
    """
    if contract == "iq_binary":
        win = (actual_ret > 0) if side == "BUY" else (actual_ret < 0)
        return payout if win else -1.0
    elif contract == "spot_gold":
        spread = spread_bps / 10000.0
        signed = actual_ret if side == "BUY" else -actual_ret
        return signed - spread
    else:
        raise ValueError(f"Unknown contract: {contract}")


# --------------------------------------------------------------------------- #
# Core walk-forward
# --------------------------------------------------------------------------- #
def cost_aware_walkforward(
    filename: str,
    contract: str = "iq_binary",
    payout: float = 0.80,
    spread_bps: float = 20,
    commission_bps: float = 0,
    horizon: int = 1,
    threshold: float = 0.56,
    min_trades: int = 200,
    refit_every: int = 250,
    min_train: int = 1500,
    mode: str = "global",
    max_depth: int = 3,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
) -> CostGateResult:
    """TRUE expanding-window walk-forward with cost-adjusted PnL.

    Train on bars [0:i], predict bar i, record trade, refit every refit_every.
    Never train on bars >= i. Zero lookahead.
    """
    df = load_history(filename)
    X, y, fwd_ret = build_features(df, horizon=horizon)

    feat_cols = list(X.columns)
    regime_col = feat_cols.index("atr_regime") if "atr_regime" in feat_cols else None
    trend_col = feat_cols.index("weekly_trend") if "weekly_trend" in feat_cols else None

    Xv, yv, rv = X.values, y.values, fwd_ret.values
    idx = X.index
    n = len(Xv)

    # --- OOS walk-forward ---
    global_model = None
    regime_models = {}
    last_fit = -10**9
    trades = []

    for i in range(min_train, n - 1):
        if i - last_fit >= refit_every or (mode == "global" and global_model is None):
            if mode == "regime" and regime_col is not None:
                regime_models.clear()
                reg_vals = Xv[:i, regime_col]
                t1 = np.nanquantile(reg_vals, 1 / 3)
                t2 = np.nanquantile(reg_vals, 2 / 3)
                for b in range(3):
                    if b == 0:
                        mask = reg_vals <= t1
                    elif b == 1:
                        mask = (reg_vals > t1) & (reg_vals <= t2)
                    else:
                        mask = reg_vals > t2
                    if mask.sum() < 50:
                        continue
                    m = _new_model(max_depth, n_estimators, learning_rate)
                    m.fit(Xv[:i][mask], yv[:i][mask])
                    regime_models[b] = (t1, t2, m)
            else:
                global_model = _new_model(max_depth, n_estimators, learning_rate)
                global_model.fit(Xv[:i], yv[:i])
            last_fit = i

        # predict
        if mode == "regime" and regime_col is not None and regime_models:
            xg = Xv[i : i + 1]
            t1_val, t2_val, _ = list(regime_models.values())[0][:3]
            b = (
                0 if xg[0, regime_col] <= t1_val
                else (1 if xg[0, regime_col] <= t2_val else 2)
            )
            if b not in regime_models:
                continue
            p_up = regime_models[b][2].predict_proba(xg)[0, 1]
        elif global_model is not None:
            p_up = global_model.predict_proba(Xv[i : i + 1])[0, 1]
        else:
            continue

        # decide side
        if mode == "meta":
            trend = Xv[i, trend_col] if trend_col is not None else 0
            if trend > 0:
                side = "BUY"
            elif trend < 0:
                side = "SELL"
            else:
                continue
            take = (p_up >= threshold) if side == "BUY" else (p_up <= 1 - threshold)
            if not take:
                continue
            prob = p_up if side == "BUY" else 1 - p_up
        else:
            if p_up >= threshold:
                side, prob = "BUY", p_up
            elif p_up <= 1 - threshold:
                side, prob = "SELL", 1 - p_up
            else:
                continue

        actual_ret = rv[i]
        pnl = _trade_pnl(contract, side, actual_ret, payout, spread_bps)
        win = pnl > 0

        trades.append({
            "timestamp": idx[i].isoformat(),
            "side": side,
            "prob": float(prob),
            "actual_ret": float(actual_ret),
            "pnl": float(pnl),
            "win": int(win),
        })

    trades_df = pd.DataFrame(trades)

    # --- In-sample evaluation (same model on full training window) ---
    is_hit, is_exp_bps = _in_sample_eval(
        Xv, yv, rv, min_train, n, refit_every, mode,
        regime_col, trend_col, max_depth, n_estimators, learning_rate,
        threshold, contract, payout, spread_bps,
    )

    # --- Baseline (buy-and-hold next-bar direction on same OOS period) ---
    base_rets = rv[min_train : n - 1]
    baseline_hit = float((base_rets > 0).mean()) if len(base_rets) > 0 else 0.0
    # baseline expectancy (no model, just buy-and-hold): mean signed return
    if contract == "iq_binary":
        baseline_exp_bps = ((baseline_hit * payout) + ((1 - baseline_hit) * (-1.0))) * 10000
    else:
        spread = spread_bps / 10000.0
        baseline_exp_bps = (float(base_rets.mean()) - spread) * 10000 if len(base_rets) > 0 else 0.0

    # --- Aggregate OOS stats ---
    if len(trades_df) == 0:
        return CostGateResult(
            contract=contract, total_trades=0, hits=0, hit_rate=0.0,
            win_cost=0.0, loss_cost=0.0, expectancy_per_trade=0.0,
            expectancy_per_trade_bps=0.0, profit_factor=0.0,
            sharpe_annualized=0.0, max_drawdown_bps=0.0, deflated_sharpe=0.0,
            baseline_hit_rate=baseline_hit, baseline_expectancy_bps=baseline_exp_bps,
            in_sample_hit_rate=is_hit, in_sample_expectancy_bps=is_exp_bps,
            gate_pass=False,
            gate_details={"reason": "no trades"},
        )

    total = len(trades_df)
    hits = int(trades_df["win"].sum())
    hit_rate = hits / total

    win_reward, loss_cost = _contract_costs(contract, payout, spread_bps)
    pnl_arr = trades_df["pnl"].values
    exp_per_trade = float(pnl_arr.mean())
    exp_bps = exp_per_trade * 10000

    # profit factor
    gross_wins = pnl_arr[pnl_arr > 0].sum()
    gross_losses = -pnl_arr[pnl_arr < 0].sum()
    profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    # Sharpe (annualised, assuming 1 trade per bar on 1h data → 252*24 bars/year)
    pnl_std = pnl_arr.std()
    sharpe = float(exp_per_trade / pnl_std * np.sqrt(252 * 24)) if pnl_std > 0 else 0.0

    # Max drawdown
    cum = pnl_arr.cumsum()
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd_bps = float(dd.min()) * 10000

    # Deflated Sharpe
    n_trials = 4 * 3 * 3  # thresholds x modes x horizons = 36 in a sweep
    dstr = deflated_sharpe_ratio(
        sharpe_observed=sharpe,
        n_trials=n_trials,
        n_observations=total,
        skew=float(pd.Series(pnl_arr).skew()),
        kurtosis=float(pd.Series(pnl_arr).kurtosis()) + 3,
    )

    # Gate
    in_sample_gap = abs(hit_rate - is_hit) * 100  # percentage points
    gate_hit = hit_rate >= threshold
    gate_exp = exp_per_trade > 0
    gate_trades = total >= min_trades
    gate_gap = in_sample_gap <= 3.0  # within 3pp of in-sample
    gate_pass = gate_hit and gate_exp and gate_trades and gate_gap

    details = {
        "threshold": threshold,
        "mode": mode,
        "horizon": horizon,
        "in_sample_hit_rate": f"{is_hit:.2%}",
        "oos_hit_rate": f"{hit_rate:.2%}",
        "in_sample_oos_gap_pp": f"{in_sample_gap:.2f}pp",
        "gate_hit_rate": "PASS" if gate_hit else "FAIL",
        "gate_expectancy": "PASS" if gate_exp else "FAIL",
        "gate_trades": "PASS" if gate_trades else "FAIL",
        "gate_in_sample_gap": "PASS" if gate_gap else "FAIL",
    }

    return CostGateResult(
        contract=contract,
        total_trades=total,
        hits=hits,
        hit_rate=hit_rate,
        win_cost=win_reward,
        loss_cost=loss_cost,
        expectancy_per_trade=exp_per_trade,
        expectancy_per_trade_bps=exp_bps,
        profit_factor=profit_factor,
        sharpe_annualized=sharpe,
        max_drawdown_bps=max_dd_bps,
        deflated_sharpe=dstr,
        baseline_hit_rate=baseline_hit,
        baseline_expectancy_bps=baseline_exp_bps,
        in_sample_hit_rate=is_hit,
        in_sample_expectancy_bps=is_exp_bps,
        gate_pass=gate_pass,
        gate_details=details,
    )


# --------------------------------------------------------------------------- #
# In-sample evaluation helper
# --------------------------------------------------------------------------- #
def _in_sample_eval(
    Xv, yv, rv, min_train, n, refit_every, mode,
    regime_col, trend_col, max_depth, n_estimators, learning_rate,
    threshold, contract, payout, spread_bps,
) -> tuple[float, float]:
    """Train on full [0:n-1], evaluate on same window → in-sample hit_rate, exp_bps."""
    if n < min_train + 50:
        return 0.0, 0.0

    # train on full data
    if mode == "regime" and regime_col is not None:
        reg_vals = Xv[:, regime_col]
        t1 = np.nanquantile(reg_vals, 1 / 3)
        t2 = np.nanquantile(reg_vals, 2 / 3)
        models = {}
        for b in range(3):
            if b == 0:
                mask = reg_vals <= t1
            elif b == 1:
                mask = (reg_vals > t1) & (reg_vals <= t2)
            else:
                mask = reg_vals > t2
            if mask.sum() < 50:
                continue
            m = _new_model(max_depth, n_estimators, learning_rate)
            m.fit(Xv[mask], yv[mask])
            models[b] = (t1, t2, m)
    else:
        gm = _new_model(max_depth, n_estimators, learning_rate)
        gm.fit(Xv, yv)

    trades = []
    for i in range(min_train, n - 1):
        if mode == "regime" and regime_col is not None and models:
            xg = Xv[i : i + 1]
            t1_val, t2_val, _ = list(models.values())[0][:3]
            b = (
                0 if xg[0, regime_col] <= t1_val
                else (1 if xg[0, regime_col] <= t2_val else 2)
            )
            if b not in models:
                continue
            p_up = models[b][2].predict_proba(xg)[0, 1]
        else:
            p_up = gm.predict_proba(Xv[i : i + 1])[0, 1]

        if mode == "meta":
            trend = Xv[i, trend_col] if trend_col is not None else 0
            if trend > 0:
                side = "BUY"
            elif trend < 0:
                side = "SELL"
            else:
                continue
            take = (p_up >= threshold) if side == "BUY" else (p_up <= 1 - threshold)
            if not take:
                continue
        else:
            if p_up >= threshold:
                side = "BUY"
            elif p_up <= 1 - threshold:
                side = "SELL"
            else:
                continue

        actual_ret = rv[i]
        pnl = _trade_pnl(contract, side, actual_ret, payout, spread_bps)
        trades.append(pnl > 0)

    if not trades:
        return 0.0, 0.0

    is_hit = sum(trades) / len(trades)
    is_exp = np.mean([_trade_pnl(contract, "BUY", rv[i], payout, spread_bps)
                       for i in range(min_train, n - 1)
                       if (rv[i] > 0)]) if len(trades) > 0 else 0.0
    # simpler: just use mean pnl
    is_pnls = []
    for i in range(min_train, n - 1):
        if mode == "regime" and regime_col is not None and models:
            xg = Xv[i : i + 1]
            t1_val, t2_val, _ = list(models.values())[0][:3]
            b = (
                0 if xg[0, regime_col] <= t1_val
                else (1 if xg[0, regime_col] <= t2_val else 2)
            )
            if b not in models:
                continue
            p_up = models[b][2].predict_proba(xg)[0, 1]
        else:
            p_up = gm.predict_proba(Xv[i : i + 1])[0, 1]

        if mode == "meta":
            trend = Xv[i, trend_col] if trend_col is not None else 0
            if trend > 0:
                side = "BUY"
            elif trend < 0:
                side = "SELL"
            else:
                continue
            take = (p_up >= threshold) if side == "BUY" else (p_up <= 1 - threshold)
            if not take:
                continue
        else:
            if p_up >= threshold:
                side = "BUY"
            elif p_up <= 1 - threshold:
                side = "SELL"
            else:
                continue

        is_pnls.append(_trade_pnl(contract, side, rv[i], payout, spread_bps))

    is_exp_bps = float(np.mean(is_pnls)) * 10000 if is_pnls else 0.0
    return is_hit, is_exp_bps


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_gate_report(result: CostGateResult) -> str:
    """Human-readable gate report."""
    lines = []
    lines.append("=" * 64)
    lines.append("  COST-AWARE WALK-FORWARD GATE REPORT")
    lines.append("=" * 64)
    lines.append(f"  Contract:              {result.contract}")
    lines.append(f"  Win reward:            {result.win_cost:+.2f}")
    lines.append(f"  Loss cost:             {result.loss_cost:+.2f}")
    lines.append("")
    lines.append(f"  --- OOS Performance ---")
    lines.append(f"  Total trades:          {result.total_trades}")
    lines.append(f"  Hits:                  {result.hits}")
    lines.append(f"  Hit rate:              {result.hit_rate:.2%}")
    lines.append(f"  Expectancy/trade:      {result.expectancy_per_trade:+.4f} ({result.expectancy_per_trade_bps:+.1f} bps)")
    lines.append(f"  Profit factor:         {result.profit_factor:.2f}")
    lines.append(f"  Sharpe (annualised):   {result.sharpe_annualized:.2f}")
    lines.append(f"  Max drawdown:          {result.max_drawdown_bps:+.0f} bps")
    lines.append(f"  Deflated Sharpe:       {result.deflated_sharpe:.4f}")
    lines.append("")
    lines.append(f"  --- Baseline (buy-and-hold) ---")
    lines.append(f"  Baseline hit rate:     {result.baseline_hit_rate:.2%}")
    lines.append(f"  Baseline expectancy:   {result.baseline_expectancy_bps:+.1f} bps")
    lines.append("")
    lines.append(f"  --- In-Sample vs OOS ---")
    lines.append(f"  In-sample hit rate:    {result.in_sample_hit_rate:.2%}")
    lines.append(f"  In-sample expectancy:  {result.in_sample_expectancy_bps:+.1f} bps")
    gap = abs(result.hit_rate - result.in_sample_hit_rate) * 100
    lines.append(f"  Gap (IS vs OOS):       {gap:.2f}pp")
    lines.append("")
    lines.append(f"  --- Gate ---")
    for k, v in result.gate_details.items():
        if k in ("threshold", "mode", "horizon"):
            lines.append(f"  {k:<22} {v}")
        else:
            lines.append(f"  {k:<22} {v}")
    lines.append("")
    if result.gate_pass:
        lines.append("  *** GATE: PASS ***")
    else:
        lines.append("  *** GATE: FAIL ***")
    lines.append("=" * 64)
    report = "\n".join(lines)
    print(report)
    return report


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def sweep(filename: str, contract: str = "iq_binary",
          payout: float = 0.80, spread_bps: float = 20) -> list[CostGateResult]:
    """Sweep thresholds × modes × horizons, return sorted results."""
    thresholds = [0.56, 0.60, 0.65, 0.70]
    modes = ["global", "regime", "meta"]
    horizons = [1, 2, 4]

    results = []
    total_combos = len(thresholds) * len(modes) * len(horizons)
    done = 0

    for h in horizons:
        for m in modes:
            for t in thresholds:
                done += 1
                print(f"\n[{done}/{total_combos}] threshold={t} mode={m} horizon={h}")
                try:
                    r = cost_aware_walkforward(
                        filename, contract=contract, payout=payout,
                        spread_bps=spread_bps, horizon=h,
                        threshold=t, mode=m, min_train=1500,
                    )
                    results.append(r)
                    status = "PASS" if r.gate_pass else "FAIL"
                    print(f"  -> {status}  hit={r.hit_rate:.2%} exp={r.expectancy_per_trade_bps:+.1f}bps "
                          f"trades={r.total_trades} PF={r.profit_factor:.2f}")
                except Exception as e:
                    print(f"  -> ERROR: {e}")

    # sort by expectancy descending
    results.sort(key=lambda r: r.expectancy_per_trade_bps, reverse=True)
    return results


def print_sweep_scoreboard(results: list[CostGateResult]) -> str:
    """Print sorted scoreboard table."""
    lines = []
    lines.append("=" * 100)
    lines.append("  SWEEP SCOREBOARD  (sorted by expectancy_after_cost, descending)")
    lines.append("=" * 100)
    header = f"  {'#':>3} {'Threshold':>9} {'Mode':<8} {'Horizon':>7} {'Trades':>7} " \
             f"{'HitRate':>8} {'Exp(bps)':>9} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8} " \
             f"{'DStr':>6} {'IS Hit':>7} {'Gap':>6} {'GATE':>5}"
    lines.append(header)
    lines.append("  " + "-" * 96)

    for i, r in enumerate(results, 1):
        gap = abs(r.hit_rate - r.in_sample_hit_rate) * 100
        gate_sym = "PASS" if r.gate_pass else "FAIL"
        line = (
            f"  {i:>3} {r.gate_details.get('threshold', ''):>9} "
            f"{r.gate_details.get('mode', ''):<8} "
            f"{r.gate_details.get('horizon', ''):>7} "
            f"{r.total_trades:>7} "
            f"{r.hit_rate:>7.2%} "
            f"{r.expectancy_per_trade_bps:>+9.1f} "
            f"{r.profit_factor:>6.2f} "
            f"{r.sharpe_annualized:>7.2f} "
            f"{r.max_drawdown_bps:>+8.0f} "
            f"{r.deflated_sharpe:>6.4f} "
            f"{r.in_sample_hit_rate:>6.2%} "
            f"{gap:>5.1f}pp "
            f"{gate_sym:>5}"
        )
        lines.append(line)

    lines.append("  " + "-" * 96)
    pass_count = sum(1 for r in results if r.gate_pass)
    lines.append(f"  Total configs: {len(results)}  |  PASS: {pass_count}  |  FAIL: {len(results) - pass_count}")
    lines.append("=" * 100)
    report = "\n".join(lines)
    print(report)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/cost_gate.py iq_binary <filename> [--threshold 0.56] [--mode global] [--horizon 1]")
        print("  python src/cost_gate.py spot_gold <filename> [--spread 20]")
        print("  python src/cost_gate.py sweep <filename> [--contract iq_binary]")
        sys.exit(1)

    contract = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else "xau_usd_1h_real.csv"
    # strip leading "data/" if present — load_history() prepends DATA/
    if filename.startswith("data/"):
        filename = filename[len("data/"):]

    # parse optional args
    kwargs = {}
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            kwargs["threshold"] = float(args[i + 1]); i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            kwargs["mode"] = args[i + 1]; i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            kwargs["horizon"] = int(args[i + 1]); i += 2
        elif args[i] == "--spread" and i + 1 < len(args):
            kwargs["spread_bps"] = float(args[i + 1]); i += 2
        elif args[i] == "--payout" and i + 1 < len(args):
            kwargs["payout"] = float(args[i + 1]); i += 2
        else:
            i += 1

    if contract == "sweep":
        c = kwargs.pop("contract", "iq_binary")
        payout = kwargs.pop("payout", 0.80)
        spread = kwargs.pop("spread_bps", 20)
        results = sweep(filename, contract=c, payout=payout, spread_bps=spread)
        print_sweep_scoreboard(results)
    else:
        result = cost_aware_walkforward(filename, contract=contract, **kwargs)
        print_gate_report(result)


if __name__ == "__main__":
    main()
