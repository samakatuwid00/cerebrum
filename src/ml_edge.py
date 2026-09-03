"""Predictive ML engine for cerebrum_trader_bot.

Unlike src/backtest.py (per-(hour,day_of_week) lookup table = seasonal average),
this module trains a learned model on per-bar features that are *conditionally*
informative about the next bar, and validates with a TRUE walk-forward
(expanding window, refit every `refit_every` bars, zero lookahead).

Features per bar (all computable from data available at bar close):
  rsi, bollinger_pos, momentum          - existing indicators
  atr_pct, macd_hist                    - existing MTF indicators
  atr_regime                            - ATR percentile vs trailing window (vol regime)
  dxy_ret_1, dxy_ret_4, dxy_rsi         - DXY lead features (dollar drives gold)
  vwap_dev_atr                          - dislocation from VWAP in ATR units
  hour_sin, hour_cos, dow_sin, dow_cos  - cyclical calendar encodings
  weekly_trend                          - +1/-1/0 weekly regime

Model: sklearn GradientBoostingClassifier (or HistGradientBoosting) with
class_weight handling. Validated with walk-forward.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

from src.pipeline import (
    load_history,
    compute_rsi,
    compute_bollinger_position,
    compute_momentum,
    compute_atr_pct,
    compute_macd,
    load_weekly_history,
    compute_weekly_trend,
    DXY_1H_CSV,
    US10Y_1H_CSV,
    _load_csv,
)

FEATURES = [
    "rsi", "bollinger_pos", "momentum",
    "macd_hist", "atr_regime",
    "dxy_ret_1", "dxy_rsi",
    "vwap_dev_atr",
    "hour_sin", "hour_cos", "dow_cos",
    "weekly_trend",
    # cross-asset lead (append-only; order not shifted per hard constraint)
    "us10y_ret_1", "us10y_ret_4", "us10y_rsi",
]
# Pruned after permutation importance showed negative/zero contribution:
# atr_pct, dow_sin, dxy_ret_4 (hour_cos kept as pair with hour_sin).
DROP_FEATURES = ["atr_pct", "dow_sin", "dxy_ret_4"]


# --------------------------------------------------------------------------- #
# Feature builders
# --------------------------------------------------------------------------- #
def _atr_regime(atr_pct: pd.Series, window: int = 500) -> pd.Series:
    """Percentile rank of current ATR% vs trailing `window` bars, in [0,1].

    1.0 = highest volatility in the trailing window. Uses a rolling window so
    it is computable live (no full-sample lookahead).
    """
    def rank_last(x):
        return (x <= x[-1]).mean()
    return atr_pct.rolling(window, min_periods=max(50, window // 5)).apply(rank_last, raw=True)


def _vwap_dev(df: pd.DataFrame, atr: pd.Series, window: int = 20) -> pd.Series:
    """(close - VWAP) / ATR over trailing `window` bars. ATR units = comparable."""
    vol = df["volume"].astype(float)
    # If volume is all zero (yfinance gold sometimes is), fall back to SMA proxy
    if vol.abs().sum() == 0:
        vwap = df["close"].rolling(window, min_periods=1).mean()
    else:
        pv = df["close"] * vol
        vwap = (pv.rolling(window, min_periods=1).sum()
                / vol.rolling(window, min_periods=1).sum().replace(0, np.nan))
        vwap = vwap.fillna(df["close"].rolling(window, min_periods=1).mean())
    return (df["close"] - vwap) / atr.replace(0, np.nan)


def _dxy_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dollar-index lead features aligned (ffill) onto the target bar index.

    dxy_ret_1 / dxy_ret_4 : DXY % change over last 1 / 4 bars
    dxy_rsi               : RSI(14) of DXY close
    """
    if not DXY_1H_CSV.exists():
        return pd.DataFrame(index=target_index,
                            columns=["dxy_ret_1", "dxy_ret_4", "dxy_rsi"], dtype=float)
    dxy = _load_csv(DXY_1H_CSV)
    dxy = dxy.reindex(dxy.index.union(target_index)).ffill().reindex(target_index)
    out = pd.DataFrame(index=target_index)
    out["dxy_ret_1"] = dxy["close"].pct_change(1)
    out["dxy_ret_4"] = dxy["close"].pct_change(4)
    out["dxy_rsi"] = compute_rsi(dxy["close"], period=14)
    return out


def _us10y_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """10Y-Treasury-yield lead features aligned (ffill) onto the target bar index.

    us10y_ret_1 / us10y_ret_4 : US10Y % change over last 1 / 4 bars
    us10y_rsi                 : RSI(14) of US10Y close

    Mirror of _dxy_features(). If US10Y_1H_CSV is missing, return an empty
    DataFrame with those columns (NaN-filled) so the matrix degrades gracefully.
    """
    if not US10Y_1H_CSV.exists():
        return pd.DataFrame(index=target_index,
                            columns=["us10y_ret_1", "us10y_ret_4", "us10y_rsi"], dtype=float)
    us10y = _load_csv(US10Y_1H_CSV)
    us10y = us10y.reindex(us10y.index.union(target_index)).ffill().reindex(target_index)
    out = pd.DataFrame(index=target_index)
    out["us10y_ret_1"] = us10y["close"].pct_change(1)
    out["us10y_ret_4"] = us10y["close"].pct_change(4)
    out["us10y_rsi"] = compute_rsi(us10y["close"], period=14)
    return out


def _weekly_trend_series(target_index: pd.DatetimeIndex) -> pd.Series:
    """Weekly trend (+1/-1/0) mapped onto each bar, using only weeks already
    complete at that bar (no lookahead)."""
    weekly = load_weekly_history()
    trend = compute_weekly_trend(weekly)
    # reindex ffill: the latest completed week's trend applies to all bars after it
    return trend.reindex(trend.index.union(target_index)).ffill().reindex(target_index).fillna(0)


def build_features(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, fwd_ret) for a gold OHLCV frame.

    X : feature matrix aligned to df.index (NaN rows dropped)
    y : 1 if close `horizon` bars ahead > this close else 0
    fwd_ret : forward return over `horizon` bars (for PnL)
    """
    df = df.copy()
    close = df["close"].astype(float)

    atr = compute_atr_pct(df)  # fraction, not %
    macd_line, signal_line, macd_hist = compute_macd(close)

    X = pd.DataFrame(index=df.index)
    X["rsi"] = compute_rsi(close)
    X["bollinger_pos"] = compute_bollinger_position(close)
    X["momentum"] = compute_momentum(close)
    X["atr_pct"] = atr
    X["macd_hist"] = macd_hist
    X["atr_regime"] = _atr_regime(atr)
    X["vwap_dev_atr"] = _vwap_dev(df, atr * close)  # atr back to price units

    # cross-asset lead
    X = X.join(_dxy_features(df.index))
    X = X.join(_us10y_features(df.index))

    # calendar (cyclical)
    hour = df.index.hour + df.index.minute / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # weekly regime
    X["weekly_trend"] = _weekly_trend_series(df.index)

    # target: direction `horizon` bars ahead
    future = close.shift(-horizon)
    y = (future > close).astype(float)
    y[future.isna()] = np.nan

    # forward return over `horizon` bars for PnL
    fwd_ret = future / close - 1.0

    # drop underperforming features before the validity mask
    X = X.drop(columns=[c for c in DROP_FEATURES if c in X.columns], errors="ignore")

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid].astype(int)
    fwd_ret = fwd_ret[valid]
    return X, y, fwd_ret


# --------------------------------------------------------------------------- #
# Walk-forward model evaluation
# --------------------------------------------------------------------------- #
def _new_model(max_depth, n_estimators, learning_rate):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        max_iter=n_estimators,
        learning_rate=learning_rate,
        early_stopping=False,
    )


def walkforward_model(filename: Optional[str] = None,
                      threshold: float = 0.53,
                      refit_every: int = 250,
                      min_train: int = 1500,
                      max_depth: int = 3,
                      n_estimators: int = 200,
                      learning_rate: float = 0.05,
                      horizon: int = 1,
                      mode: str = "global"):
    """Expanding-window walk-forward. Zero lookahead.

    mode:
      "global" : one model on all training bars (original behaviour)
      "regime" : fit T/Median/B tertile models on atr_regime; route each bar
                 to its bucket's model (each bar still uses ONE model - no double)
      "meta"   : primary direction = weekly_trend sign global-mean filter;
                 meta model gates the take/skip decision
    """
    df = load_history(filename)
    print(f"[ml] Loaded {len(df)} bars -> mode={mode}")
    X, y, fwd_ret = build_features(df, horizon=horizon)
    print(f"[ml] Feature matrix: {X.shape}, positives={y.mean():.2%}, horizon={horizon}")

    feat_cols = list(X.columns)
    regime_col = feat_cols.index("atr_regime") if "atr_regime" in feat_cols else None
    trend_col = feat_cols.index("weekly_trend") if "weekly_trend" in feat_cols else None

    Xv, yv, rv = X.values, y.values, fwd_ret.values
    idx = X.index
    n = len(Xv)

    global_model = None
    regime_models = {}
    last_fit = -10**9
    trades = []

    for i in range(min_train, n - 1):
        if i - last_fit >= refit_every or (mode == "global" and global_model is None):
            if mode == "regime" and regime_col is not None:
                regime_models.clear()
                reg_vals = Xv[:i, regime_col]
                t1 = np.nanquantile(reg_vals, 1/3)
                t2 = np.nanquantile(reg_vals, 2/3)
                for b in range(3):
                    if b == 0: mask = reg_vals <= t1
                    elif b == 1: mask = (reg_vals > t1) & (reg_vals <= t2)
                    else: mask = reg_vals > t2
                    m = _new_model(max_depth, n_estimators, learning_rate)
                    m.fit(Xv[:i][mask], yv[:i][mask])
                    regime_models[b] = (t1, t2, m)
            else:
                global_model = _new_model(max_depth, n_estimators, learning_rate)
                global_model.fit(Xv[:i], yv[:i])
            last_fit = i

        # pick model and probability of UP
        if mode == "regime" and regime_col is not None:
            xg = Xv[i:i+1]
            t1, t2, _ = list(regime_models.values())[0][:3]
            b = 0 if xg[0, regime_col] <= t1 else (1 if xg[0, regime_col] <= t2 else 2)
            if b not in regime_models:
                continue
            p_up = regime_models[b][2].predict_proba(xg)[0, 1]
        else:
            p_up = global_model.predict_proba(Xv[i:i+1])[0, 1]

        # decide the trade by mode
        if mode == "meta":
            trend = Xv[i, trend_col] if trend_col is not None else 0
            if trend > 0:
                side = "BUY"
            elif trend < 0:
                side = "SELL"
            else:
                continue  # primary says no position
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

        r = rv[i]
        win = (r > 0) if side == "BUY" else (r < 0)
        trades.append({
            "timestamp": idx[i].isoformat(),
            "side": side, "prob": float(prob),
            "actual_ret": float(r), "win": int(win),
        })

    trades_df = pd.DataFrame(trades)
    summary = _summarize(trades_df, threshold)
    base_rets = rv[min_train:n - 1]
    summary["baseline_hit_rate"] = float((base_rets > 0).mean())
    summary["baseline_pnl_bps"] = float(base_rets.sum() * 10000)
    summary["mode"] = mode
    return trades_df, summary


def _summarize(trades_df: pd.DataFrame, threshold: float) -> dict:
    if len(trades_df) == 0:
        return {}
    total = len(trades_df)
    wins = trades_df["win"].sum()
    hit = wins / total
    rets = trades_df["actual_ret"]
    signed = trades_df["actual_ret"] * np.where(trades_df["side"] == "BUY", 1, -1)
    pnl = signed * 10000
    cum = pnl.cumsum()
    max_dd = (cum - cum.expanding().max()).min()
    sharpe = (signed.mean() / signed.std() * np.sqrt(252 * 24)) if signed.std() > 0 else 0
    return {
        "threshold": threshold,
        "total_trades": int(total),
        "wins": int(wins),
        "hit_rate": float(hit),
        "avg_return_bps": float(signed.mean() * 10000),
        "total_pnl_bps": float(pnl.sum()),
        "sharpe_annualized": float(sharpe),
        "max_drawdown_bps": float(max_dd),
    }


# --------------------------------------------------------------------------- #
# Feature importance (diagnostic, in-sample on last refit)
# --------------------------------------------------------------------------- #
def feature_importance_report(filename: Optional[str] = None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    df = load_history(filename)
    X, y, _ = build_features(df)
    split = int(len(X) * 0.7)
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)
    model.fit(X.values[:split], y.values[:split])
    imp = permutation_importance(model, X.values[split:], y.values[split:],
                                 n_repeats=5, random_state=0)
    order = np.argsort(-imp.importances_mean)
    print("\n[ml] Permutation importance (out-of-sample 30%):")
    for i in order:
        print(f"  {FEATURES[i]:<18} {imp.importances_mean[i]:+.4f} +/- {imp.importances_std[i]:.4f}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "walkforward"
    fname = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "importance":
        feature_importance_report(fname)
    else:
        for thr in [0.53, 0.55, 0.58]:
            t, s = walkforward_model(filename=fname, threshold=thr)
            if s:
                print(f"  thr={thr}: hit={s['hit_rate']:.2%} trades={s['total_trades']} "
                      f"pnl={s['total_pnl_bps']:.0f}bps sharpe={s['sharpe_annualized']:.2f} "
                      f"maxDD={s['max_drawdown_bps']:.0f}bps")
            else:
                print(f"  thr={thr}: no trades")
