"""Live inference for the meta-labeling model (safe paper mode).

Trains on the full history ONCE at startup, then at each 4h bar close
produces a BUY/SELL + confidence message for Telegram. The "paper" part:
we never place orders — we only emit the signal for manual review.
"""
from __future__ import annotations

import csv
import os
import pathlib
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.ml_edge import build_features, _new_model
from src.pipeline import (
    load_history, get_weekly_trend_at, load_weekly_history,
    compute_atr_pct, compute_macd,
)

MODEL = {
    "filename": None,
    "horizon": 4,          # best result on 4h bars
    "threshold": 0.50,     # meta mode: P(take) bar, side comes from weekly trend
    "min_train": 500,
    "refit_every": 250,
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
}

# Live MTF veto knobs (env-overridable)
ATR_MIN_PCT = float(os.environ.get("CEREBRUM_ATR_MIN_PCT", "0.003"))
MACD_VETO_ENABLED = os.environ.get("CEREBRUM_MACD_VETO", "1") not in ("0", "false", "False")
MACD_HIST_EPS = float(os.environ.get("CEREBRUM_MACD_EPS", "0.0"))  # deadband around 0

_model = None
_model_trained_rows = 0
_weekly_df = None

# Paper-log CSV path
ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER_LOG = ROOT / "data" / "meta_paper_log.csv"

def _log_meta_decision(res: dict) -> None:
    """Append a meta-labeling decision to the paper-log CSV."""
    try:
        # Create CSV with header if it doesn't exist
        write_header = not PAPER_LOG.exists()
        
        with PAPER_LOG.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "side", "trade", "probability", "take_probability",
                    "threshold", "trend", "trend_label", "reason", "horizon",
                    "edge_source", "trained_rows"
                ])
            
            now = datetime.now(timezone.utc).isoformat()
            writer.writerow([
                now,
                res.get("side", ""),
                res.get("trade", False),
                res.get("probability", 0.0),
                res.get("take_probability", 0.0),
                res.get("threshold", 0.0),
                res.get("trend", 0),
                res.get("trend_label", ""),
                res.get("reason", ""),
                res.get("horizon", 0),
                res.get("edge_source", ""),
                res.get("trained_rows", 0),
            ])
    except Exception as e:
        print(f"[meta-live] paper-log write failed: {e}")


def _train_once(filename: Optional[str]):
    """Load history, build features, train on everything available up to now."""
    global _model, _model_trained_rows, _weekly_df
    df = load_history(filename)
    X, y, _ = build_features(df, horizon=MODEL["horizon"])
    model = _new_model(MODEL["max_depth"], MODEL["n_estimators"], MODEL["learning_rate"])
    model.fit(X.values, y.values)
    _model = model
    _model_trained_rows = len(X)
    _weekly_df = load_weekly_history()
    print(f"[meta-live] trained on {len(X)} bars -> horizon={MODEL['horizon']}")


def _live_mtf_veto(df: pd.DataFrame, side: str) -> tuple[bool, str, dict]:
    """Live volatility + momentum veto on *current* bar (no slot-average).

    Returns (vetoed, reason_suffix, diagnostics_dict).
    - ATR veto: if current ATR% < ATR_MIN_PCT => low-vol chop, force WATCH
    - MACD veto: if side BUY but hist < -eps or side SELL but hist > +eps => oppose
      We do NOT silently scale prob; we veto NO-TRADE with explicit reason so
      paper log shows why. Caller may also apply 20% confidence penalty if desired.
    """
    diag: dict = {}
    try:
        atr_pct = compute_atr_pct(df).iloc[-1]
        diag["atr_pct"] = float(atr_pct) if pd.notna(atr_pct) else None
    except Exception:
        diag["atr_pct"] = None
    try:
        _, _, hist = compute_macd(df["close"])
        macd_hist = hist.iloc[-1]
        diag["macd_hist"] = float(macd_hist) if pd.notna(macd_hist) else None
    except Exception:
        diag["macd_hist"] = None

    # ATR veto (current bar, not seasonal average)
    atr = diag.get("atr_pct")
    if atr is not None and atr < ATR_MIN_PCT:
        return True, f" | ATR filter veto: atr={atr:.4f} < {ATR_MIN_PCT:.4f} (low vol)", diag

    # MACD veto — only when clearly opposing (outside eps deadband)
    mh = diag.get("macd_hist")
    if MACD_VETO_ENABLED and mh is not None and mh is not None:
        if side == "BUY" and mh < -abs(MACD_HIST_EPS):
            # if eps==0, any negative vetoes; if eps>0, require meaningful opposite
            if MACD_HIST_EPS == 0.0 or mh < -0.5:  # keep legacy 0.5 guard when eps==0 for less noisy veto
                # only veto if hist is clearly negative, not dust
                if mh < -0.1:
                    return True, f" | MACD veto: BUY but hist={mh:.2f} < 0 (bearish momentum)", diag
                # dust near 0 -> penalty path handled by caller, not veto
        elif side == "SELL" and mh > abs(MACD_HIST_EPS):
            if MACD_HIST_EPS == 0.0 or mh > 0.5:
                if mh > 0.1:
                    return True, f" | MACD veto: SELL but hist={mh:.2f} > 0 (bullish momentum)", diag
    return False, "", diag


def predict_meta(filename: Optional[str] = None,
                 horizon: Optional[int] = None,
                 threshold: Optional[float] = None) -> dict:
    """Return a full report dict every call; trade gate lives in res['trade'].

    Keys: side, probability, trend, trend_label, reason, trade, horizon,
          threshold, edge_source, trained_rows, diagnostics (atr_pct, macd_hist)
    """
    horizon = horizon or MODEL["horizon"]
    threshold = threshold or MODEL["threshold"]
    if _model is None:
        _train_once(filename)

    df = load_history(filename)
    X, y, fwd_ret = build_features(df, horizon=horizon)
    res = {
        "trade": False, "side": None, "probability": None,
        "reason": "insufficient data",
        "horizon": horizon, "threshold": threshold,
        "trend": 0, "trend_label": "n/a",
        "edge_source": f"meta-labeling (trained {_model_trained_rows} bars)",
        "trained_rows": _model_trained_rows,
    }
    if len(X) == 0:
        return res

    ts = df.index[-1]
    trend = get_weekly_trend_at(_weekly_df, ts)
    p_up = _model.predict_proba(X.iloc[-1:].values)[0, 1]
    res.update({
        "trend": trend,
        "trend_label": {1: "UP", -1: "DOWN", 0: "NEUTRAL"}[trend],
        "probability": round(float(p_up), 4),
    })

    # Determine intended side first (without veto)
    intended_side: Optional[str] = None
    if trend > 0:
        intended_side = "BUY"
        res["side"] = "BUY"
        res["take_probability"] = float(p_up)
        res["threshold_to_clear"] = ">= " + str(threshold)
        if p_up >= threshold:
            res["trade"] = True
            res["reason"] = f"weekly trend UP + model P(up)={p_up:.3f} >= {threshold}"
        else:
            res["reason"] = "weekly trend UP but model confidence too low"
    elif trend < 0:
        intended_side = "SELL"
        res["side"] = "SELL"
        res["take_probability"] = float(1 - p_up)
        res["threshold_to_clear"] = "<= " + str(1 - threshold)
        if p_up <= 1 - threshold:
            res["trade"] = True
            res["reason"] = f"weekly trend DOWN + model P(up)={p_up:.3f} <= {1 - threshold}"
        else:
            res["reason"] = "weekly trend DOWN but model confidence too low"
    else:
        res["side"] = "WATCH"
        intended_side = "WATCH"
        res["reason"] = "weekly trend NEUTRAL (price within ±0.5% of 20-week SMA)"

    # ---- Live MTF veto (current-bar ATR/MACD, not slot average) ----
    # Only veto if we *would* have traded; keep low-confidence NO-TRADE as is,
    # but enrich its reason with diagnostics.
    diag: dict = {}
    if intended_side in ("BUY", "SELL"):
        vetoed, veto_reason, diag = _live_mtf_veto(df, intended_side)
        res["atr_pct"] = diag.get("atr_pct")
        res["macd_hist"] = diag.get("macd_hist")
        if vetoed and res["trade"]:
            # Flip to NO-TRADE with explicit WATCH and preserve prior reason
            res["trade"] = False
            res["side"] = "WATCH"
            res["reason"] = res["reason"] + veto_reason
        elif veto_reason:
            # Was already NO-TRADE but veto would have blocked anyway — append for audit
            res["reason"] = res["reason"] + veto_reason
        # Soft penalty: if not vetoed but MACD dust opposes, shave confidence 20%
        # (mirrors backtest penalty) and re-check threshold
        elif res["trade"]:
            mh = diag.get("macd_hist")
            if mh is not None and (
                (intended_side == "BUY" and mh is not None and -0.1 <= mh < 0) or
                (intended_side == "SELL" and mh is not None and 0 < mh <= 0.1)
            ):
                # dust opposite -> 20% shave
                adj = res["take_probability"] * 0.8
                if adj < threshold:
                    res["trade"] = False
                    res["side"] = "WATCH"
                    res["reason"] = res["reason"] + f" | MACD dust penalty: prob {res['take_probability']:.3f} -> {adj:.3f} < {threshold}"
                else:
                    res["take_probability"] = adj
                    res["reason"] = res["reason"] + f" | MACD dust shave 20% -> {adj:.3f}"
    else:
        # WATCH case: still compute diag for logging
        _, _, diag = _live_mtf_veto(df, "BUY")  # dummy side, just to get diag
        res["atr_pct"] = diag.get("atr_pct")
        res["macd_hist"] = diag.get("macd_hist")
    
    # Log every decision to paper-log CSV
    _log_meta_decision(res)
    
    return res
