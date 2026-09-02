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
from src.pipeline import load_history, get_weekly_trend_at, load_weekly_history

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


def predict_meta(filename: Optional[str] = None,
                 horizon: Optional[int] = None,
                 threshold: Optional[float] = None) -> dict:
    """Return a full report dict every call; trade gate lives in res['trade'].

    Keys: side, probability, trend, trend_label, reason, trade, horizon,
          threshold, edge_source, trained_rows
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

    if trend > 0:
        res["side"] = "BUY"
        res["take_probability"] = float(p_up)
        res["threshold_to_clear"] = ">= " + str(threshold)
        if p_up >= threshold:
            res["trade"] = True
            res["reason"] = f"weekly trend UP + model P(up)={p_up:.3f} >= {threshold}"
        else:
            res["reason"] = "weekly trend UP but model confidence too low"
    elif trend < 0:
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
        res["reason"] = "weekly trend NEUTRAL (price within ±0.5% of 20-week SMA)"
    
    # Log every decision to paper-log CSV
    _log_meta_decision(res)
    
    return res
