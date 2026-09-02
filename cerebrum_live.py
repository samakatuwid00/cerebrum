"""Cerebrum Trader Bot – live orchestrator.

Runs in three threads:
  1. FX feed  – polls USD/EUR every 30s, appends to data/usd_eur_live.csv
  2. Pipeline – every CYCLE_SECONDS reads the live CSV + synthetic baseline
                and prints the next buy/sell window
  3. (Telegram alerts are sent if creds are present)

Stop with Ctrl-C.
"""

import csv
import datetime as dt
import os
import pathlib
import sys
import threading
import time
import pytz

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env first so the SSID / Telegram creds are visible
from src.config_loader import load_env
load_env()

from src import pipeline as pl           # BUY engine (legacy)
from src import pipeline_30m as pl30     # 30-min scanner (legacy)
from src import sell_window as sw        # SELL engine (legacy)
from src import fx_feed                  # live FX poll
from src import alert                    # Telegram
from src import live_edge                # live rolling 1‑h edge
from src import meta_live                # meta-labeling predictive engine

CYCLE_SECONDS = 60     # how often we re-evaluate the signal
HEARTBEAT_SECONDS = 4 * 3600  # 4 hours – throttle to 4h candle boundaries
LIVE_CSV      = ROOT / "data" / "usd_eur_live.csv"
HISTORIC_CSV  = ROOT / "data" / "usd_eur_hourly.csv"  # baseline

# Threshold + lambda: read from .env if set, else fall back to config defaults.
# CEREBRUM_THRESHOLD / CEREBRUM_LAMBDA in .env override config/settings.py.
from config import settings as _settings
THRESHOLD = float(os.environ.get("CEREBRUM_THRESHOLD") or _settings.CONFIDENCE_THRESHOLD)
LAMBDA    = float(os.environ.get("CEREBRUM_LAMBDA")    or _settings.LAMBDA_NEWS)

# Meta-labeling engine (new predictive path, safe paper mode).
# The old slot-table scanners below still run as a fallback when meta gives no trade.
META_FILE = os.environ.get("CEREBRUM_META_FILE") or None  # e.g. xau_usd_4h_real.csv
META_HORIZON = int(os.environ.get("CEREBRUM_META_HORIZON", "4"))
META_THRESHOLD = float(os.environ.get("CEREBRUM_META_THRESHOLD", "0.50"))

_stop = threading.Event()

# ---------------------------------------------------------------------------
def _load_live_rates() -> list[tuple[dt.datetime, float]]:
    """Read the live tick file.  Returns [(datetime, rate), ...] ascending."""
    if not LIVE_CSV.exists():
        return []
    out = []
    with LIVE_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = dt.datetime.fromisoformat(r["datetime"])
                out.append((ts, float(r["rate"])))
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    return out

def _summarise_live(rates: list[tuple[dt.datetime, float]]) -> str:
    if not rates:
        return "no live data yet"
    last_ts, last_rate = rates[-1]
    if len(rates) < 2:
        return f"last {last_rate:.6f} @ {last_ts.isoformat()} (1 sample)"
    first_rate = rates[0][1]
    change_bps = (last_rate - first_rate) / first_rate * 10_000
    return (f"last {last_rate:.6f} @ {last_ts.strftime('%H:%M:%S')} UTC   "
            f"Δ {change_bps:+.2f} bps over {len(rates)} samples")

# ---------------------------------------------------------------------------
def _feed_loop():
    while not _stop.is_set():
        try:
            fx_feed.poll_once()
        except Exception as e:
            print(f"[feed] {e}", file=sys.stderr)
        for _ in range(30):
            if _stop.is_set():
                return
            time.sleep(1)

def _signal_loop():
    """Re-evaluate BUY + SELL signal every CYCLE_SECONDS.

    Loads the best available history (XAU/USD 4h preferred, EUR/USD 30m fallback,
    synthetic last-resort) and finds the highest-probability upcoming 4h slot.
    Falls back to live rolling edge from IQ feed if enough bars are collected.
    """
    df_hist = pl.load_history()
    asset = "XAU/USD" if "xau" in str(ROOT / "data" / "xau_usd_4h_real.csv").lower() and pl.load_history.__module__ else "?"
    synthetic_edge = pl.compute_edge(df_hist)
    print(f"[signal] baseline edge loaded ({len(synthetic_edge)} slots, "
          f"{len(df_hist)} bars from {df_hist.index.min().date()} -> {df_hist.index.max().date()})",
          flush=True)

    # wait until we have at least one live tick (FX or IQ)
    while not _stop.is_set() and not (LIVE_CSV.exists() or _iq_csv_exists()):
        time.sleep(1)

    while not _stop.is_set():
        try:
            now = dt.datetime.now(pytz.UTC)
            
            # Throttle to 4h boundaries: only run at 00, 04, 08, 12, 16, 20 UTC
            # But still check every minute to catch the boundary quickly
            if now.hour % 4 != 0:
                # Not at a 4h boundary – skip this cycle
                pass
            else:
                rates = _load_live_rates()
                live_summary = _summarise_live(rates)

                # pick the best edge available right now (legacy fallback)
                live_e = live_edge.compute_live_edge(min_bars_required=24)
                if live_e is not None and len(live_e) >= 12:
                    edge = live_e
                    edge_source = f"LIVE ({len(live_e)} h)"
                else:
                    edge = synthetic_edge
                    edge_source = "baseline"

                print(f"\n— {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {live_summary}")

                # ===== PREDICTIVE ENGINE: meta-labeling (paper mode) =====
                meta_res = meta_live.predict_meta(filename=META_FILE,
                                                 horizon=META_HORIZON,
                                                 threshold=META_THRESHOLD)
                if meta_res.get("trade"):
                    side = meta_res["side"]
                    prob = meta_res["take_probability"] or meta_res["probability"]
                    src = meta_res["edge_source"]
                    print(f"[META/{side}] prob={prob:.3f} horizon={meta_res['horizon']} "
                          f"thr={meta_res['threshold']} src={src}")
                    alert.send_message(
                        alert.format_window(
                            {"window_start_utc": now.isoformat(),
                             "probability": prob,
                             "side": side,
                             "edge_mean_ret": 0.0,
                             "n_samples": meta_res["trained_rows"]},
                            side=side,
                            threshold=meta_res["threshold"],
                            asset="XAU/USD",
                            df=df_hist,
                            timeframe=f"{meta_res['horizon']}h-meta",
                        )
                    )
                else:
                    reason = meta_res["reason"]
                    trend_label = meta_res["trend_label"]
                    prob = meta_res["probability"] or 0.0
                    side = meta_res["side"] or "?"
                    trend_emoji = {1: "\U0001f4c8", -1: "\U0001f4c9", 0: "\U0001f6cc"}.get(meta_res["trend"], "\U0001f6cc")
                    status = (f"{trend_emoji} [META STATUS] trend={trend_label} | P(up)={prob:.3f} | "
                              f"reason: {reason}")
                    print(status)
                    alert.send_message(status)
                    # ===== LEGACY FALLBACK: old slot-table engine =====
                    best = pl.find_best_window_in_next_24h(edge, threshold=THRESHOLD,
                                                            lam=LAMBDA, top_n=6)
                    buy  = [s for s in best if s.get("side") == "BUY"]
                    sell = [s for s in best if s.get("side") == "SELL"]
                    buy  = buy[0]  if buy  else {"none": None}
                    sell = sell[0] if sell else {"none": None}
                    if "none" in buy:
                        print("[BUY]  : no window above threshold")
                    else:
                        when = dt.datetime.fromisoformat(buy["window_start_utc"])
                        mins = (when - now).total_seconds() / 60
                        print(f"[BUY]  -> {when.strftime('%H:%M')} UTC (in {mins:+.0f} min) "
                              f"prob={buy['probability']:.3f}")
                        alert.send_message(alert.format_window(buy, side="BUY",
                                                                threshold=THRESHOLD,
                                                                asset="XAU/USD",
                                                                df=df_hist, timeframe="4h"))
                    if "none" in sell:
                        print("[SELL] : no window above threshold")
                    else:
                        when = dt.datetime.fromisoformat(sell["window_start_utc"])
                        mins = (when - now).total_seconds() / 60
                        print(f"[SELL] -> {when.strftime('%H:%M')} UTC (in {mins:+.0f} min) "
                              f"prob={sell['probability']:.3f}")
                        alert.send_message(alert.format_window(sell, side="SELL",
                                                                threshold=THRESHOLD,
                                                                asset="XAU/USD",
                                                                df=df_hist, timeframe="4h"))

                # ---- 30-minute hybrid scanner ----
                # Aggregates 1-min IQ bars into 30-min candles, computes edge,
                # finds the best 30m window above threshold. Sends a Telegram alert
                # that includes action buttons for manual approval (hybrid mode).
                try:
                    minute_bars = pl30._load_iq_minute_bars()
                    if len(minute_bars) >= 30:
                        df_30m = pl30.aggregate_to_30m(minute_bars)
                        edge_30m = pl30.compute_30m_edge(df_30m)
                        best_30m = pl30.find_next_30m_window(edge_30m, threshold=THRESHOLD,
                                                            lam=LAMBDA)
                        if "none" not in best_30m:
                            when_30 = dt.datetime.fromisoformat(best_30m["window_start_utc"])
                            mins_30 = (when_30 - now).total_seconds() / 60
                            side_30 = best_30m["side"]
                            msg_30  = (f"[{side_30} 30m] -> {when_30.strftime('%Y-%m-%d %H:%M')} UTC  "
                                        f"(in {mins_30:+.0f} min)  prob={best_30m['probability']:.3f}  "
                                        f"edge={best_30m['edge_mean_ret']:+.4%}  n={best_30m['n_samples']}")
                            print(msg_30)
                            alert.send_message(alert.format_window(best_30m, side=side_30,
                                                                     threshold=THRESHOLD,
                                                                     asset="XAU/USD",
                                                                     df=df_30m,
                                                                     timeframe="30m"))
                        else:
                            print("[30m]  : no window above threshold")
                    else:
                        print(f"[30m]  : need >=30 minute bars (have {len(minute_bars)})")
                except Exception as e:
                    print(f"[30m scan] error: {e}", file=sys.stderr)

        except Exception as e:
            print(f"[signal] error: {e}", file=sys.stderr)
        for _ in range(CYCLE_SECONDS):
            if _stop.is_set():
                return
            time.sleep(1)

def _iq_csv_exists() -> bool:
    p = pathlib.Path(ROOT / "data" / "iq_live_bars.csv")
    return p.exists() and p.stat().st_size > 0

# ---------------------------------------------------------------------------
def main():
    print("╭──────────────────────────────────────────────╮")
    print("│  Cerebrum Trader Bot — LIVE MODE             │")
    print("│  USD/EUR via open.er-api.com  (no signup)    │")
    print("╰──────────────────────────────────────────────╯")
    t_feed   = threading.Thread(target=_feed_loop,   name="feed",   daemon=True)
    t_signal = threading.Thread(target=_signal_loop, name="signal", daemon=True)
    t_feed.start()
    t_signal.start()

    # Telegram status
    tg_token = os.environ.get("CEREBRUM_TELEGRAM_TOKEN", "")
    tg_chat  = os.environ.get("CEREBRUM_TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        print(f"[live] Telegram alerts ENABLED (chat_id={tg_chat[:6]}…)", flush=True)
    else:
        missing = []
        if not tg_token: missing.append("CEREBRUM_TELEGRAM_TOKEN")
        if not tg_chat:  missing.append("CEREBRUM_TELEGRAM_CHAT_ID")
        print(f"[live] Telegram alerts DISABLED (missing: {', '.join(missing)})",
              flush=True)

    # Optionally launch the IQ Option collector in a background thread
    if os.environ.get("IQOPTION_SSID") or (
        pathlib.Path(ROOT / ".env").exists()
        and "IQOPTION_SSID=" in pathlib.Path(ROOT / ".env").read_text(encoding="utf-8", errors="ignore")
    ):
        try:
            from src import iq_feed
            t_iq = threading.Thread(target=iq_feed.main, name="iq", daemon=True)
            t_iq.start()
            print("[live] IQ Option collector thread started (SSID detected)", flush=True)
        except Exception as e:
            print(f"[live] failed to start IQ collector: {e}", file=sys.stderr)
    else:
        print("[live] no IQOPTION_SSID in .env – IQ feed disabled", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[live] shutting down…")
        _stop.set()
        t_feed.join(timeout=5)
        t_signal.join(timeout=5)
        print("[live] done.")

if __name__ == "__main__":
    main()