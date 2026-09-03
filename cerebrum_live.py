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
    """Re-evaluate BUY + SELL + NO-TRADE signal every CYCLE_SECONDS.

    Loads XAU/USD 1h as primary (explicit filename, not degenerate daily),
    builds baseline edge, and delegates BUY/SELL/WATCH to the MTF-aware
    meta-labeling engine. Legacy slot-table fallback is kept but now
    behind the unified NO-TRADE gate.
    """
    # Explicit 1h feed for intraday (hour,dow) edge; fallback to whatever load_history picks
    try:
        df_hist = pl.load_history("xau_usd_1h_real.csv")
    except FileNotFoundError:
        df_hist = pl.load_history()
    asset = "XAU/USD"
    synthetic_edge = pl.compute_edge(df_hist)
    print(f"[signal] baseline edge loaded ({len(synthetic_edge)} slots, "
          f"{len(df_hist)} bars from {df_hist.index.min().date()} -> {df_hist.index.max().date()})",
          flush=True)
    # Also log live ATR/MACD diag once at startup for operability
    try:
        _atr = pl.compute_atr_pct(df_hist).iloc[-1]
        _, _, _hist = pl.compute_macd(df_hist["close"])
        print(f"[signal] live diag: ATR%={_atr:.4f} MACD hist={_hist.iloc[-1]:.2f}", flush=True)
    except Exception as e:
        print(f"[signal] live diag failed: {e}", flush=True)

    # wait until we have at least one live tick (FX or IQ)
    while not _stop.is_set() and not (LIVE_CSV.exists() or _iq_csv_exists()):
        time.sleep(1)

    _last_4h_slot: dt.datetime | None = None

    while not _stop.is_set():
        try:
            now = dt.datetime.now(pytz.UTC)
            
            # Throttle to once per 4h candle: 00,04,08,12,16,20 UTC, first 5 min only
            slot_hour = (now.hour // 4) * 4
            slot = now.replace(hour=slot_hour, minute=0, second=0, microsecond=0)
            is_new_slot = (_last_4h_slot is None or slot != _last_4h_slot)
            within_window = (now - slot).total_seconds() < 300  # first 5 min of slot
            # Fallback epoch check for clock skew: int(now.timestamp()) % 14400 < 60
            epoch_ok = int(now.timestamp()) % 14400 < 300

            if not (is_new_slot and within_window and epoch_ok):
                if is_new_slot and not within_window:
                    # Past slot but we started mid-candle — wait for next 4h, don't spam old slot
                    pass
                elif not is_new_slot:
                    # Already alerted this 4h candle — debounce
                    pass
                # Sleep and retry; add debug every 30 min to show throttling is alive
                if now.minute % 30 == 0 and now.second < 5:
                    nxt = slot + dt.timedelta(hours=4) if not is_new_slot else slot
                    if not is_new_slot:
                        nxt = _last_4h_slot + dt.timedelta(hours=4) if _last_4h_slot else slot
                    print(f"[throttle] skip {now.strftime('%H:%M:%S')} UTC — next 4h slot {nxt.strftime('%H:%M')} UTC", flush=True)
                pass
            else:
                _last_4h_slot = slot
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

                # ===== PREDICTIVE ENGINE: meta-labeling (MTF-aware, paper mode) =====
                # filename must match df_hist's 1h feed; META_FILE overrides only if set
                meta_filename = META_FILE or "xau_usd_1h_real.csv"
                meta_res = meta_live.predict_meta(filename=meta_filename,
                                                 horizon=META_HORIZON,
                                                 threshold=META_THRESHOLD)
                # Unified BUY / SELL / WATCH(=NO-TRADE) handling
                # meta_res["trade"]=True => BUY/SELL, else WATCH with explicit reason
                if meta_res.get("trade"):
                    side = meta_res["side"]
                    prob = meta_res["take_probability"] or meta_res["probability"]
                    src = meta_res["edge_source"]
                    atr = meta_res.get("atr_pct", None)
                    hist = meta_res.get("macd_hist", None)
                    atr_s = f" ATR%={atr:.4f}" if atr is not None else ""
                    hist_s = f" hist={hist:.2f}" if hist is not None else ""
                    print(f"[META/{side}] prob={prob:.3f} horizon={meta_res['horizon']} "
                          f"thr={meta_res['threshold']} src={src}{atr_s}{hist_s} | {meta_res['reason']}")
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
                    atr = meta_res.get("atr_pct", None)
                    hist = meta_res.get("macd_hist", None)
                    status = alert.format_watch(prob, trend_label, meta_res["trend"], reason, atr, hist)
                    print(status)
                    alert.send_message(status)
                    # Legacy fallback is now *gated* — only emit if meta was high-confidence WATCH
                    # due to weekly neutral, not low-vol/MACD veto. Prevents noisy slot-table spam.
                    if "weekly trend NEUTRAL" in reason or "confidence too low" in reason:
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
                    else:
                        # ATR/MACD vetoed — stay in WATCH, do not spam legacy windows
                        print(f"[FALLBACK] suppressed — MTF veto: {reason}")

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