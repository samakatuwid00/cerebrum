"""Telegram / Discord notifier for the Cerebrum Trader Bot.

Reads credentials from environment variables (so secrets stay out of git):
  - CEREBRUM_TELEGRAM_TOKEN   - Bot token from @BotFather
  - CEREBRUM_TELEGRAM_CHAT_ID - Chat ID where alerts should go

If either is missing, `send_message()` becomes a no-op (logs to stdout only).
"""
import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _token() -> str | None:
    return os.getenv("CEREBRUM_TELEGRAM_TOKEN")


def _chat_id() -> str | None:
    return os.getenv("CEREBRUM_TELEGRAM_CHAT_ID")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def send_message(text: str, *, parse_mode: str | None = None) -> bool:
    """Send a message via Telegram.  Returns True on success, False otherwise.

    Always prints the text to stdout so the bot remains useful even without
    network/credentials.  Strips emoji before the local echo so the console
    never hits UnicodeEncodeError, but sends the full UTF-8 string to
    Telegram (which accepts unicode).
    """
    # 1) Always echo locally - ASCII-safe version (no emoji in the prefix
    #    so the bot's own logs work on any Windows console encoding).
    safe = text.encode("ascii", "replace").decode("ascii")
    print(f"[alert] {safe}")
    sys.stdout.flush()

    # 2) Try Telegram if creds present
    token = _token()
    chat_id = _chat_id()
    if not token or not chat_id:
        return False  # no creds - silent no-op

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            return bool(body.get("ok"))
    except Exception as e:
        print(f"[alert] Telegram send failed: {e}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------
# Evidence helpers
# --------------------------------------------------------------------------
def _evidence_block(df: pd.DataFrame, target_hour: int, target_dow: int,
                    n_similar: int = 3) -> str:
    """Build the historical-evidence section for a slot.

    Returns a multi-line string, or "" if there's not enough data.
    """
    if df is None or len(df) < 30:
        return ""
    try:
        # Import here to avoid circular dependency at module load
        from src.pipeline import get_evidence
        ev = get_evidence(df, target_hour=target_hour, target_dow=target_dow,
                          n_similar=n_similar)
    except Exception:
        return ""

    target_hit = ev.get("target_hit_rate")
    target_avg = ev.get("target_avg_move")
    target_n   = ev.get("target_count", 0)
    if target_hit is None or target_n is None or target_n < 5:
        return ""

    wins = int(round(target_hit * target_n))
    line1 = f"   - Hit rate:  {target_hit*100:.1f}% ({wins}/{target_n} wins)"
    line2 = f"   - Avg move:  {target_avg*100:+.3f}%"

    # Best / worst actual move in this slot
    try:
        slot_mask = (df.index.hour == target_hour) & (df.index.dayofweek == target_dow)
        moves = df.loc[slot_mask, "close"].pct_change().dropna()
        if len(moves) >= 5:
            best_idx  = moves.idxmax()
            worst_idx = moves.idxmin()
            line3 = (f"   - Best move: {moves[best_idx]*100:+.2f}%  "
                     f"({best_idx.strftime('%Y-%m-%d')})")
            line4 = (f"   - Worst:     {moves[worst_idx]*100:+.2f}%  "
                     f"({worst_idx.strftime('%Y-%m-%d')})")
        else:
            line3 = line4 = ""
    except Exception:
        line3 = line4 = ""

    body = "\n".join(x for x in (line1, line2, line3, line4) if x)
    return body


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def format_window(result: dict, side: str = "BUY", threshold: float | None = None,
                   asset: str = "XAU/USD", df: Optional[pd.DataFrame] = None,
                   timeframe: str = "30m") -> str:
    """Return a pretty multi-line message for a single buy/sell window.

    Sections:
      1. Header  (side, asset, confidence)
      2. Time    (when, time until)
      3. Stats   (edge, threshold, sample size)
      4. Evidence (if df is passed: hit rate, avg move, best/worst)
      5. Action  (what the user should do manually)

    timeframe: "30m" or "4h" - controls action line language
    """
    when = datetime.fromisoformat(result["window_start_utc"])
    th = threshold if threshold is not None else float(
        os.getenv("CEREBRUM_THRESHOLD") or 0.55
    )
    prob = result.get("probability", 0.0)
    edge = result.get("edge_mean_ret", 0.0)
    n = result.get("n_samples", 0)

    # 1. Header with emoji (send_message() handles ASCII-safe for console)
    if side == "BUY":
        arrow = "\U0001f7e2 BUY"  # green circle
        conf_label = "\U0001f4aa STRONG" if prob >= 0.75 else "\U0001f44d MEDIUM" if prob >= 0.60 else "\U0001f7e1 WEAK"
    else:
        arrow = "\U0001f534 SELL"  # red circle
        conf_label = "\U0001f4aa STRONG" if prob >= 0.75 else "\U0001f44d MEDIUM" if prob >= 0.60 else "\U0001f7e1 WEAK"

    header = f"{arrow} {asset}  |  {conf_label} CONFIDENCE: {prob*100:.0f}%"

    # 2. Time
    now = datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    minutes_until = int((when - now).total_seconds() / 60)
    if minutes_until < 0:
        time_str = f"at {when.strftime('%Y-%m-%d %H:%M UTC')}  (in the past)"
    elif minutes_until < 60:
        time_str = f"at {when.strftime('%Y-%m-%d %H:%M UTC')}  (in {minutes_until} min)"
    elif minutes_until < 60 * 24:
        hours = minutes_until // 60
        mins  = minutes_until % 60
        time_str = (f"at {when.strftime('%Y-%m-%d %H:%M UTC')}  "
                    f"(in {hours}h {mins}m)")
    else:
        days = minutes_until // (60 * 24)
        hours = (minutes_until % (60 * 24)) // 60
        time_str = (f"at {when.strftime('%Y-%m-%d %H:%M UTC')}  "
                    f"(in {days}d {hours}h)")

    # 3. Stats
    stats = (
        f"Edge: {edge*100:+.3f}%  |  "
        f"Threshold: {th:.2f}  |  "
        f"Sample size: n={n}"
    )

    # 4. Evidence
    target_dow = result.get("day_of_week", when.weekday())
    target_hour = result.get("hour", when.hour)
    evidence = _evidence_block(df, target_hour, target_dow) if df is not None else ""

    # 5. Action
    if timeframe == "30m":
        expiry = "30m"
    else:
        expiry = "4h"
    if side == "BUY":
        action = (f"Action: place a {expiry} CALL on {asset} at {when.strftime('%H:%M UTC')}.  "
                  f"(we don't place orders for you - open IQ Option manually)")
    else:
        action = (f"Action: place a {expiry} PUT on {asset} at {when.strftime('%H:%M UTC')}.  "
                  f"(we don't place orders for you - open IQ Option manually)")

    # Assemble
    parts = [header, time_str, stats]
    if evidence:
        parts.append("")
        parts.append("Historical evidence (in-sample, this exact slot):")
        parts.append(evidence)
        parts.append("   (in-sample means the model saw this data; backtest hit-rate is lower)")
    parts.append("")
    parts.append(action)

    return "\n".join(parts)


# --------------------------------------------------------------------------
# CLI self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "window_start_utc": datetime.now(timezone.utc).replace(microsecond=0)
                              .isoformat().replace("+00:00", "Z"),
        "probability": 0.612,
        "edge_mean_ret": 0.0012,
        "sentiment": 0.15,
        "n_samples": 87,
    }
    ok = send_message(format_window(sample, side="BUY"))
    print(f"send_message returned: {ok}")