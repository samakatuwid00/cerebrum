"""IQ Option live-bars collector (WebSocket).

Protocol overview
-----------------
The IQ Option trading WS is **plain JSON-over-WebSocket** (NOT Engine.IO or
Socket.IO).  See https://github.com/LuKks/iqoption for the reference
implementation we mirror.

Frames we send:

    {"name":"authenticate",     "request_id":"<rand>", "local_time":<sec>,
     "msg":{"ssid":"...", "protocol":3, "session_id":"", "client_session_id":""}}
    {"name":"setOptions",       "request_id":"<rand>", "local_time":<sec>,
     "msg":{"sendResults":true}}
    {"name":"subscribeMessage", "request_id":"<rand>", "local_time":<sec>,
     "msg":{"name":"candle-generated", "version":"1.0",
            "params":{"routingFilters":{"active_id":1,"size":1}}}}

Frames we expect:

    {"name":"authenticated", "request_id":"<same>", "msg":{"isSuccessful":true}}
    {"name":"candle-generated", "msg":{
        "active_id":1, "size":1, "at":<attoseconds>,
        "from":<unixsec>, "to":<unixsec>,
        "open":.., "min":.., "max":.., "close":..,
        "ask":.., "bid":.., "volume":0, "phase":"T"}}

EUR/USD live = active_id=1, size=1 (1-min bars).
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

WS_URL = "wss://ws.iqoption.com/echo/websocket"

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from src.config_loader import load_env
load_env()
from config import settings as _settings

SSID = os.getenv("IQOPTION_SSID") or _settings.IQOPTION_SSID or ""

DB_PATH = _pl.Path(__file__).resolve().parent.parent / "data" / "cerebrum.db"

# Parallel CSV mirror so cerebrum_live.py can build a LIVE edge without
# touching the SQLite DB.  Each row is a 1-min bar.
LIVE_CSV = _pl.Path(__file__).resolve().parent.parent / "data" / "iq_live_bars.csv"

EURUSD_ACTIVE_ID = 1   # live EUR/USD on IQ Option
BAR_SIZE_SECONDS = 60  # 1-minute candles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _new_request_id() -> str:
    return f"{int(time.time())}_{int.from_bytes(os.urandom(4), 'big')}"


def _frame(name: str, msg: dict) -> str:
    return json.dumps({
        "name": name,
        "request_id": _new_request_id(),
        "local_time": int(time.time()),
        "msg": msg,
    })


# ---------------------------------------------------------------------------
# DB setup -- one row per 1-minute bar
# ---------------------------------------------------------------------------
def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bars (
            bar_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         REAL NOT NULL,
            open       REAL NOT NULL,
            high       REAL NOT NULL,
            low        REAL NOT NULL,
            close      REAL NOT NULL,
            volume     INTEGER NOT NULL DEFAULT 0,
            feed_source TEXT NOT NULL DEFAULT 'iqoption'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            correlation_id INTEGER,
            ts            TEXT,
            action        TEXT,
            confidence    REAL
        )
        """
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Reconnect / session loop
# ---------------------------------------------------------------------------
async def _connect_and_stream():
    """Top-level reconnect loop with exponential backoff."""
    backoff = 5
    while True:
        try:
            await _run_session()
            backoff = 5  # clean exit -> reset
        except Exception as e:
            print(f"[iq_feed] session error: {type(e).__name__}: {e}; "
                  f"retrying in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def _run_session():
    """One WS session: connect -> authenticate -> subscribe -> read bars."""
    import websockets

    print(f"[iq_feed] connecting to {WS_URL} (SSID len={len(SSID)})", flush=True)
    session_id = ""
    client_session_id = ""
    async with websockets.connect(WS_URL, ping_interval=20) as ws:
        # 1) AUTHENTICATE
        await ws.send(_frame("authenticate", {
            "ssid": SSID,
            "protocol": 3,
            "session_id": "",
            "client_session_id": "",
        }))
        print("[iq_feed] sent authenticate", flush=True)

        # 2) Read frames until we see "authenticated" or 10s elapse.
        #    In the same loop we also handle "front" (carries session_id)
        #    and "timeSync" (server wants us to echo our local time back).
        auth = None
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(),
                                             timeout=max(0.1, deadline - time.time()))
            except asyncio.TimeoutError:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            name = data.get("name")

            if name == "authenticated":
                auth = data
                # capture client_session_id if present
                csid = data.get("client_session_id")
                if csid:
                    client_session_id = csid
                break  # auth done, move on
            elif name == "front":
                # server welcome frame; carries session_id
                sid = data.get("session_id")
                if sid:
                    session_id = sid
                print(f"[iq_feed] front received (session_id={session_id})",
                      flush=True)
            elif name == "timeSync":
                # echo our local time back (server wants the diff)
                await ws.send(json.dumps({
                    "name": "timeSync",
                    "msg": int(time.time() * 1000),  # ms, matching what server sent
                }))
            # else: ignore other frame types during handshake

        if not auth:
            raise RuntimeError("authentication timed out (10s)")

        # 'msg' may be bool (true=ok) or a dict; handle both
        msg = auth.get("msg")
        is_ok = (msg is True) or (isinstance(msg, dict) and msg.get("isSuccessful", False))
        if not is_ok:
            err = msg if isinstance(msg, str) else "unknown"
            raise RuntimeError(f"authentication rejected by IQ Option: {err}")
        print(f"[iq_feed] authenticated OK (client_session_id={client_session_id})",
              flush=True)

        # 3) SET OPTIONS (enable result responses)
        await ws.send(_frame("setOptions", {"sendResults": True}))
        print("[iq_feed] sent setOptions", flush=True)

        # 4) SUBSCRIBE to EUR/USD 1-min candles
        await ws.send(_frame("subscribeMessage", {
            "name": "candle-generated",
            "version": "1.0",
            "params": {
                "routingFilters": {
                    "active_id": EURUSD_ACTIVE_ID,
                    "size": BAR_SIZE_SECONDS,
                }
            },
        }))
        print(f"[iq_feed] subscribed to EUR/USD 1-min candles "
              f"(active_id={EURUSD_ACTIVE_ID}, size={BAR_SIZE_SECONDS}s)",
              flush=True)

        # 5) Read forever; write bars to DB + CSV
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                # idle -> heartbeat by re-issuing setOptions
                await ws.send(_frame("setOptions", {"sendResults": True}))
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[iq_feed] connection closed by server", flush=True)
                return

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            name = data.get("name")
            if name == "candle-generated":
                _write_bar(data.get("msg") or {})
            elif name == "timeSync":
                # echo our local time back so the server keeps the connection
                await ws.send(json.dumps({
                    "name": "timeSync",
                    "msg": int(time.time() * 1000),
                }))
            # else: ignore other acks (setOptions, subscribeMessage, etc.)


async def _wait_for(ws, *, name: str, timeout: float) -> dict | None:
    """Read frames until one matches `name`, or timeout. Returns None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data.get("name") == name:
            return data
    return None


def _write_bar(msg: dict) -> None:
    """Persist one IQ Option candle to SQLite + CSV mirror.

    IQ Option sends a 'candle-generated' event on every tick of the current
    bar (same `from` timestamp, H/L/C grow as price moves).  We want ONE row
    per bar, so we INSERT a new row only when the `from` changes; otherwise
    we UPDATE the existing row's H/L/C/volume.
    """
    open_p = msg.get("open")
    high   = msg.get("max")     # IQ sends "max" not "high"
    low    = msg.get("min")     # IQ sends "min" not "low"
    close  = msg.get("close")
    volume = msg.get("volume", 0) or 0
    from_ts = msg.get("from")   # unix seconds, start of bar
    if None in (open_p, high, low, close, from_ts):
        print(f"[iq_feed] skipping incomplete bar: {msg}", flush=True)
        return
    ts_sec = float(from_ts)

    # SQLite: INSERT new bar, or UPDATE the existing one (same ts).
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM bars WHERE ts = ? AND feed_source = 'iqoption' LIMIT 1",
            (ts_sec,),
        )
        exists = cur.fetchone() is not None
        if exists:
            cur.execute(
                "UPDATE bars SET high = MAX(high, ?), "
                "low = MIN(low, ?), close = ?, volume = ? "
                "WHERE ts = ? AND feed_source = 'iqoption'",
                (high, low, close, int(volume), ts_sec),
            )
        else:
            cur.execute(
                "INSERT INTO bars (ts, open, high, low, close, volume, feed_source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts_sec, open_p, high, low, close, int(volume), "iqoption"),
            )
        con.commit()
    except Exception as e:
        print(f"[iq_feed] DB insert/update error: {e}", file=sys.stderr)
    finally:
        con.close()

    # CSV mirror: same logic -- only the first bar of a minute starts a row,
    # subsequent ticks just overwrite the last line.
    try:
        LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
        if not LIVE_CSV.exists():
            LIVE_CSV.write_text("datetime,open,high,low,close,volume\n",
                                encoding="utf-8")
        # Read existing last line to check ts
        lines = LIVE_CSV.read_text(encoding="utf-8").strip().split("\n")
        last_line = lines[-1] if len(lines) > 1 else ""
        is_same_bar = last_line.startswith(_to_utc_iso(ts_sec))
        if is_same_bar:
            # replace last line
            new_last = (f"{_to_utc_iso(ts_sec)},{open_p:.6f},{high:.6f},"
                        f"{low:.6f},{close:.6f},{int(volume)}\n")
            LIVE_CSV.write_text("\n".join(lines[:-1] + [new_last]) + "\n",
                                encoding="utf-8")
        else:
            with LIVE_CSV.open("a", encoding="utf-8") as f:
                f.write(
                    f"{_to_utc_iso(ts_sec)},{open_p:.6f},{high:.6f},"
                    f"{low:.6f},{close:.6f},{int(volume)}\n"
                )
    except Exception as e:
        print(f"[iq_feed] CSV mirror error: {e}", file=sys.stderr)

    print(
        f"[iq_feed] bar {_to_utc_iso(ts_sec)}  "
        f"O={open_p:.5f}  H={high:.5f}  L={low:.5f}  C={close:.5f}  V={volume}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if not SSID:
        print(
            "WARNING: IQOPTION_SSID not set - falling back to open.er.api.com FX feed.",
            file=sys.stderr,
        )
        return 0

    _ensure_db()
    print(f"[iq_feed] starting IQ Option feed -> DB: {DB_PATH}", flush=True)
    try:
        asyncio.run(_connect_and_stream())
    except KeyboardInterrupt:
        print("\n[iq_feed] stopped by user", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())