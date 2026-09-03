#!/usr/bin/env bash
# bootstrap_hermes.sh — Hermes post-clone setup for 24/7 paper mode
# Usage (Hermes chat):
#   git clone <repo> cerebrum_trader_bot && cd cerebrum_trader_bot
#   bash scripts/bootstrap_hermes.sh
# Then create .env via Hermes chat (see step 3), then:
#   bash scripts/bootstrap_hermes.sh --run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:- --setup}"

log() { echo "[bootstrap] $*"; }

# 1. Python venv (Linux - recreates, never copy Windows .venv_cerebrum)
if [ ! -d ".venv" ]; then
  log "Creating .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
log "Upgrading pip + installing requirements ..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
log "pip done: $(python --version) / $(pip --version | cut -d' ' -f1-2)"

# 2. Fetch fresh market data (data/*.csv is gitignored on purpose - regenerate)
if [ ! -f "data/xau_usd_1h_real.csv" ] || [ ! -f "data/xau_usd_daily_real.csv" ]; then
  log "Fetching XAU history (yfinance) ..."
  python data/fetch_xau_1h.py || log "WARN: fetch_xau_1h failed - check yfinance/network"
  python data/fetch_xau_daily.py || log "WARN: fetch_xau_daily failed"
else
  log "Data exists: $(wc -l < data/xau_usd_1h_real.csv) 1h bars, $(wc -l < data/xau_usd_daily_real.csv) daily bars"
fi
# Optional: ensure weekly file is built
if [ ! -f "data/xau_usd_weekly_real.csv" ]; then
  log "Weekly will be auto-built on first run"
fi

# 3. .env handling - NEVER commit this file (gitignored)
if [ ! -f ".env" ]; then
  log "No .env found - creating template from .env.example"
  cp .env.example .env
  cat <<'EOF'
[bootstrap] >>> ACTION NEEDED in Hermes chat <<<
Create your live .env now (paste your 3 secrets). In Hermes chat run:

cat > .env << 'EOV'
IQOPTION_SSID=your_long_hex_from_iqoption
CEREBRUM_TELEGRAM_TOKEN=your_bot_token_from_BotFather
CEREBRUM_TELEGRAM_CHAT_ID=your_chat_id
# optional tuning (defaults are fine for paper test)
CEREBRUM_THRESHOLD=0.55
CEREBRUM_META_THRESHOLD=0.50
EOV
cat .env   # verify

Then re-run: bash scripts/bootstrap_hermes.sh --run
EOF
  # Don't exit as error - let Hermes see the message
  if [ "$MODE" != "--run" ]; then
    exit 0
  fi
fi

# Validate .env has tokens (don't print them)
if ! grep -q "CEREBRUM_TELEGRAM_TOKEN=..*" .env || grep -q "CEREBRUM_TELEGRAM_TOKEN=$" .env; then
  log "WARN: CEREBRUM_TELEGRAM_TOKEN looks empty in .env - Telegram alerts will be DISABLED"
fi

# 4. Smoke tests before 24/7
log "Running smoke tests ..."
python -m py_compile cerebrum_live.py src/pipeline.py src/meta_live.py src/backtest.py
python -c "from src.pipeline import load_history; df=load_history(); print(f'[smoke] load_history {len(df)} bars {df.index.min()} -> {df.index.max()}')"
python -c "from src.meta_live import predict_meta; r=predict_meta(filename='xau_usd_1h_real.csv'); print(f\"[smoke] meta_live {r['side']} p={r['probability']} trend={r['trend_label']} | {r['reason'][:90]}\")"

if [ "$MODE" != "--run" ]; then
  cat <<'EOF'
[bootstrap] Setup done.
Next: bash scripts/bootstrap_hermes.sh --run   # starts 24/7 paper mode
EOF
  exit 0
fi

# 5. Run 24/7 paper mode (Hermes keeps it alive)
# Prefer systemd if available and user has sudo, else nohup
log "Starting cerebrum_live.py in paper mode (24/7) ..."
# Kill old nohup if any (idempotent)
pkill -f "cerebrum_live.py" || true
sleep 1

# Use nohup so it survives Hermes chat disconnect
nohup .venv/bin/python cerebrum_live.py > bot.log 2>&1 &
PID=$!
sleep 2
if kill -0 "$PID" 2>/dev/null; then
  log "Started PID $PID - tail bot.log:"
  tail -n 30 bot.log || true
  log "Check Telegram for 'Cerebrum Trader Bot — LIVE MODE' and META STATUS every 4h at 00,04,08,12,16,20 UTC"
  log "Logs: tail -f bot.log  |  paper trades: tail -f data/meta_paper_log.csv"
  # Also offer systemd alternative
  if command -v systemctl >/dev/null 2>&1; then
    log "Tip: for reboot-survival, run: sudo cp scripts/cerebrum.service /etc/systemd/system/ && sudo systemctl enable --now cerebrum"
  fi
else
  log "FAILED to start - check bot.log:"
  cat bot.log || true
  exit 1
fi
