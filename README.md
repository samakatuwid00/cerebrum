# Cerebrum Trader Bot

A Python trading signal bot for XAU/USD (Gold) that produces timestamped, evidence-backed buy/sell signals via Telegram alerts.

## Features

- **Meta-labeling predictive engine** - ML-based signal generation using weekly trend alignment
- **Multi-timeframe analysis** - 4h primary with 30m hybrid scanner
- **Walk-forward validation** - Deflated Sharpe ratio and purged cross-validation
- **Telegram alerts** - Real-time notifications with BUY/SELL signals
- **Paper trading mode** - Log all decisions without executing trades
- **Live data feeds** - IQ Option WebSocket + FX rate polling
- **Cross-asset drivers** - DXY (US Dollar Index) and US10Y (10Y Treasury yield) as causal, leak-safe features for the 1h model. US10Y moves *inversely* to gold (higher real yields → stronger dollar → weaker gold), so a rising yield nudges the gold-up probability down. DXY and US10Y gate separately; both can be combined later for a compound nudge.

## Architecture

```
cerebrum_live.py          # Main orchestrator (3 threads)
├── fx_feed.py            # USD/EUR polling every 30s
├── signal_loop           # Re-evaluates every 4h boundary
│   ├── meta_live.py      # ML predictive engine
│   ├── pipeline.py       # Legacy slot-table engine
│   └── pipeline_30m.py   # 30-minute scanner
├── alert.py              # Telegram notifications
└── iq_feed.py            # IQ Option WebSocket (optional)
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/cerebrum_trader_bot.git
cd cerebrum_trader_bot
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
# - IQOPTION_SSID (from IQ Option DevTools)
# - CEREBRUM_TELEGRAM_TOKEN (from @BotFather)
# - CEREBRUM_TELEGRAM_CHAT_ID (your chat ID)
```

### 3. Fetch Historical Data

```bash
python data/fetch_xau_4h.py    # 4-hour candles
python data/fetch_xau_1h.py    # 1-hour candles
```

### 4. Run Backtest

```bash
python -m src.backtest
```

### 5. Start Live Bot

```bash
python cerebrum_live.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CEREBRUM_THRESHOLD` | 0.55 | Signal confidence threshold |
| `CEREBRUM_LAMBDA` | 0.3 | News sentiment weight |
| `CEREBRUM_META_HORIZON` | 4 | Prediction horizon (hours) |
| `CEREBRUM_META_THRESHOLD` | 0.50 | Meta-labeling threshold |

## Alert Behavior

- **Every 4 hours** (00, 04, 08, 12, 16, 20 UTC): Full analysis runs
- **BUY/SELL alerts**: Only when probability >= threshold
- **Status updates**: Sent every cycle to confirm bot is running
- **Paper log**: All decisions logged to `data/meta_paper_log.csv`

## Quality Gates

The bot passes validation when:
- Hit rate >= 53%
- Trades >= 200
- Sharpe ratio >= 1.0
- Deflated Sharpe >= 95% (multiple testing correction)
- CV hit rate >= 52%

## Project Structure

```
cerebrum_trader_bot/
├── cerebrum_live.py          # Main entry point
├── config/
│   └── settings.py           # Configuration defaults
├── src/
│   ├── alert.py              # Telegram notifications
│   ├── backtest.py           # Backtesting engine
│   ├── fx_feed.py            # Live FX rates
│   ├── iq_feed.py            # IQ Option WebSocket
│   ├── live_edge.py          # Rolling live edge
│   ├── meta_live.py          # ML meta-labeling
│   ├── ml_edge.py            # ML feature engineering
│   ├── pipeline.py           # Core signal engine
│   ├── pipeline_30m.py       # 30-minute scanner
│   └── sell_window.py        # SELL signal logic
├── data/
│   ├── fetch_xau_4h.py       # Data fetcher
│   └── *.csv                 # Historical data
├── docs/
│   └── *.md                  # Documentation
├── .env.example              # Environment template
└── .gitignore
```

## 24/7 Cloud Deployment

For continuous operation, deploy to a cloud server:

```bash
# On your cloud server (e.g., AWS, DigitalOcean, Hetzner)
git clone https://github.com/yourusername/cerebrum_trader_bot.git
cd cerebrum_trader_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables
export IQOPTION_SSID="your_ssid"
export CEREBRUM_TELEGRAM_TOKEN="your_token"
export CEREBRUM_TELEGRAM_CHAT_ID="your_chat_id"

# Run with nohup or screen
nohup python cerebrum_live.py > bot.log 2>&1 &
```

## Disclaimer

This bot is for educational and paper trading purposes only. It does not place real trades. Always do your own research before trading.

## License

MIT
