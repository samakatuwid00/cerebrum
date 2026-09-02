# Backtest Harness Plan — cerebrum_trader_bot

**Status:** DRAFT v1 — matches actual repo state on Sep 1 2026
**Target asset:** XAU/USD (gold vs USD), 4-hour candles
**Current backtest state:** `src/backtest.py` exists but is hardcoded for EUR/USD 30-min
**Current data state:** `data/fetch_xau_4h.py` works, `data/xau_usd_4h_real.csv` is the target file

---

## 1. Goal

Extend the existing backtest harness so it works against XAU/USD 4h data **without breaking** the EUR/USD 30-min path. Then add the LLM weekly journal review as the killer feature.

The harness must answer 4 questions per strategy:

1. Does the edge exist in-sample? (hit_rate, avg return, total PnL)
2. Does it survive walk-forward validation? (no lookahead)
3. What's the realistic risk profile? (Sharpe, max drawdown, profit factor)
4. Is it tradeable live? (passes the 53% / 200-trade quality gate, then runs paper)

---

## 2. Current State Audit (what's already in the repo)

| File | What it does | Status |
|---|---|---|
| `src/backtest.py` | Vectorized + walk-forward backtest, EUR/USD 30-min, hour-minute slot edge | ✅ Working, but EUR/USD-specific |
| `src/pipeline.py` | `load_history()`, `compute_edge()`, `combine_probability()`, `load_sentiment_score()` | ✅ Core logic, but reads `usd_eur_hourly.csv` |
| `src/live_edge.py` | Aggregates live IQ 1-min bars into 1h OHLC, computes hourly edge | ✅ Works, uses `iq_live_bars.csv` |
| `src/sentiment.py` | News sentiment overlay | ✅ Works |
| `src/alert.py` | Telegram alerts | ✅ Works, disabled by default |
| `data/fetch_xau_4h.py` | Yahoo Finance XAU/USD 1h → 4h aggregation | ✅ Working, runs but file output not verified yet |
| `data/usd_eur_hourly*.csv` | Existing EUR/USD data | ✅ |
| `data/iq_live_bars.csv` | Live IQ feed bars | ⚠️ Probably empty/stale |

**Critical gap:** `pipeline.load_history()` and `compute_edge()` hardcode `data/usd_eur_hourly.csv` and 30-min slot logic. The XAU/USD path requires:
- Reading from `data/xau_usd_4h_real.csv`
- Using **4-hour slots** instead of (hour, minute) — too sparse otherwise
- Gold-specific sentiment (DXY, real yields, Fed language) vs forex sentiment

---

## 3. Target Architecture (final, after merge)

```
cerebrum_trader_bot/
├── data/
│   ├── xau_usd_4h_real.csv           # ← PRIMARY: Yahoo Finance 4h bars
│   ├── usd_eur_hourly_30min_real.csv  # ← SECONDARY: paper-trade companion
│   ├── iq_live_bars.csv              # ← live feed (XAU/USD from cerebrum_live)
│   └── journal/
│       └── trades_YYYY-MM-DD.md      # ← NEW: LLM-readable trade journal
├── src/
│   ├── backtest.py                   # REFACTOR: support multi-asset, multi-TF
│   ├── pipeline.py                   # REFACTOR: load_history() takes (asset, tf)
│   ├── live_edge.py                  # ADAPT: aggregate to 4h for XAU/USD
│   ├── paper_trader.py               # ← NEW: live paper trading, no real money
│   ├── journal.py                    # ← NEW: auto-log every trade to markdown
│   ├── journal_review.py             # ← NEW: weekly LLM review CLI
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                   # Strategy interface (entry/sl/tp logic)
│   │   ├── xau_4h_ema_pullback.py    # ← Strategy #1: 20-EMA pullback on 4h
│   │   ├── xau_4h_session_break.py   # ← Strategy #2: London/NY session break
│   │   └── eur_30m_hour_edge.py      # ← Migrated from current pipeline
│   └── (existing files unchanged)
├── docs/
│   ├── BACKTEST_HARNESS_PLAN.md      # ← this file
│   ├── STRATEGY_PLAYBOOK.md          # ← rules for each strategy in plain English
│   └── JOURNAL_REVIEW_PROMPT.md      # ← LLM prompt for weekly review
└── config/
    └── settings.py                   # EXTEND: asset selection, TF, threshold
```

---

## 4. Implementation Phases

### Phase 1 — Multi-asset data plumbing (Day 1–2)

**Why first:** Everything else depends on this. No point writing strategies if the harness can't load XAU/USD.

1. Refactor `src/pipeline.py`:
   - `load_history(asset="XAUUSD", timeframe="4h")` — switches CSV based on params
   - `compute_edge(df, slot="4h")` — bucket by 4-hour window instead of (hour, minute)
   - Keep existing EUR/USD path working via default params

2. Verify `data/fetch_xau_4h.py` produces a usable file:
   ```bash
   PYTHONPATH=. .venv_cerebrum/Scripts/python.exe data/fetch_xau_4h.py
   head -5 data/xau_usd_4h_real.csv
   wc -l data/xau_usd_4h_real.csv   # expect ≥2000 rows (2y × 6 bars/day)
   ```

   **VERIFIED Sep 1 2026:** Yahoo Finance delisted `XAUUSD=X`; switched ticker to
   `GC=F` (COMEX gold futures). Sanity range widened to $2,000–$5,000.
   Pull returned **3,037 4h bars from 2024-09-02 → 2026-09-01**, close min/max
   $2,504 / $5,590 (one outlier bar above $5,000 — flag for Phase 3 outlier
   handling). Mean close $3,703. File verified clean.

3. Update `config/settings.py`:
   ```python
   ASSET: str = "XAUUSD"           # was implicit USD/EUR
   TIMEFRAME: str = "4h"           # was 30min
   DATA_PATH: str = "data/xau_usd_4h_real.csv"
   ```

**Done when:** `python -m src.backtest --asset XAUUSD --tf 4h` runs end-to-end on the existing EUR/USD code path.

---

### Phase 2 — Strategy interface + first XAU/USD strategy (Day 3–5)

1. Create `src/strategies/base.py` with the Strategy contract:
   ```python
   class Strategy:
       name: str
       def signal(self, df: pd.DataFrame) -> Signal | None:
           """Return Signal(side, sl, tp) or None."""
       def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
           """Walk df, return trades DataFrame."""
   ```

2. Port the existing edge logic to `src/strategies/eur_30m_hour_edge.py` as the reference implementation.

3. Implement **Strategy #1: XAU/USD 4h EMA pullback** (your first real XAU strategy):
   - **Entry:** Price pulls back to 20-EMA on 4h, RSI(14) between 40–60, in the direction of the daily 50-EMA trend
   - **Stop:** 1.5 × ATR(14) on 4h behind the entry
   - **Target:** 3 × ATR (R:R = 2:1)
   - **Filter:** No trades 30 min before/after high-impact USD news (NFP, CPI, FOMC)

4. Run backtest against `data/xau_usd_4h_real.csv`:
   ```bash
   PYTHONPATH=. .venv_cerebrum/Scripts/python.exe -m src.backtest \
       --asset XAUUSD --tf 4h --strategy xau_4h_ema_pullback \
       --start 2022-01-01 --end 2025-12-31
   ```

**Done when:** You have a trades DataFrame with ≥ 50 trades, run/print the quality gate (53% / 200 trades — gold may need a softer gate like 52% / 100 trades because 4h bars produce fewer signals than 30-min).

---

### Phase 3 — Walk-forward + metrics dashboard (Day 6–8)

1. Refactor `run_backtest_walkforward()` in `src/backtest.py` to:
   - Train on rolling 6-month window
   - Test on next 1-month window
   - Stitch out-of-sample trades into a single equity curve
   - Compute: hit_rate, profit_factor (= gross_wins / gross_losses), Sharpe, max DD, avg win / avg loss, expectancy per trade

2. Add per-strategy HTML report (similar to existing `render_equity_curve.py`):
   - Equity curve chart
   - Drawdown underwater chart
   - Hour-of-day / day-of-week breakdown
   - Distribution of trade durations

3. Define the **XAU/USD quality gate** (softer than EUR/USD 30-min):
   - Hit rate ≥ 52%
   - Total trades ≥ 100 (4h bars are sparse)
   - Profit factor ≥ 1.3
   - Max drawdown ≤ 20%
   - Walk-forward hit rate within 3% of in-sample

**Done when:** All 5 gate criteria can be checked with one CLI flag: `--gate`.

---

### Phase 4 — Paper trader + auto-journal (Day 9–12)

This is where it gets useful in practice, not just theoretical.

1. Create `src/paper_trader.py`:
   - Polls IQ Option WS for XAU/USD 4h candle closes (same WS code path as `cerebrum_live.py`)
   - Runs each enabled strategy → gets signal → **does NOT execute**, just logs
   - Tracks running P&L against a virtual $10,000 account
   - Writes every "would-have-traded" decision to `data/journal/trades_YYYY-MM-DD.md`

2. Journal markdown format (one row per signal):
   ```markdown
   ## 2026-09-01 16:00 UTC — XAU/USD 4h
   
   - Strategy: xau_4h_ema_pullback
   - Side: BUY
   - Entry: 2487.50 | TP: 2515.00 | SL: 2470.00
   - Reasoning: 4h pullback to 20-EMA, RSI=52, daily trend up, no news in next 4h
   - Outcome (filled in at close): WIN +27.50 / +1.10% / R=+2.0
   
   ## 2026-09-02 20:00 UTC — XAU/USD 4h
   ...
   ```

3. Auto-update journal at bar close using `live_edge.compute_live_edge()` (already exists).

**Done when:** `python -m src.paper_trader` runs in the background, journal grows daily, you can review a week of paper trades by reading one markdown file.

---

### Phase 5 — Weekly LLM review (Day 14, the killer feature)

1. `src/journal_review.py` is a thin wrapper:
   ```python
   # Reads last 7 days of journal, builds review prompt, calls local LLM via OmniRoute
   ```

2. `docs/JOURNAL_REVIEW_PROMPT.md` contains the structured prompt the LLM gets. Pattern:

   ```
   You are a trading coach reviewing my last 7 days of paper trades.
   
   TRADES:
   {paste journal markdown}
   
   Find:
   1. Recurring mistakes (entry timing, stop placement, holding losers)
   2. Time-of-day / day-of-week biases
   3. Emotional patterns (trades taken after a loss, revenge trades, etc.)
   4. Strategy-specific weaknesses (which strategy had the worst R:R distribution?)
   5. Top 3 actionable changes for next week
   
   Be specific, blunt, and reference trade numbers.
   ```

3. Output gets saved to `data/journal/weekly_reviews/YYYY-MM-DD.md` so you can compare weeks.

**This is the highest-leverage AI use case in your whole trading pipeline.** A human coach costs $200/hr; this is free, runs every Sunday, and grounds the LLM in your actual trades — not generic advice.

---

### Phase 6 — Live merge (Day 15+, gated)

**Hard precondition:** 30+ paper trades, walk-forward passed, journal reviewed for 2+ weeks, you personally agree the strategy has edge.

1. Add `--live` flag to `cerebrum_live.py` (or new `src/cerebrum_live_xau.py`)
2. Wire strategies → broker WS → position sizing (1% risk per trade, fixed)
3. Real-time journal gets appended alongside paper trades
4. Telegram alerts enabled (`ENABLE_TELEGRAM = True` in `config/settings.py`)

**This step is intentionally last.** The whole point of phases 1–5 is to make this safe.

---

## 5. Asset-Specific Notes for XAU/USD

**Why gold is harder than EUR/USD 30-min:**

- **Wide intraday range:** 200–500 pips/day vs EUR/USD 50–80
- **News-driven:** DXY, real yields, Fed speakers can move it 50 pips in minutes
- **Session structure matters:** Asia = ranging, London = trend start, NY = continuation or reversal
- **Round-number psychology:** 2400, 2450, 2500 attract liquidity grabs
- **Spreads widen** during news, around market close, and on Mondays

**Implications for the harness:**

- 4h timeframe is correct — 5m/15m is noise on gold unless you're a scalper
- Position sizing must be in **dollar risk**, not pip count — a 100-pip SL on gold at 0.10 lot = $10, not $10 on EUR/USD
- Slippage assumption: 1.5× ATR on entry for backtest realism
- News filter is non-optional for XAU/USD (see Phase 2 strategy filter)

---

## 6. Risk Management Defaults (XAU/USD 4h)

| Setting | Value | Rationale |
|---|---|---|
| Risk per trade | 1% of account | Standard beginner rule |
| Max open risk | 3% of account | Allows 2–3 concurrent setups |
| Max daily loss | 2% of account | Stop trading for the day |
| Max weekly loss | 5% of account | Stop trading for the week |
| R:R minimum | 2:1 | Math: even 40% win rate is profitable |
| Max holding time | 7 days (28 × 4h bars) | If it hasn't worked by then, it won't |
| Slippage assumption (backtest) | 1.5 × ATR(14) | Conservative |

---

## 7. Verification Gates (this plan is "done" when...)

- [ ] `data/xau_usd_4h_real.csv` exists with ≥ 2000 rows and clean OHLC
- [ ] `src/pipeline.py` accepts `(asset, timeframe)` params without breaking EUR/USD path
- [ ] `src/strategies/base.py` exists with at least 1 XAU/USD strategy implementation
- [ ] `src/backtest.py` produces trades_df + summary dict with: hit_rate, profit_factor, sharpe, max_dd, by_hour
- [ ] Quality gate check is a single CLI flag
- [ ] `src/paper_trader.py` runs continuously, writes to `data/journal/`
- [ ] `src/journal_review.py` produces a weekly review markdown file
- [ ] At least 2 weeks of paper trading completed with journal review before any live capital

---

## 8. Open Questions for User (nikoo)

1. **Account size for paper trading?** (default $10,000 in plan — adjust for your real account)
2. **Allowed trade sessions?** (London only / NY only / both? — affects strategy filter)
3. **News source for the filter?** (ForexFactory CSV is what the current pipeline uses; would you want a live news API like Finnhub?)
4. **Should the EUR/USD 30-min path stay alive or get retired?** (proposal: keep it as a parallel paper-trade companion for learning, retire from live)
5. **Telegram alerts on what?** (signal-only / signal+outcome / daily summary / weekly review only?)