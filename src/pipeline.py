"""Core engine for the Cerebrum Trader Bot.

Responsibilities
----------------
* Load real XAU/USD 4-hour history (preferred) or fall back to EUR/USD 30-min.
* Compute statistical edge per (hour, day_of_week) slot for 4h bars.
* Combine edge + multi-feature model into a probability per slot.
* Find the next slot whose probability exceeds a threshold.
* Provide historical evidence per target slot.

Output rows are OHLCV with columns:
    datetime, open, high, low, close, volume
"""
from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ----------------------------------------------------------------------------- #
# Paths & loading
# ----------------------------------------------------------------------------- #
XAU_DAILY_CSV = DATA / "xau_usd_daily_real.csv"           # preferred (Gold daily, 26y)
XAU_1H_CSV    = DATA / "xau_usd_1h_real.csv"               # fallback (Gold 1h, 730d)
XAU_4H_CSV    = DATA / "xau_usd_4h_real.csv"               # fallback (Gold 4h)
EUR_30M_CSV   = DATA / "usd_eur_hourly_30min_real.csv"     # fallback (EUR 30m)
SYNTH_CSV     = DATA / "usd_eur_hourly_synthetic.csv"      # last-resort fallback
OLD_HOURLY_CSV = DATA / "usd_eur_hourly.csv"               # legacy alias
DXY_1H_CSV    = DATA / "dxy_1h_real.csv"                   # cross-asset: US Dollar Index (1h)
US10Y_1H_CSV  = DATA / "us10y_1h_real.csv"                 # cross-asset: 10Y Treasury yield (1h, ^TNX percent)
ECON_CAL_CSV  = DATA / "economic_calendar.csv"             # hardcoded high-impact US events 2024-2026

# Weekly timeframe (computed from daily)
XAU_WEEKLY_CSV = DATA / "xau_usd_weekly_real.csv"          # computed from daily


def _load_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # ensure tz-aware UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def compute_weekly_bars(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV bars to weekly (Monday-Sunday, anchored to Monday 00:00 UTC).
    
    Returns DataFrame with columns: open, high, low, close, volume
    indexed by week-start datetime (Monday 00:00 UTC).
    """
    # Ensure we have a proper datetime index
    df = daily_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Daily DataFrame must have DatetimeIndex")
    
    # Resample to weekly, anchored to Monday
    # 'W-MON' = week ending Monday (which means week starts previous Monday)
    weekly = df.resample('W-MON').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # The index is the week END (Monday), shift to week START (previous Monday)
    weekly.index = weekly.index - pd.Timedelta(days=6)
    
    return weekly


def load_weekly_history() -> pd.DataFrame:
    """Load or compute weekly XAU/USD bars from daily data."""
    if XAU_WEEKLY_CSV.exists():
        return _load_csv(XAU_WEEKLY_CSV)
    
    # Compute from daily
    daily = _load_csv(XAU_DAILY_CSV)
    weekly = compute_weekly_bars(daily)
    
    # Save for future use
    weekly.to_csv(XAU_WEEKLY_CSV)
    print(f"[pipeline] Computed {len(weekly)} weekly bars from daily data, saved to {XAU_WEEKLY_CSV}")
    
    return weekly


def compute_weekly_trend(weekly_df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute weekly trend using SMA.
    Returns Series with values:
      +1 = uptrend (close > SMA)
      -1 = downtrend (close < SMA)
      0 = neutral/flat (very close to SMA)
    """
    sma = weekly_df['close'].rolling(window=period, min_periods=period).mean()
    diff = weekly_df['close'] - sma
    # Neutral band: within 0.5% of SMA
    neutral_band = 0.005 * sma
    
    trend = pd.Series(0, index=weekly_df.index, dtype=int)
    trend[diff > neutral_band] = 1    # uptrend
    trend[diff < -neutral_band] = -1  # downtrend
    # else 0 (neutral)
    
    return trend


def get_weekly_trend_at(weekly_df: pd.DataFrame, target_dt: pd.Timestamp) -> int:
    """
    Get weekly trend (+1, -1, 0) for the week containing target_dt.
    
    Uses the most recent complete weekly bar at or before target_dt.
    """
    trend = compute_weekly_trend(weekly_df)
    # Find the latest week <= target_dt
    valid = trend[trend.index <= target_dt]
    if len(valid) == 0:
        return 0  # no data yet
    return int(valid.iloc[-1])


# ----------------------------------------------------------------------------- #
# Multi-timeframe probability combiner
# ----------------------------------------------------------------------------- #
def combine_probability_mtf(edge_row: pd.Series,
                            daily_rsi: float = 50.0,
                            daily_bollinger_pos: float = 0.5,
                            daily_momentum: float = 0.0,
                            weekly_trend: int = 0,
                            lam: float = 0.3,
                            sentiment: float = 0.0,
                            rsi_boost: float = 0.05,
                            bb_boost: float = 0.05,
                            mom_boost: float = 0.05,
                            mtf_boost: float = 0.10) -> float:
    """
    Multi-timeframe probability combiner.
    
    Core: Sharpe-logistic probability (edge + sentiment)
    + RSI/Bollinger/Momentum nudges (from combine_probability_full)
    + Weekly trend alignment nudge (NEW):
       - If weekly trend is +1 (uptrend) and daily signal is BUY -> +mtf_boost
       - If weekly trend is -1 (downtrend) and daily signal is SELL -> +mtf_boost
       - If weekly trend opposes daily signal -> -mtf_boost (penalty)
       - If weekly trend is 0 (neutral) -> no change
    
    This acts as a FILTER: opposing weekly trend reduces probability,
    potentially below threshold, preventing counter-trend trades.
    """
    # Build base probability with full feature stack
    base = combine_probability_full(
        edge_row, 
        rsi=daily_rsi, 
        bollinger_pos=daily_bollinger_pos, 
        momentum=daily_momentum,
        lam=lam, 
        sentiment=sentiment,
        rsi_boost=rsi_boost, 
        bb_boost=bb_boost, 
        mom_boost=mom_boost,
    )
    
    # Determine daily signal direction from live probability (not stale slot side)
    # base is P(up) after RSI/BB/momentum nudges; >=0.5 => BUY bias, <0.5 => SELL bias
    daily_side = "BUY" if base >= 0.5 else "SELL"
    # Allow explicit override if caller passed side on the row (live path)
    if "side" in edge_row and edge_row.get("side") in ("BUY", "SELL"):
        daily_side = edge_row.get("side")
    
    # Weekly trend alignment — filter, not predictor
    if weekly_trend == 1 and daily_side == 'BUY':
        base += mtf_boost
    elif weekly_trend == -1 and daily_side == 'SELL':
        # SELL confidence is 1 - P(up), so boosting SELL means lowering P(up)
        base -= mtf_boost
    elif weekly_trend == 1 and daily_side == 'SELL':
        # Counter-trend SELL in uptrend -> penalize SELL (raise P(up) toward 0.5)
        base += mtf_boost
    elif weekly_trend == -1 and daily_side == 'BUY':
        # Counter-trend BUY in downtrend -> penalize BUY
        base -= mtf_boost
    # weekly_trend == 0 -> no change
    
    return float(round(max(0.01, min(0.99, base)), 4))


def load_history(filename: Optional[str] = None) -> pd.DataFrame:
    """Load the best available history.

    Priority when filename is explicit: that file.
    Priority when filename is None (live / backtest default):
        1. data/xau_usd_1h_real.csv           (Gold 1h, yfinance) — preferred intraday
        2. data/xau_usd_4h_real.csv           (Gold 4h, yfinance, fallback)
        3. data/xau_usd_daily_real.csv        (Gold daily, only for weekly generation)
        3b. data/xau_usd_weekly_real.csv      (should not be used as bar feed)
        4. data/usd_eur_hourly_30min_real.csv (EUR 30m fallback)
        5. data/usd_eur_hourly_synthetic.csv  (kept as fallback)
        6. data/usd_eur_hourly.csv            (legacy synthetic)

    NOTE: XAU_DAILY_CSV is deliberately *not* in the auto-pick list for
    intraday engines — its hour is always 00:00, so (hour,dow) edge is
    meaningless and backtests on it look artificially good. Callers that
    truly want daily must pass filename explicitly.
    """
    if filename:
        return _load_csv(DATA / filename)
    for p in (XAU_1H_CSV, XAU_4H_CSV, EUR_30M_CSV, SYNTH_CSV, OLD_HOURLY_CSV):
        if p.exists():
            return _load_csv(p)
    # Daily only as last resort with explicit warning
    if XAU_DAILY_CSV.exists():
        import warnings
        warnings.warn(f"load_history() falling back to daily {XAU_DAILY_CSV.name} — (hour,dow) edge will be degenerate. Pass filename explicitly if daily is intended.")
        return _load_csv(XAU_DAILY_CSV)
    raise FileNotFoundError(
        f"No data file found in {DATA}. Run data/fetch_xau_1h.py first."
    )


# Back-compat shim for code that imports the old name.
def load_hourly_parquet(filename: str = "usd_eur_hourly.csv") -> pd.DataFrame:
    """Legacy loader name; now reads CSV only (parquet was never produced)."""
    return load_history(filename)


# ----------------------------------------------------------------------------- #
# Edge computation (per slot)
# ----------------------------------------------------------------------------- #
def compute_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-slot statistical edge grouped by (hour, day_of_week).

    For 4-hour bars this gives 6 hour-of-day slots × 7 days = 42 unique slots,
    each with ~70-80 samples in 2 years of data — much more meaningful than
    30-minute bars where most slots have fewer than 50 samples.

    Output columns: hour, day_of_week, mean_ret, std_ret, win_rate, count.
    Slots with fewer than 30 samples are dropped (too noisy).
    """
    # If df already has 'ret' column (from backtest), drop it
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    ret = df["close"].pct_change().rename("ret")
    d = df.join(ret)
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek  # Monday=0, Sunday=6
    grp = d.groupby(["hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    edge = edge[edge["count"] >= 30].copy()
    return edge.reset_index(drop=True)


# ----------------------------------------------------------------------------- #
# RSI feature (Wilder's smoothing)
# ----------------------------------------------------------------------------- #
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI on a price series.

    Uses Wilder's exponential smoothing (equivalent to EMA with alpha=1/period)
    so the result matches TradingView/MetaTrader conventions. Returns values
    in [0, 100]; first `period` bars are NaN.
    """
    s = pd.Series(series).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (all gains), RSI is exactly 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


# ----------------------------------------------------------------------------- #
# Edge with optional RSI / Bollinger / Momentum features (per slot)
# ----------------------------------------------------------------------------- #
def compute_edge_with_features(df: pd.DataFrame, period: int = 14,
                                min_count: int = 30,
                                bb_period: int = 20, bb_std: float = 2.0,
                                mom_period: int = 10) -> pd.DataFrame:
    """Per-(hour, day_of_week) edge table with extra feature columns.

    Adds three per-bar features (computed at close):
      * rsi           : Wilder RSI(period), in [0, 100]
      * bollinger_pos : position within Bollinger(bb_period, bb_std) bands, [0, 1]
      * momentum      : simple N-bar percentage return, dimensionless

    The aggregated edge table reports the slot-level *mean* of each feature:
      * rsi_at_open
      * bollinger_pos_at_open
      * momentum_at_open

    Falls back to compute_edge() (with NaN feature columns) if there isn't
    enough data to compute features on at least `min_count` bars.
    """
    base = compute_edge(df)
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    rsi = compute_rsi(df["close"], period=period)
    bb = compute_bollinger_position(df["close"], period=bb_period, std_dev=bb_std)
    mom = compute_momentum(df["close"], period=mom_period)
    d = df[["close"]].copy()
    d["ret"] = df["close"].pct_change()
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["rsi"] = rsi
    d["bollinger_pos"] = bb
    d["momentum"] = mom
    # Use only bars where ALL features are defined
    d = d.dropna(subset=["rsi", "ret", "bollinger_pos", "momentum"])
    if len(d) < min_count:
        # Not enough data — return base edge with NaN feature columns so the
        # caller can detect the fallback.
        out = base.copy()
        out["rsi_at_open"] = np.nan
        out["bollinger_pos_at_open"] = np.nan
        out["momentum_at_open"] = np.nan
        return out
    grp = d.groupby(["hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    rsi_by_slot = grp["rsi"].mean().rename("rsi_at_open").reset_index()
    bb_by_slot = grp["bollinger_pos"].mean().rename("bollinger_pos_at_open").reset_index()
    mom_by_slot = grp["momentum"].mean().rename("momentum_at_open").reset_index()
    edge = edge.merge(rsi_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(bb_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(mom_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge[edge["count"] >= min_count].copy()
    return edge.reset_index(drop=True)
    # ----------------------------------------------------------------------------- #
# ATR feature (Average True Range) — volatility filter
# ----------------------------------------------------------------------------- #
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).
    
    Returns ATR values in price units (same as close). First `period` bars are NaN.
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder smoothing == EMA with alpha = 1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR as percentage of close price (dimensionless, comparable across assets).
    """
    atr = compute_atr(df, period)
    return atr / df['close']


# ----------------------------------------------------------------------------- #
# MACD feature (Moving Average Convergence Divergence)
# ----------------------------------------------------------------------------- #
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard MACD with signal line and histogram.
    
    Returns (macd_line, signal_line, histogram)
    """
    s = pd.Series(series).astype(float)
    ema_fast = s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ----------------------------------------------------------------------------- #
# Edge with ATR + MACD features (per slot)
# ----------------------------------------------------------------------------- #
def compute_edge_with_atr_macd(df: pd.DataFrame, period: int = 14,
                                min_count: int = 30,
                                bb_period: int = 20, bb_std: float = 2.0,
                                mom_period: int = 10,
                                atr_period: int = 14,
                                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    Per-(hour, day_of_week) edge table with ATR and MACD feature columns.
    
    Adds per-bar features (computed at close):
      * rsi           : Wilder RSI(period), in [0, 100]
      * bollinger_pos : position within Bollinger(bb_period, bb_std) bands, [0, 1]
      * momentum      : simple N-bar percentage return, dimensionless
      * atr_pct       : ATR as % of close (volatility measure)
      * macd_hist     : MACD histogram (trend strength)
    
    The aggregated edge table reports the slot-level *mean* of each feature:
      * rsi_at_open
      * bollinger_pos_at_open
      * momentum_at_open
      * atr_pct_at_open
      * macd_hist_at_open
    
    Falls back to compute_edge() (with NaN feature columns) if there isn't
    enough data to compute features on at least `min_count` bars.
    """
    base = compute_edge(df)
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    
    # Compute all features
    rsi = compute_rsi(df["close"], period=period)
    bb = compute_bollinger_position(df["close"], period=bb_period, std_dev=bb_std)
    mom = compute_momentum(df["close"], period=mom_period)
    atr_pct = compute_atr_pct(df, period=atr_period)
    macd_line, signal_line, macd_hist = compute_macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    
    d = df[["close"]].copy()
    d["ret"] = df["close"].pct_change()
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["rsi"] = rsi
    d["bollinger_pos"] = bb
    d["momentum"] = mom
    d["atr_pct"] = atr_pct
    d["macd_hist"] = macd_hist
    
    # Use only bars where ALL features are defined
    d = d.dropna(subset=["rsi", "ret", "bollinger_pos", "momentum", "atr_pct", "macd_hist"])
    if len(d) < min_count:
        # Not enough data — return base edge with NaN feature columns so the
        # caller can detect the fallback.
        out = base.copy()
        out["rsi_at_open"] = np.nan
        out["bollinger_pos_at_open"] = np.nan
        out["momentum_at_open"] = np.nan
        out["atr_pct_at_open"] = np.nan
        out["macd_hist_at_open"] = np.nan
        return out
    
    grp = d.groupby(["hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    
    rsi_by_slot = grp["rsi"].mean().rename("rsi_at_open").reset_index()
    bb_by_slot = grp["bollinger_pos"].mean().rename("bollinger_pos_at_open").reset_index()
    mom_by_slot = grp["momentum"].mean().rename("momentum_at_open").reset_index()
    atr_by_slot = grp["atr_pct"].mean().rename("atr_pct_at_open").reset_index()
    macd_by_slot = grp["macd_hist"].mean().rename("macd_hist_at_open").reset_index()
    
    edge = edge.merge(rsi_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(bb_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(mom_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(atr_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(macd_by_slot, on=["hour", "day_of_week"], how="left")
    
    edge = edge[edge["count"] >= min_count].copy()
    return edge.reset_index(drop=True)


# ----------------------------------------------------------------------------- #
# Bollinger Bands feature (position relative to bands, in [0, 1])
# ----------------------------------------------------------------------------- #
def compute_bollinger_position(series: pd.Series, period: int = 20,
                                std_dev: float = 2.0) -> pd.Series:
    """Return the close's position within the Bollinger Bands, clipped to [0, 1].

    bollinger_pos = (close - lower) / (upper - lower)
      * 0.0 → at or below lower band (oversold)
      * 0.5 → at the moving average
      * 1.0 → at or above upper band (overbought)

    First `period` bars are NaN. If the bands collapse (upper == lower) the
    value is set to 0.5 (neutral) to avoid division by zero.
    """
    s = pd.Series(series).astype(float)
    ma = s.rolling(window=period, min_periods=period).mean()
    sd = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + std_dev * sd
    lower = ma - std_dev * sd
    width = (upper - lower)
    # Mark warmup bars (where std is undefined) as NaN instead of substituting 0.5,
    # so callers can dropna them cleanly without polluting aggregates.
    pos = (s - lower) / width.replace(0.0, np.nan)
    pos = pos.where(ma.notna() & sd.notna(), np.nan)
    return pos.clip(lower=0.0, upper=1.0)


# ----------------------------------------------------------------------------- #
# Momentum feature (simple N-bar percentage return)
# ----------------------------------------------------------------------------- #
def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Simple percentage return over the last `period` bars: s/s.shift(period) - 1.

    Positive → recent uptrend (trend-continuation bullish bias).
    Negative → recent downtrend (trend-continuation bearish bias).
    First `period` bars are NaN.
    """
    s = pd.Series(series).astype(float)
    return s / s.shift(period) - 1.0


# ----------------------------------------------------------------------------- #
# ATR feature (Average True Range) — volatility filter
# ----------------------------------------------------------------------------- #
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).
    
    Returns ATR values in price units (same as close). First `period` bars are NaN.
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder smoothing == EMA with alpha = 1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR as percentage of close price (dimensionless, comparable across assets).
    """
    atr = compute_atr(df, period)
    return atr / df['close']


# ----------------------------------------------------------------------------- #
# MACD feature (Moving Average Convergence Divergence)
# ----------------------------------------------------------------------------- #
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard MACD with signal line and histogram.
    
    Returns (macd_line, signal_line, histogram)
    """
    s = pd.Series(series).astype(float)
    ema_fast = s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ----------------------------------------------------------------------------- #
# Edge with ATR + MACD features (per slot)
# ----------------------------------------------------------------------------- #
def compute_edge_with_atr_macd(df: pd.DataFrame, period: int = 14,
                                min_count: int = 30,
                                bb_period: int = 20, bb_std: float = 2.0,
                                mom_period: int = 10,
                                atr_period: int = 14,
                                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    Per-(hour, day_of_week) edge table with ATR and MACD feature columns.
    
    Adds per-bar features (computed at close):
      * rsi           : Wilder RSI(period), in [0, 100]
      * bollinger_pos : position within Bollinger(bb_period, bb_std) bands, [0, 1]
      * momentum      : simple N-bar percentage return, dimensionless
      * atr_pct       : ATR as % of close (volatility measure)
      * macd_hist     : MACD histogram (trend strength)
    
    The aggregated edge table reports the slot-level *mean* of each feature:
      * rsi_at_open
      * bollinger_pos_at_open
      * momentum_at_open
      * atr_pct_at_open
      * macd_hist_at_open
    
    Falls back to compute_edge() (with NaN feature columns) if there isn't
    enough data to compute features on at least `min_count` bars.
    """
    base = compute_edge(df)
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    
    # Compute all features
    rsi = compute_rsi(df["close"], period=period)
    bb = compute_bollinger_position(df["close"], period=bb_period, std_dev=bb_std)
    mom = compute_momentum(df["close"], period=mom_period)
    atr_pct = compute_atr_pct(df, period=atr_period)
    macd_line, signal_line, macd_hist = compute_macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    
    d = df[["close"]].copy()
    d["ret"] = df["close"].pct_change()
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["rsi"] = rsi
    d["bollinger_pos"] = bb
    d["momentum"] = mom
    d["atr_pct"] = atr_pct
    d["macd_hist"] = macd_hist
    
    # Use only bars where ALL features are defined
    d = d.dropna(subset=["rsi", "ret", "bollinger_pos", "momentum", "atr_pct", "macd_hist"])
    if len(d) < min_count:
        # Not enough data — return base edge with NaN feature columns so the
        # caller can detect the fallback.
        out = base.copy()
        out["rsi_at_open"] = np.nan
        out["bollinger_pos_at_open"] = np.nan
        out["momentum_at_open"] = np.nan
        out["atr_pct_at_open"] = np.nan
        out["macd_hist_at_open"] = np.nan
        return out
    
    grp = d.groupby(["hour", "day_of_week"])
    edge = grp["ret"].agg(
        mean_ret="mean",
        std_ret="std",
        win_rate=lambda s: (s > 0).mean(),
        count="count",
    ).reset_index()
    
    rsi_by_slot = grp["rsi"].mean().rename("rsi_at_open").reset_index()
    bb_by_slot = grp["bollinger_pos"].mean().rename("bollinger_pos_at_open").reset_index()
    mom_by_slot = grp["momentum"].mean().rename("momentum_at_open").reset_index()
    atr_by_slot = grp["atr_pct"].mean().rename("atr_pct_at_open").reset_index()
    macd_by_slot = grp["macd_hist"].mean().rename("macd_hist_at_open").reset_index()
    
    edge = edge.merge(rsi_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(bb_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(mom_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(atr_by_slot, on=["hour", "day_of_week"], how="left")
    edge = edge.merge(macd_by_slot, on=["hour", "day_of_week"], how="left")
    
    edge = edge[edge["count"] >= min_count].copy()
    return edge.reset_index(drop=True)
    """Load ``data/dxy_1h_real.csv`` (US Dollar Index, 1h bars)."""
    if not DXY_1H_CSV.exists():
        raise FileNotFoundError(
            f"DXY file not found at {DXY_1H_CSV}. Run data/fetch_dxy.py first."
        )
    return _load_csv(DXY_1H_CSV)


def merge_dxy_features(df: pd.DataFrame, dxy: pd.DataFrame | None = None
                        ) -> pd.DataFrame:
    """Merge DXY (US Dollar Index) returns into a Gold DataFrame.

    Adds three causal features computed at each Gold bar's timestamp using the
    matching DXY close:

        dxy_close          : aligned DXY close (level, for sanity)
        dxy_return_1h      : DXY 1h percentage return (current / prev - 1)
        dxy_return_4h      : DXY 4h percentage return
        dxy_return_24h     : DXY 24h percentage return

    The merge uses ``merge_asof`` on the UTC timestamp index with ``backward``
    direction (use the most recent DXY bar at or before each Gold bar). This
    handles the case where DXY is missing a few hours that Gold has (e.g.
    weekend partial bars) without dropping Gold bars.

    If DXY is not available, returns the input DataFrame unchanged with NaN
    feature columns so callers can degrade gracefully.
    """
    if dxy is None:
        if not DXY_1H_CSV.exists():
            out = df.copy()
            out["dxy_close"] = np.nan
            out["dxy_return_1h"] = np.nan
            out["dxy_return_4h"] = np.nan
            out["dxy_return_24h"] = np.nan
            return out
        dxy = load_dxy_history()

    dxy = dxy.copy()
    dxy = dxy.rename(columns={"close": "dxy_close"})
    dxy["dxy_return_1h"] = dxy["dxy_close"].pct_change(1)
    dxy["dxy_return_4h"] = dxy["dxy_close"].pct_change(4)
    dxy["dxy_return_24h"] = dxy["dxy_close"].pct_change(24)

    # Both indices are tz-aware UTC; sort for merge_asof
    gold = df.copy().sort_index()
    dxy = dxy[["dxy_close", "dxy_return_1h",
               "dxy_return_4h", "dxy_return_24h"]].sort_index()

    merged = pd.merge_asof(
        left=gold,
        right=dxy,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=pd.Timedelta("2h"),  # DXY is hourly; allow up to 2h slack
    )
    return merged


def load_us10y_history() -> pd.DataFrame:
    """Load ``data/us10y_1h_real.csv`` (10Y Treasury yield, 1h bars, ^TNX %).

    Mirror of load_dxy_history(). Raises FileNotFoundError pointing at the
    fetcher so the caller can degrade or fetch first.
    """
    if not US10Y_1H_CSV.exists():
        raise FileNotFoundError(
            f"US10Y file not found at {US10Y_1H_CSV}. Run data/fetch_us10y.py first."
        )
    return _load_csv(US10Y_1H_CSV)


def merge_us10y_features(df: pd.DataFrame, us10y: pd.DataFrame | None = None
                         ) -> pd.DataFrame:
    """Merge US10Y (10-Year Treasury yield) returns into a Gold DataFrame.

    Adds three causal features computed at each Gold bar's timestamp using the
    matching US10Y close (yield in percent):

        us10y_close         : aligned US10Y level (for sanity)
        us10y_return_1h     : US10Y 1h percentage return (current / prev - 1)
        us10y_return_4h     : US10Y 4h percentage return
        us10y_return_24h    : US10Y 24h percentage return

    The merge uses ``merge_asof`` on the UTC timestamp index with ``backward``
    direction (use the most recent US10Y bar at or before each Gold bar). This
    is leak-safe: no future yield bar influences the current Gold bar's features.

    If US10Y is not available, returns the input DataFrame unchanged with NaN
    feature columns so callers can degrade gracefully.
    """
    if us10y is None:
        if not US10Y_1H_CSV.exists():
            out = df.copy()
            out["us10y_close"] = np.nan
            out["us10y_return_1h"] = np.nan
            out["us10y_return_4h"] = np.nan
            out["us10y_return_24h"] = np.nan
            return out
        us10y = load_us10y_history()

    us10y = us10y.copy()
    us10y = us10y.rename(columns={"close": "us10y_close"})
    us10y["us10y_return_1h"] = us10y["us10y_close"].pct_change(1)
    us10y["us10y_return_4h"] = us10y["us10y_close"].pct_change(4)
    us10y["us10y_return_24h"] = us10y["us10y_close"].pct_change(24)

    # Both indices are tz-aware UTC; sort for merge_asof
    gold = df.copy().sort_index()
    us10y = us10y[["us10y_close", "us10y_return_1h",
                   "us10y_return_4h", "us10y_return_24h"]].sort_index()

    merged = pd.merge_asof(
        left=gold,
        right=us10y,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=pd.Timedelta("2h"),  # US10Y is hourly; allow up to 2h slack
    )
    return merged


# ----------------------------------------------------------------------------- #
# Economic calendar (hardcoded high-impact US events)
# ----------------------------------------------------------------------------- #
def load_economic_calendar() -> pd.DataFrame:
    """Load ``data/economic_calendar.csv`` (FOMC/NFP/CPI/PPI 2024-2026).

    Returns a DataFrame indexed by event date (UTC midnight) with columns:
        event_type : str  (FOMC | NFP | CPI | PPI)
        impact     : str  (HIGH)

    The dates are calendar-day granularity — bar-level proximity is computed
    downstream via compute_news_proximity().
    """
    if not ECON_CAL_CSV.exists():
        raise FileNotFoundError(
            f"Economic calendar not found at {ECON_CAL_CSV}. "
            "Run scripts/build_econ_calendar.py first."
        )
    df = pd.read_csv(ECON_CAL_CSV, parse_dates=["date"])
    # Normalise to UTC midnight so subtract-arithmetic is clean
    df["date"] = df["date"].dt.tz_localize("UTC")
    return df.reset_index(drop=True)


def compute_news_proximity(df: pd.DataFrame, calendar: pd.DataFrame | None = None,
                            window_hours: float = 24.0) -> pd.Series:
    """For each bar in ``df``, return hours-to-nearest-high-impact-event.

    Returns a Series indexed like ``df`` (DatetimeIndex, UTC) whose values are:

      * np.nan  — no calendar provided, or calendar empty (caller falls back)
      * float   — signed hours to the nearest event in ``calendar``:
                    positive → event is in the future
                    negative → event already happened
                    0.0      → bar is within the same hour as an event

    Bar granularity matches ``df``: for a 1h index this resolves to whole
    hours. ``window_hours`` (legacy arg, not used for filtering here) is
    retained for API symmetry with earlier drafts.
    """
    if calendar is None or len(calendar) == 0:
        return pd.Series(np.nan, index=df.index, name="news_proximity_h")

    # Event timestamps at their canonical intraday hour:
    #   FOMC decision  : 14:00 ET  = 18:00 UTC
    #   NFP / CPI / PPI :  8:30 AM ET = 12:30 UTC
    # Calendar dates are midnight UTC; shift to the actual release time.
    times = []
    for _, row in calendar.iterrows():
        d = row["date"]  # tz-aware UTC midnight
        etype = row["event_type"]
        if etype == "FOMC":
            release = d + pd.Timedelta(hours=18)        # 14:00 ET
        else:
            release = d + pd.Timedelta(hours=12, minutes=30)  # 8:30 ET
        times.append(release.to_pydatetime())
    event_times = np.array(
        [np.datetime64(t.replace(tzinfo=None)) for t in times],
        dtype="datetime64[ns]",
    )

    bar_times = df.index.tz_convert("UTC").tz_localize(None).to_numpy()
    # For each bar, find the nearest event time (in hours)
    # O(B * E) where B = bars, E = events. With ~11k bars and ~130 events
    # this is ~1.4M scalar subtractions — trivial.
    out = np.empty(len(bar_times), dtype=float)
    for i, bt in enumerate(bar_times):
        delta_ns = event_times - bt
        nearest_ns = delta_ns[np.abs(delta_ns).argmin()]
        out[i] = nearest_ns / np.timedelta64(1, "h")
    return pd.Series(out, index=df.index, name="news_proximity_h")


def within_news_window(proximity: pd.Series, near_h: float = 4.0,
                       far_h: float = 24.0) -> pd.DataFrame:
    """Bucket a proximity Series (hours to nearest event) into 3 regimes.

    Returns a DataFrame with boolean/int columns:
        near_news  : 1 if |proximity| <= near_h   (within ±4h of an event)
        mid_news   : 1 if near_h < |proximity| <= far_h
        quiet      : 1 otherwise (more than 24h from any event)

    Only HIGH-impact events are in the source calendar, so all three flags
    imply a high-impact event is nearby.
    """
    abs_p = proximity.abs()
    near = (abs_p <= near_h).astype(int)
    mid = ((abs_p > near_h) & (abs_p <= far_h)).astype(int)
    quiet = (abs_p > far_h).astype(int)
    return pd.DataFrame(
        {"near_news": near, "mid_news": mid, "quiet": quiet},
        index=proximity.index,
    )


# ----------------------------------------------------------------------------- #
# RSI-aware probability combiner
# ----------------------------------------------------------------------------- #
def combine_probability_rsi(edge_row: pd.Series, rsi: float = 50.0,
                             lam: float = 0.3, sentiment: float = 0.0,
                             rsi_boost: float = 0.05) -> float:
    """Sharpe-logistic probability, optionally nudged by RSI.

    base = combine_probability(edge_row, lam, sentiment)  — direction "up" prob
    delta = rsi_boost * tanh((50 - rsi) / 20)              — + if oversold, - if overbought
    prob_up = clip(base + delta, 0.01, 0.99)
    """
    base = combine_probability(edge_row, lam=lam, sentiment=sentiment)
    if rsi is None or (isinstance(rsi, float) and np.isnan(rsi)):
        return base
    delta = rsi_boost * float(np.tanh((50.0 - rsi) / 20.0))
    return float(round(max(0.01, min(0.99, base + delta)), 4))


# ----------------------------------------------------------------------------- #
# Full-feature probability combiner (RSI + Bollinger + Momentum)
# ----------------------------------------------------------------------------- #
def combine_probability_full(edge_row: pd.Series,
                              rsi: float = 50.0,
                              bollinger_pos: float = 0.5,
                              momentum: float = 0.0,
                              lam: float = 0.3,
                              sentiment: float = 0.0,
                              rsi_boost: float = 0.05,
                              bb_boost: float = 0.05,
                              mom_boost: float = 0.05) -> float:
    """Sharpe-logistic probability nudged by RSI, Bollinger, and Momentum.

    Each feature contributes a small additive nudge in probability space; the
    weights are intentionally small (default 0.05) to keep the model close to
    the Sharpe-logistic core and avoid overfitting on any single indicator.

    RSI nudge       : +rsi_boost * tanh((50 - rsi) / 20)
                       oversold (low rsi) → boost prob_up
    Bollinger nudge : if bollinger_pos < 0.1 → +bb_boost * (0.1 - pos) / 0.1
                      if bollinger_pos > 0.9 → -bb_boost * (pos - 0.9) / 0.1
                       near lower band → boost prob_up; near upper → boost prob_dn
    Momentum nudge  : +mom_boost * tanh(momentum * 50)
                       positive momentum → trend continuation (boost prob_up)
                       negative momentum → trend continuation (boost prob_dn)
    """
    base = combine_probability(edge_row, lam=lam, sentiment=sentiment)
    delta = 0.0

    # RSI — NaN-safe
    if rsi is not None and not (isinstance(rsi, float) and np.isnan(rsi)):
        delta += rsi_boost * float(np.tanh((50.0 - rsi) / 20.0))

    # Bollinger position — NaN-safe, asymmetric nudge
    if bollinger_pos is not None and not (isinstance(bollinger_pos, float) and np.isnan(bollinger_pos)):
        bp = float(bollinger_pos)
        if bp < 0.1:
            delta += bb_boost * (0.1 - bp) / 0.1        # up to +bb_boost
        elif bp > 0.9:
            delta -= bb_boost * (bp - 0.9) / 0.1        # down to -bb_boost

    # Momentum — NaN-safe, trend-continuation
    if momentum is not None and not (isinstance(momentum, float) and np.isnan(momentum)):
        delta += mom_boost * float(np.tanh(momentum * 50.0))

    return float(round(max(0.01, min(0.99, base + delta)), 4))


# ----------------------------------------------------------------------------- #
# DXY-aware probability combiner (cross-asset causal feature)
# ----------------------------------------------------------------------------- #
def combine_probability_dxy(edge_row: pd.Series,
                             dxy_return_1h: float = 0.0,
                             dxy_return_4h: float = 0.0,
                             dxy_return_24h: float = 0.0,
                             lam: float = 0.3,
                             sentiment: float = 0.0,
                             dxy_boost: float = 0.12,
                             dxy_scale: float = 50.0) -> float:
    """Sharpe-logistic probability nudged by DXY (US Dollar Index) returns.

    DXY moves INVERSELY to Gold: when DXY rises, gold tends to fall, and vice versa.
    We translate DXY returns into a Gold-direction nudge by negating them:

        nudge = dxy_boost * (
            - tanh(dxy_return_1h  * dxy_scale)   # 1h: short-horizon causality
            - tanh(dxy_return_4h  * dxy_scale)   # 4h: medium-horizon
            - tanh(dxy_return_24h * dxy_scale)   # 24h: regime signal
        )

    Each term contributes at most +/- dxy_boost/3 ≈ +/-0.04, so the combined
    nudge is bounded by +/- dxy_boost (≈ +/-0.12 by default). The default
    scale (50) means a 2% DXY move (extreme) produces a near-saturated nudge.

    NaN-safe: any NaN feature contributes 0.
    """
    base = combine_probability(edge_row, lam=lam, sentiment=sentiment)
    delta = 0.0

    def _neg_tanh(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return -float(np.tanh(float(x) * dxy_scale))

    delta += dxy_boost * _neg_tanh(dxy_return_1h)
    delta += dxy_boost * _neg_tanh(dxy_return_4h)
    delta += dxy_boost * _neg_tanh(dxy_return_24h)

    return float(round(max(0.01, min(0.99, base + delta)), 4))


def combine_probability_us10y(edge_row: pd.Series,
                               us10y_return_1h: float = 0.0,
                               us10y_return_4h: float = 0.0,
                               us10y_return_24h: float = 0.0,
                               lam: float = 0.3,
                               sentiment: float = 0.0,
                               us10y_boost: float = 0.10,
                               us10y_scale: float = 10.0) -> float:
    """Sharpe-logistic probability nudged by US10Y (10Y Treasury yield) returns.

    US10Y moves POSITIVELY with the dollar (and thus NEGATIVELY to gold): when
    real yields rise, gold tends to fall, and vice versa. The sign convention
    therefore matches DXY — we negate the yield returns:

        nudge = us10y_boost * (
            - tanh(us10y_return_1h  * us10y_scale)   # 1h: short-horizon causality
            - tanh(us10y_return_4h  * us10y_scale)   # 4h: medium-horizon
            - tanh(us10y_return_24h * us10y_scale)   # 24h: regime signal
        ) / 3

    Each term contributes at most +/- us10y_boost/3, so the combined nudge is
    bounded by +/- us10y_boost (≈ +/-0.10 by default). The default scale (10)
    means a 10% yield move (extreme) produces a near-saturated nudge.

    NaN-safe: any NaN feature contributes 0.
    """
    base = combine_probability(edge_row, lam=lam, sentiment=sentiment)
    delta = 0.0

    def _neg_tanh(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return -float(np.tanh(float(x) * us10y_scale))

    delta += us10y_boost * _neg_tanh(us10y_return_1h)
    delta += us10y_boost * _neg_tanh(us10y_return_4h)
    delta += us10y_boost * _neg_tanh(us10y_return_24h)
    delta = delta / 3.0

    return float(round(max(0.01, min(0.99, base + delta)), 4))


# ----------------------------------------------------------------------------- #
# News-aware probability combiner (economic-calendar proximity)
# ----------------------------------------------------------------------------- #
def combine_probability_news(edge_row: pd.Series,
                              rsi: float = 50.0,
                              bollinger_pos: float = 0.5,
                              momentum: float = 0.0,
                              dxy_return_1h: float = 0.0,
                              dxy_return_4h: float = 0.0,
                              dxy_return_24h: float = 0.0,
                              news_proximity_h: float = np.nan,
                              lam: float = 0.3,
                              sentiment: float = 0.0,
                              rsi_boost: float = 0.05,
                              bb_boost: float = 0.05,
                              mom_boost: float = 0.05,
                              dxy_boost: float = 0.12,
                              dxy_scale: float = 50.0,
                              near_h: float = 4.0,
                              far_h: float = 24.0,
                              near_pullback: float = 0.20,
                              mid_pullback: float = 0.05) -> float:
    """Sharpe-logistic probability with full feature stack + news proximity.

    Builds on combine_probability_full() and adds two news-related effects:

    1. **Confidence pullback near events.** Within ±``near_h`` hours of a
       high-impact release (FOMC/NFP/CPI/PPI), the directional probability is
       pulled TOWARD 0.5 by ``near_pullback`` (default 0.20). The reasoning is
       that within the event window, gold's direction is essentially random
       on the bar (volatility is huge, but its SIGN is unpredictable), so
       high-confidence probabilities are untrustworthy and should be
       discounted.

    2. **Smaller pullback 4–24h out.** In the anticipation window, we apply
       ``mid_pullback`` (default 0.05) — a much milder effect to avoid
       over-trading just before the release.

    The net result: probabilities emitted near news are pushed toward neutral
    (0.5), which, when thresholded, prevents the model from placing trades
    in the unpredictable event window. This is a *filter*, not a *predictor* —
    we are reducing false-confidence rather than trying to call direction.

    Notes
    -----
    All sub-features (RSI, Bollinger, momentum, DXY) are passed to
    combine_probability_full(); news is applied as a post-hoc symmetric
    pullback toward 0.5.

    Pullback formula
    ----------------
        pull = near_pullback if |proximity_h| <= near_h
               elif |proximity_h| <= far_h: mid_pullback
               else: 0
        prob_up_pulled = 0.5 + (prob_up - 0.5) * (1 - pull)
        prob_dn_pulled = 1 - prob_up_pulled   (symmetric)
    """
    # Build base probability with RSI/Bollinger/Momentum stack
    base = combine_probability_full(
        edge_row, rsi=rsi, bollinger_pos=bollinger_pos, momentum=momentum,
        lam=lam, sentiment=sentiment,
        rsi_boost=rsi_boost, bb_boost=bb_boost, mom_boost=mom_boost,
    )

    # Layer the DXY cross-asset nudge
    delta_dxy = 0.0

    def _neg_tanh(x, scale=dxy_scale):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return -float(np.tanh(float(x) * scale))

    delta_dxy += dxy_boost * _neg_tanh(dxy_return_1h)
    delta_dxy += dxy_boost * _neg_tanh(dxy_return_4h)
    delta_dxy += dxy_boost * _neg_tanh(dxy_return_24h)
    prob_up = float(max(0.01, min(0.99, base + delta_dxy)))

    # News proximity pullback toward 0.5
    if (news_proximity_h is None
            or (isinstance(news_proximity_h, float) and np.isnan(news_proximity_h))):
        pull = 0.0
    else:
        abs_h = abs(float(news_proximity_h))
        if abs_h <= near_h:
            pull = near_pullback
        elif abs_h <= far_h:
            pull = mid_pullback
        else:
            pull = 0.0

    if pull > 0:
        prob_up = 0.5 + (prob_up - 0.5) * (1.0 - pull)
        prob_up = float(max(0.01, min(0.99, prob_up)))

    return float(round(prob_up, 4))


# Back-compat alias used by live_edge / cerebrum_live.
def compute_hourly_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy alias kept for callers that still expect per-hour edge."""
    if "ret" in df.columns:
        df = df.drop(columns=["ret"])
    ret = df["close"].pct_change().rename("ret")
    d = df.join(ret)
    edge = (
        d.groupby(d.index.hour)["ret"]
        .agg(mean_ret="mean", std_ret="std",
             win_rate=lambda s: (s > 0).mean(), count="count")
        .reset_index()
        .rename(columns={"index": "hour"})
    )
    return edge


# ----------------------------------------------------------------------------- #
# Sentiment (placeholder)
# ----------------------------------------------------------------------------- #
def load_sentiment_score(for_hour: int) -> float:
    """Return a sentiment delta in [-1, +1] for the given UTC hour.

    In this version we don't have a news API. Real production should swap this
    for an economic-calendar lookup (FOMC, ECB, NFP, etc.).
    """
    return 0.0


# ----------------------------------------------------------------------------- #
# Probability combination (Sharpe-style logistic)
# ----------------------------------------------------------------------------- #
def combine_probability(edge_row: pd.Series, lam: float = 0.3,
                         sentiment: float = 0.0) -> float:
    """Convert an edge row into a directional probability.

    WHY THIS LOGISTIC (vs the old 1/(1+exp(-10*mean_ret))):
      The old formula took a single scaler (mean_ret) that was almost always
      near zero for FX (a few bps per 30-min). With a steep logistic the
      output saturated at exactly 0.5 and we could never cross any
      non-trivial threshold -- hence the "coin flip" symptom.

      New formula uses the Sharpe-like ratio `mean_ret / std_ret`. The
      denominator shrinks toward zero when volatility is low (the moments
      where a small edge is meaningful) and grows when volatility is high
      (the moments where the same mean_ret is mostly noise). This produces
      a wide spread of probabilities (typically 0.30 - 0.70) and is the
      standard way academics present t-statistic-based trading signals.

      `lam * sentiment` is a small additive offset in probability space.
    """
    mean_ret = float(edge_row["mean_ret"])
    std_ret = float(edge_row["std_ret"]) if float(edge_row["std_ret"]) > 1e-9 else 1e-9
    sharpe = mean_ret / std_ret              # for FX 30-min, typically in [-0.5, +0.5]
    raw = sharpe + lam * sentiment
    # logistic steepness: 8.0 maps a Sharpe of 0.5 → prob ≈ 0.98
    prob = 1.0 / (1.0 + np.exp(-8.0 * raw))
    return float(round(prob, 4))


# ----------------------------------------------------------------------------- #
# Window finders (4-hour bars: 6 slots per day, aligned to UTC boundaries)
# ----------------------------------------------------------------------------- #
def find_next_window(edge: pd.DataFrame, threshold: float = 0.65,
                     lam: float = 0.3, future_only: bool = True) -> dict:
    """Scan the next 6 four-hour slots from now and return the first one whose
    probability is ≥ threshold.

    Returns a dict with window_start_utc, hour, day_of_week, probability,
    edge_mean_ret, sentiment, side.  If none found returns {'none': None}.
    """
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    # 4h boundaries: 00, 04, 08, 12, 16, 20 UTC
    boundaries = [0, 4, 8, 12, 16, 20]
    current_block = (now.hour // 4) * 4
    if now.hour % 4 == 0 and now.minute == 0 and now.second == 0:
        next_boundary_hour = current_block
    else:
        idx = boundaries.index(current_block)
        next_boundary_hour = boundaries[(idx + 1) % 6] if idx < 5 else 0
        if idx == 5:  # currently in 20-00 block
            next_boundary_hour = 0
        # if not exactly on a boundary, next is the next 4h boundary
        if now.hour % 4 != 0 or now.minute > 0 or now.second > 0:
            next_block_idx = (now.hour // 4) + 1
            if next_block_idx >= 6:
                next_boundary_hour = 0
            else:
                next_boundary_hour = boundaries[next_block_idx]
    # Build the next 6 4h boundaries starting from next_boundary
    now_date = now.date()
    seen = set()
    for offset in range(6):
        block_idx = (boundaries.index(next_boundary_hour) + offset) % 6
        day_offset = (boundaries.index(next_boundary_hour) + offset) // 6
        slot_hour = boundaries[block_idx]
        slot_date = now_date + datetime.timedelta(days=day_offset)
        slot_dt = datetime.datetime.combine(
            slot_date, datetime.time(hour=slot_hour, minute=0, tzinfo=datetime.timezone.utc)
        )
        key = (slot_hour, slot_dt.weekday())
        if key in seen:
            continue
        seen.add(key)
        match = edge[(edge["hour"] == slot_hour) & (edge["day_of_week"] == slot_dt.weekday())]
        if match.empty:
            continue
        sent = load_sentiment_score(slot_hour)
        prob = combine_probability(match.iloc[0], lam=lam, sentiment=sent)
        if prob >= threshold:
            return {
                "window_start_utc": slot_dt.isoformat(),
                "hour": slot_hour,
                "day_of_week": slot_dt.weekday(),
                "probability": prob,
                "edge_mean_ret": float(match.iloc[0]["mean_ret"]),
                "edge_std_ret": float(match.iloc[0]["std_ret"]),
                "sentiment": sent,
                "side": "BUY",
                "n_samples": int(match.iloc[0]["count"]),
            }
    return {"none": None}


def find_best_window_in_next_24h(edge: pd.DataFrame, threshold: float = 0.50,
                                  lam: float = 0.3, top_n: int = 5,
                                  future_only: bool = True,
                                  use_mtf: bool = True,
                                  weekly_df: pd.DataFrame | None = None,
                                  atr_min_pct: float = 0.003) -> list[dict]:
    """
    Return up to `top_n` future 4-hour slots ranked by descending prob above
    `threshold`.  The first item is the BEST recommended slot to act on.

    For 4h bars there are 6 slots per day (00, 04, 08, 12, 16, 20 UTC), so the
    next 24h contains 6 unique (hour, day_of_week) slots.
    
    NEW: If `use_mtf=True` and `weekly_df` provided, applies multi-timeframe filter:
      - Weekly trend alignment (boost aligned, penalize counter-trend)
      - ATR volatility filter (skip if ATR% < atr_min_pct)
      - MACD histogram confirmation (align with signal direction)
    """
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    boundaries = [0, 4, 8, 12, 16, 20]
    # Find the next 4h boundary strictly after `now`
    cur_block = now.hour // 4
    next_block = cur_block + 1 if (now.hour % 4 != 0 or now.minute > 0 or now.second > 0) else cur_block
    if next_block >= 6:
        next_block = 0
        day_offset_start = 1
    else:
        day_offset_start = 0
    now_date = now.date()
    scored = []
    seen = set()
    
    # Load weekly data if not provided but MTF is requested
    if use_mtf and weekly_df is None:
        try:
            weekly_df = load_weekly_history()
        except Exception:
            weekly_df = None
    
    # Pre-compute current features for MTF
    atr_pct_now = None
    macd_hist_now = None
    if use_mtf:
        # We'd need the full dataframe for this - for now use the edge's mean ATR
        pass
    
    for offset in range(6):
        block_idx = (next_block + offset) % 6
        day_offset = day_offset_start + (next_block + offset) // 6
        slot_hour = boundaries[block_idx]
        slot_date = now_date + datetime.timedelta(days=day_offset)
        slot_dt = datetime.datetime.combine(
            slot_date, datetime.time(hour=slot_hour, minute=0, tzinfo=datetime.timezone.utc)
        )
        key = (slot_hour, slot_dt.weekday())
        if key in seen:
            continue
        seen.add(key)
        match = edge[(edge["hour"] == slot_hour) & (edge["day_of_week"] == slot_dt.weekday())]
        if match.empty:
            continue
        sent = load_sentiment_score(slot_hour)
        
        row = match.iloc[0]
        
        # Base probability
        prob_up = combine_probability(row, lam=lam, sentiment=sent)
        
        # Apply multi-timeframe filter if enabled
        if use_mtf and weekly_df is not None:
            # Weekly trend
            weekly_trend = get_weekly_trend_at(weekly_df, pd.Timestamp(slot_dt))
            
            # Get features from edge row
            daily_rsi = row.get('rsi_at_open', 50.0)
            daily_bb = row.get('bollinger_pos_at_open', 0.5)
            daily_mom = row.get('momentum_at_open', 0.0)
            daily_atr = row.get('atr_pct_at_open', 0.0)
            daily_macd = row.get('macd_hist_at_open', 0.0)
            
            # ATR volatility filter - skip if too low
            if daily_atr is not None and not np.isnan(daily_atr):
                if daily_atr < atr_min_pct:
                    continue  # Skip this slot - insufficient volatility
            
            # MACD confirmation - only take if MACD aligns with signal
            if daily_macd is not None and not np.isnan(daily_macd):
                prob_up_macd = prob_up
                prob_dn_macd = 1.0 - prob_up_macd
                side = "BUY" if prob_up_macd >= prob_dn_macd else "SELL"
                if (side == "BUY" and daily_macd < 0) or (side == "SELL" and daily_macd > 0):
                    # MACD opposes signal - reduce probability
                    prob_up = prob_up * 0.8  # 20% penalty
            
            # Weekly trend alignment using the new combiner
            prob_up = combine_probability_mtf(
                row,
                daily_rsi=daily_rsi,
                daily_bollinger_pos=daily_bb,
                daily_momentum=daily_mom,
                weekly_trend=weekly_trend,
                lam=lam,
                sentiment=sent,
            )
        
        prob_dn = 1.0 - prob_up
        scored.append({
            "window_start_utc": slot_dt.isoformat(),
            "hour": slot_hour,
            "day_of_week": slot_dt.weekday(),
            "probability_up": prob_up,
            "probability_dn": prob_dn,
            "edge_mean_ret": float(row["mean_ret"]),
            "edge_std_ret": float(row["std_ret"]),
            "sentiment": sent,
            "n_samples": int(row["count"]),
        })
    for s in scored:
        s["probability"] = max(s["probability_up"], s["probability_dn"])
        s["side"] = "BUY" if s["probability_up"] >= s["probability_dn"] else "SELL"
    scored.sort(key=lambda s: s["probability"], reverse=True)
    above = [s for s in scored if s["probability"] >= threshold]
    return above[:top_n]


# ----------------------------------------------------------------------------- #
# Evidence helper (for 4h bars)
# ----------------------------------------------------------------------------- #
def get_evidence(df: pd.DataFrame, target_hour: int, target_dow: Optional[int] = None,
                  n_similar: int = 3, lookback_days: int = 90) -> dict:
    """Pull historical evidence for a 4-hour slot.

    Returns a dict with:
        target_hit_rate   - fraction of past slots where price went UP
        target_avg_move   - mean pct move in that slot
        target_count      - sample size
        best_similar      - list of up to n_similar slots (hour, day_of_week) most
                            similar to (target_hour, target_dow) ranked by hit rate.
    """
    slot_mask = (df.index.hour == target_hour)
    if target_dow is not None:
        slot_mask &= (df.index.dayofweek == target_dow)
    sub = df[slot_mask].copy()
    if len(sub) < 2:
        return {"target_hit_rate": None, "target_avg_move": None,
                "target_count": len(sub), "best_similar": []}
    moves = sub["close"].pct_change().dropna()
    if len(moves) < 2:
        return {"target_hit_rate": None, "target_avg_move": None,
                "target_count": len(sub), "best_similar": []}
    hit_rate = float((moves > 0).mean())
    avg_move = float(moves.mean())

    # most-similar slots = same day_of_week, neighbouring hours
    candidates = []
    for dh in range(-3, 4):
        if dh == 0:
            continue
        h = (target_hour + dh) % 24
        mask2 = (df.index.hour == h)
        if target_dow is not None:
            mask2 &= (df.index.dayofweek == target_dow)
        sub2 = df[mask2]
        if len(sub2) < 5:
            continue
        m2 = sub2["close"].pct_change().dropna()
        if len(m2) < 2:
            continue
        candidates.append({
            "hour": h,
            "day_of_week": target_dow,
            "hit_rate": float((m2 > 0).mean()),
            "avg_move": float(m2.mean()),
            "count": int(len(m2)),
        })
    candidates.sort(key=lambda c: c["hit_rate"], reverse=True)
    return {
        "target_hit_rate": hit_rate,
        "target_avg_move": avg_move,
        "target_count": int(len(sub)),
        "best_similar": candidates[:n_similar],
    }


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main():
    df = load_history()
    edge = compute_edge(df)
    print(f"[pipeline] {len(df)} bars loaded "
          f"({df.index.min().date()} → {df.index.max().date()})")
    print(f"[pipeline] {len(edge)} unique (hour, day_of_week) slots with ≥30 samples")

    best = find_best_window_in_next_24h(edge, threshold=0.50, top_n=5)
    if not best:
        print("[pipeline] no slot above 0.50 in next 24h")
    else:
        print("[pipeline] best 5 slots (probability descending):")
        for s in best:
            when = pd.Timestamp(s["window_start_utc"])
            print(f"  {when.strftime('%Y-%m-%d %H:%M UTC')}  "
                  f"side={s['side']}  prob={s['probability']:.3f}  "
                  f"edge={s['edge_mean_ret']:+.4%}  n={s['n_samples']}")


if __name__ == "__main__":
    main()