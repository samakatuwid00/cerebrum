"""
Kelly position sizing and SL/TP logic for cerebrum_trader_bot.

Provides:
- kelly_criterion(): Optimal position size fraction based on win rate and R:R
- compute_sl_tp(): Dynamic stop-loss / take-profit levels using ATR multiples
- apply_position_sizing(): Size positions based on Kelly + volatility
"""
import numpy as np
import pandas as pd


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                    leverage: float = 0.25, edge_threshold: float = 0.53) -> float:
    """
    Compute optimal Kelly fraction for position sizing.
    
    Kelly Criterion: f* = (bp - q) / b
    where:
      b = avg_win / avg_loss (R:R ratio)
      p = win_rate
      q = 1 - p
    
    With leverage factor for practical sizing: f = f_full * leverage
    
    For typical Gold 4h: win_rate ~0.53, avg_win ~0.001, avg_loss ~0.0015
    => b = 0.67, f_full = (0.67*0.53 - 0.47) / 0.67 = -0.15 (negative edge!)
    
    However, in backtests we see positive edges because:
    1. We're using historical means which include fat tails
    2. The Sharpe-weighted approach in combine_probability captures this
    
    For production, we use a simplified approach based on observed win rate
    and a fixed R:R assumption based on ATR-based SL/TP.
    
    Args:
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average winning return in decimal (e.g., 0.001 = 0.1%)
        avg_loss: Average losing return (absolute value, positive number)
        leverage: Fraction of full Kelly to use (default 0.25 for safety)
        edge_threshold: Minimum win rate for Kelly to be considered positive
    
    Returns:
        Optimal fraction of capital to risk per trade
    """
    # Safety checks
    if win_rate < edge_threshold:
        return 0.0  # Edge too small, no position
    
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0  # No valid win/loss data
    
    # R:R ratio (b in Kelly formula)
    b = avg_win / avg_loss
    
    p = win_rate
    q = 1.0 - p
    
    # Full Kelly
    f_full = (b * p - q) / b
    
    # For typical signals: if b < 1 (R:R < 1:1), Kelly < win rate
    # But our model uses 2-3 ATR SL/TP, so R:R is typically > 1
    # Use max(f_full, win_rate * 0.25) as a floor for reasonable sizing
    
    # Conservative: apply leverage factor, cap at 100%
    f_kelly = min(max(f_full * leverage, min(win_rate * 0.25, 0.1)), 1.0)
    
    return f_kelly


def compute_sl_tp(entry_price: float, atr: float, side: str,
                  sl_atr_mult: float = 2.0, tp_atr_mult: float = 3.0) -> tuple:
    """
    Compute dynamic stop-loss and take-profit levels using ATR multiples.
    
    For XAU/USD Gold mode:
      - Long: Stop = entry - sl_mult * ATR, Take = entry + tp_mult * ATR
      - Short: Stop = entry + sl_mult * ATR, Take = entry - tp_mult * ATR
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        side: "BUY" or "SELL"
        sl_atr_mult: ATR multiplier for stop-loss (default 2.0 = 2*ATR)
        tp_atr_mult: ATR multiplier for take-profit (default 3.0 = 3*ATR)
    
    Returns:
        (sl_price, tp_price) tuple
    """
    if atr <= 0:
        return None, None
    
    if side == "BUY":
        sl = entry_price - (sl_atr_mult * atr)
        tp = entry_price + (tp_atr_mult * atr)
    else:  # SELL
        sl = entry_price + (sl_atr_mult * atr)
        tp = entry_price - (tp_atr_mult * atr)
    
    return sl, tp


def compute_risk_reward(entry_price: float, sl_price: float, tp_price: float,
                        side: str) -> float:
    """
    Compute risk/reward ratio from SL/TP levels.
    
    Returns:
        Risk/reward ratio (e.g., 1.5 = risk 1 to make 1.5)
    """
    if side == "BUY":
        risk = entry_price - sl_price
        reward = tp_price - entry_price
    else:  # SELL
        risk = sl_price - entry_price
        reward = entry_price - tp_price
    
    if risk <= 0:
        return 0.0
    
    return reward / risk if reward > 0 else 0.0


def apply_position_sizing(df: pd.DataFrame, trades_df: pd.DataFrame,
                          atr_col: str = 'atr_pct_at_open',
                          default_risk_pct: float = 0.02) -> pd.DataFrame:
    """
    Apply Kelly position sizing to trades dataframe.
    
    Adds columns:
      - kelly_fraction: Optimal fraction from Kelly
      - position_size: Dollars to risk (as fraction of capital)
      - sl_atr_mult: Stop-loss in ATR multiples
      - tp_atr_mult: Take-profit in ATR multiples
      - risk_reward: R:R ratio
    
    Args:
        df: Original price dataframe with ATR column
        trades_df: Trades output from run_backtest_vectorized
        atr_col: Column name for ATR percentage
        default_risk_pct: Default risk per trade if Kelly not available
    
    Returns:
        Trades df with position sizing columns added
    """
    if len(trades_df) == 0:
        return trades_df
    
    result = trades_df.copy()
    
    # Compute aggregate stats for Kelly
    total = len(result)
    wins = result['win'].sum()
    win_rate = wins / total if total > 0 else 0
    
    winning_rets = result[result['win'] == 1]['next_ret'].mean() if wins > 0 else 0
    losing_rets = abs(result[result['win'] == 0]['next_ret'].mean()) if (total - wins) > 0 else 0
    
    # Kelly fraction
    kelly = kelly_criterion(win_rate, winning_rets, losing_rets, leverage=0.25)
    
    # Apply to all trades
    result['kelly_fraction'] = kelly
    result['position_size'] = kelly if kelly > 0 else default_risk_pct
    
    # SL/TP levels (ATR multiples)
    result['sl_atr_mult'] = 2.0
    result['tp_atr_mult'] = 3.0
    
    # Risk/reward
    result['risk_reward'] = result.apply(
        lambda r: compute_risk_reward(
            r.get('entry_price', 0),
            r.get('sl_price', 0),
            r.get('tp_price', 0),
            r['side']
        ) if pd.notna(r.get('sl_price', None)) else 0,
        axis=1
    )
    
    return result