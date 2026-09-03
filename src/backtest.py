"""Vectorized backtest engine for cerebrum_trader_bot.

Compute edge ONCE from full history, then simulate trades by looking up
the signal for each bar. This is O(n) not O(n²).
"""
import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

from src.pipeline import (
    load_history,
    compute_edge,
    combine_probability,
    load_sentiment_score,
    load_weekly_history,
    compute_weekly_bars,
    compute_weekly_trend,
    get_weekly_trend_at,
    combine_probability_mtf,
    compute_edge_with_atr_macd,
    compute_atr_pct,
    compute_macd,
)


def run_backtest_vectorized(threshold: float = 0.50, lam: float = 0.3,
                            min_samples: int = 50, warmup_bars: int = 1000,
                            filename: str = None, use_mtf: bool = False,
                            atr_min_pct: float = 0.003):
    """
    Vectorized walk-forward backtest.
    
    1. Compute edge from FULL history (in-sample for the whole period)
    2. For each bar after warmup: look up signal based on (hour, day_of_week)
    3. If signal >= threshold, record trade and outcome
    
    This is the standard vectorized backtest - much faster than re-computing
    edge for every single bar. Note: this has slight lookahead bias because
    the edge uses the full 2-year history, but for a purely hour-of-day
    model this is minimal bias (the hour-of-day pattern is stable).
    
    NEW: If use_mtf=True, uses multi-timeframe filter for Gold mode:
      - Weekly trend alignment
      - ATR volatility filter
      - MACD trend confirmation
    """
    df = load_history(filename)
    print(f"[backtest] Loaded {len(df)} bars ({df.index.min()} -> {df.index.max()})")
    
    # Compute edge with ATR/MACD features if using MTF
    if use_mtf:
        edge = compute_edge_with_atr_macd(df)
        print(f"[backtest] Edge with ATR/MACD: {len(edge)} unique (hour, day_of_week) slots with >= {min_samples} samples")
        
        # Load weekly data for MTF
        weekly_df = load_weekly_history()
        weekly_trend = compute_weekly_trend(weekly_df)
        print(f"[backtest] Weekly trend computed: {len(weekly_trend)} weekly bars")
    else:
        # Standard edge computation
        edge = compute_edge(df)
        print(f"[backtest] Edge computed: {len(edge)} unique (hour, day_of_week) slots with >= {min_samples} samples")
        weekly_df = None
        weekly_trend = None
    
    # Precompute probability for each slot in edge table
    if use_mtf:
        # Use multi-timeframe probability
        edge["prob_up"] = edge.apply(
            lambda row: combine_probability_mtf(
                row,
                daily_rsi=row.get('rsi_at_open', 50.0),
                daily_bollinger_pos=row.get('bollinger_pos_at_open', 0.5),
                daily_momentum=row.get('momentum_at_open', 0.0),
                weekly_trend=0,  # Will be overridden per-bar
                lam=lam,
                sentiment=load_sentiment_score(row["hour"]),
            ),
            axis=1
        )
    else:
        edge["prob_up"] = edge.apply(
            lambda row: combine_probability(row, lam=lam, sentiment=load_sentiment_score(row["hour"])),
            axis=1
        )
    edge["prob_dn"] = 1.0 - edge["prob_up"]
    edge["side"] = edge.apply(
        lambda r: "BUY" if r["prob_up"] >= r["prob_dn"] else "SELL", axis=1
    )
    edge["prob"] = edge[["prob_up", "prob_dn"]].max(axis=1)
    
    # Precompute returns + LIVE (not slot-average) ATR/MACD for MTF veto
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    # Live indicators on actual bar (not seasonal mean) — for correct veto
    if use_mtf:
        live_atr = compute_atr_pct(df)
        _, _, live_hist = compute_macd(df["close"])
        df["_live_atr"] = live_atr
        df["_live_macd_hist"] = live_hist
    
    # Merge edge probabilities onto each bar (lookup by hour, day_of_week)
    # Preserve the DatetimeIndex by using left_index/right_index merge
    if use_mtf:
        edge_lookup = edge[["hour", "day_of_week", "prob_up", "prob_dn", "prob", "side", "mean_ret", "std_ret", "count", 
                         "rsi_at_open", "bollinger_pos_at_open", "momentum_at_open", "atr_pct_at_open", "macd_hist_at_open"]].copy()
    else:
        edge_lookup = edge[["hour", "day_of_week", "prob_up", "prob_dn", "prob", "side", "mean_ret", "std_ret", "count"]].copy()
    edge_lookup = edge_lookup.set_index(["hour", "day_of_week"])
    
    # Use join on MultiIndex - df already has DatetimeIndex, add hour/day_of_week as index levels
    df_join = df.set_index(["hour", "day_of_week"], append=True)
    df_merged = df_join.join(edge_lookup, how="left")
    df = df_merged.reset_index(level=["hour", "day_of_week"])
    df = df.sort_index()
    
    # Filter: after warmup, has signal, prob >= threshold
    # For MTF, also filter by LIVE ATR/MACD (not slot-average) + weekly handled per-bar
    if use_mtf:
        mask = (
            (df.index >= df.index[warmup_bars]) &
            (df["prob"].notna()) &
            (df["prob"] >= threshold) &
            (df["ret"].notna()) &
            (df["_live_atr"].notna()) &
            (df["_live_atr"] >= atr_min_pct) &  # LIVE volatility filter (fix: was slot-average)
            (df["_live_macd_hist"].notna())
        )
    else:
        mask = (
            (df.index >= df.index[warmup_bars]) &
            (df["prob"].notna()) &
            (df["prob"] >= threshold) &
            (df["ret"].notna())
        )
    trade_bars = df.loc[mask].copy()  # Use .loc to preserve index
    
    if len(trade_bars) == 0:
        print(f"[backtest] No trades at threshold={threshold}")
        return pd.DataFrame(), {}
    
    # For MTF, we need to apply weekly trend per-bar and LIVE MACD veto
    if use_mtf:
        # Apply weekly trend and LIVE MACD veto per trade bar
        probs_adjusted = []
        sides_adjusted = []
        veto_masks = []
        for idx, row in trade_bars.iterrows():
            ts = idx
            weekly_trend_val = get_weekly_trend_at(weekly_df, ts)
            
            # Recompute probability with actual weekly trend (base derived from slot, nudged by weekly)
            prob_up_mtf = combine_probability_mtf(
                row,
                daily_rsi=row.get('rsi_at_open', 50.0),
                daily_bollinger_pos=row.get('bollinger_pos_at_open', 0.5),
                daily_momentum=row.get('momentum_at_open', 0.0),
                weekly_trend=weekly_trend_val,
                lam=lam,
                sentiment=load_sentiment_score(row["hour"]),
            )
            
            # LIVE MACD veto — mirrors meta_live logic (current-bar hist, not slot mean)
            macd_hist_live = row.get('_live_macd_hist', 0.0)
            atr_live = row.get('_live_atr', 0.0)
            # ATR already filtered, but keep for diagnostics
            prob_dn_mtf = 1.0 - prob_up_mtf
            side = "BUY" if prob_up_mtf >= prob_dn_mtf else "SELL"
            # Veto if clearly opposing (outside dust)
            veto = False
            if side == "BUY" and macd_hist_live is not None and macd_hist_live < -0.1:
                veto = True
            elif side == "SELL" and macd_hist_live is not None and macd_hist_live > 0.1:
                veto = True
            if veto:
                veto_masks.append(True)
                # mark as vetoed, will be dropped below
                probs_adjusted.append(float('nan'))
                sides_adjusted.append(side)
                continue
            # Dust opposite -> 20% shave (mirrors meta_live soft penalty)
            if (side == "BUY" and macd_hist_live is not None and -0.1 <= macd_hist_live < 0) or \
               (side == "SELL" and macd_hist_live is not None and 0 < macd_hist_live <= 0.1):
                prob = prob_up_mtf if side == "BUY" else prob_dn_mtf
                prob = prob * 0.8
                prob_up_mtf = prob if side == "BUY" else 1 - prob
            
            probs_adjusted.append(max(prob_up_mtf, 1.0 - prob_up_mtf))
            sides_adjusted.append("BUY" if prob_up_mtf >= 0.5 else "SELL")
        
        trade_bars["prob"] = probs_adjusted
        trade_bars["side"] = sides_adjusted
        # Drop vetoed
        trade_bars = trade_bars[trade_bars["prob"].notna()].copy()
        # Re-filter by threshold after MTF adjustment
        trade_bars = trade_bars[trade_bars["prob"] >= threshold].copy()
        
        if len(trade_bars) == 0:
            print(f"[backtest] No trades after MTF filtering at threshold={threshold}")
            return pd.DataFrame(), {}
    
    # Actual outcome: next bar's return
    trade_bars["next_ret"] = trade_bars["ret"].shift(-1)
    trade_bars = trade_bars[trade_bars["next_ret"].notna()].copy()
    
    # Win logic
    trade_bars["win"] = np.where(
        trade_bars["side"] == "BUY",
        trade_bars["next_ret"] > 0,
        trade_bars["next_ret"] < 0
    ).astype(int)
    
    trades_df = trade_bars[["prob", "side", "mean_ret", "std_ret", "count", "next_ret", "win"]].copy()
    # trade_bars already has DatetimeIndex, use it directly
    trades_df.index = trade_bars.index
    trades_df = trades_df.reset_index().rename(columns={"datetime": "timestamp"})
    # timestamp is a DatetimeIndex - convert to isoformat string
    trades_df["timestamp"] = trades_df["timestamp"].apply(lambda x: x.isoformat())
    
    # Summary stats
    total = len(trades_df)
    wins = trades_df["win"].sum()
    hit_rate = wins / total
    avg_ret = trades_df["next_ret"].mean()
    trades_df["pnl_bps"] = trades_df["next_ret"] * 10000
    total_pnl_bps = trades_df["pnl_bps"].sum()
    sharpe = trades_df["next_ret"].mean() / trades_df["next_ret"].std() * np.sqrt(252 * 48) if trades_df["next_ret"].std() > 0 else 0
    
    # Max drawdown
    cumulative = trades_df["pnl_bps"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    # Best/worst hour
    trades_df["hour"] = pd.to_datetime(trades_df["timestamp"]).dt.hour
    by_hour = trades_df.groupby("hour").agg(
        trades=("win", "count"),
        hit_rate=("win", "mean"),
        avg_ret=("next_ret", "mean")
    ).sort_values("hit_rate", ascending=False)
    
    summary = {
        "threshold": threshold,
        "total_trades": int(total),
        "wins": int(wins),
        "losses": int(total - wins),
        "hit_rate": float(hit_rate),
        "avg_return_bps": float(avg_ret * 10000),
        "total_pnl_bps": float(total_pnl_bps),
        "sharpe_annualized": float(sharpe),
        "max_drawdown_bps": float(max_dd),
        "best_hour": int(by_hour.index[0]) if len(by_hour) > 0 else None,
        "worst_hour": int(by_hour.index[-1]) if len(by_hour) > 0 else None,
        "by_hour": by_hour.to_dict("index"),
    }
    
    return trades_df, summary


def run_backtest_walkforward(threshold: float = 0.50, lam: float = 0.3,
                              min_samples: int = 50, warmup_bars: int = 1000,
                              filename: str = None, use_mtf: bool = False,
                              atr_min_pct: float = 0.003):
    """
    TRUE walk-forward backtest - recomputes edge at each step from prior data only.
    Slower but zero lookahead bias. Uses rolling window for speed.
    
    NEW: If use_mtf=True, applies weekly trend alignment, ATR filter, and MACD confirmation.
    """
    df = load_history(filename)
    print(f"[backtest] Loaded {len(df)} bars ({df.index.min()} -> {df.index.max()})")
    
    # Precompute returns + LIVE (not slot-average) ATR/MACD for correct walk-forward
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    if use_mtf:
        df["_live_atr"] = compute_atr_pct(df)
        _, _, _live_hist_wf = compute_macd(df["close"])
        df["_live_macd_hist"] = _live_hist_wf
    
    trades = []
    
    # Use a rolling window of last 10000 bars to compute edge (not full history)
    # This simulates the live model that uses ~12h of live data + synthetic baseline
    window_size = 10000
    
    # For MTF, we need weekly data
    if use_mtf:
        weekly_df = load_weekly_history()
        print(f"[backtest] Using MTF: weekly trend + ATR {atr_min_pct*100:.2f}% filter + LIVE MACD confirmation")
    else:
        weekly_df = None
    
    for i in range(warmup_bars, len(df) - 1):
        # Use last `window_size` bars up to i
        start_idx = max(0, i - window_size)
        hist = df.iloc[start_idx:i].copy()
        
        if len(hist) < min_samples * 48:  # need enough data for 48 slots
            continue
        
        # Compute edge with appropriate features
        if use_mtf:
            hist_edge = compute_edge_with_atr_macd(hist)
            edge = hist_edge if len(hist_edge) > 0 else compute_edge(hist)
            
            # Get weekly trend as of this timestamp
            ts = df.index[i]
            weekly_trend_val = get_weekly_trend_at(weekly_df, ts)
        else:
            edge = compute_edge(hist)
            weekly_trend_val = None
        
        if len(edge) == 0:
            continue
        
        ts = df.index[i]
        target_hour = ts.hour
        target_dow = ts.dayofweek

        match = edge[(edge["hour"] == target_hour) & (edge["day_of_week"] == target_dow)]
        if match.empty or match.iloc[0]["count"] < min_samples:
            continue
        
        row = match.iloc[0]
        sent = load_sentiment_score(target_hour)
        
        # LIVE MTF veto checks (mirrors meta_live)
        if use_mtf:
            live_atr = df.iloc[i].get("_live_atr", None) if "_live_atr" in df.columns else None
            live_hist = df.iloc[i].get("_live_macd_hist", None) if "_live_macd_hist" in df.columns else None
            if live_atr is not None and pd.notna(live_atr) and live_atr < atr_min_pct:
                continue  # low-vol chop => NO-TRADE
        # Compute probability with or without MTF
        if use_mtf and 'rsi_at_open' in row:
            prob_up = combine_probability_mtf(
                row,
                daily_rsi=row.get('rsi_at_open', 50.0),
                daily_bollinger_pos=row.get('bollinger_pos_at_open', 0.5),
                daily_momentum=row.get('momentum_at_open', 0.0),
                weekly_trend=weekly_trend_val,
                lam=lam,
                sentiment=sent,
            )
            # LIVE MACD veto (not slot-average)
            live_hist = df.iloc[i].get("_live_macd_hist", 0.0) if "_live_macd_hist" in df.columns else row.get('macd_hist_at_open', 0.0)
            prob_dn_tmp = 1.0 - prob_up
            side_tmp = "BUY" if prob_up >= prob_dn_tmp else "SELL"
            veto = False
            if side_tmp == "BUY" and live_hist is not None and pd.notna(live_hist) and live_hist < -0.1:
                veto = True
            elif side_tmp == "SELL" and live_hist is not None and pd.notna(live_hist) and live_hist > 0.1:
                veto = True
            if veto:
                continue  # MACD opposes => NO-TRADE
            # dust opposite -> 20% shave
            if (side_tmp == "BUY" and live_hist is not None and -0.1 <= live_hist < 0) or \
               (side_tmp == "SELL" and live_hist is not None and 0 < live_hist <= 0.1):
                if side_tmp == "BUY":
                    prob_up = prob_up * 0.8
                else:
                    prob_up = 1 - (1 - prob_up) * 0.8
            prob_dn = 1.0 - prob_up
        else:
            prob_up = combine_probability(row, lam=lam, sentiment=sent)
            prob_dn = 1.0 - prob_up
            if use_mtf:
                # use_mtf but row lacks MTF cols -> still apply weekly nudge via base prob
                # (edge case: early hist with <30 samples before ATR/MACD ready)
                pass
        
        side = None
        prob = None
        
        if prob_up >= threshold and prob_up >= prob_dn:
            side = "BUY"
            prob = prob_up
        elif prob_dn >= threshold and prob_dn > prob_up:
            side = "SELL"
            prob = prob_dn
        else:
            continue
        
        actual_ret = df.iloc[i + 1]["ret"]
        if pd.isna(actual_ret):
            continue
        
        win = (actual_ret > 0) if side == "BUY" else (actual_ret < 0)
        
        trades.append({
            "timestamp": ts.isoformat(),
            "side": side,
            "prob": prob,
            "edge_mean_ret": float(row["mean_ret"]),
            "edge_std_ret": float(row["std_ret"]),
            "actual_ret": float(actual_ret),
            "win": int(win),
            "n_samples": int(row["count"]),
        })
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print(f"[backtest] No trades at threshold={threshold}")
        return trades_df, {}
    
    total = len(trades_df)
    wins = trades_df["win"].sum()
    hit_rate = wins / total
    avg_ret = trades_df["actual_ret"].mean()
    trades_df["pnl_bps"] = trades_df["actual_ret"] * 10000
    total_pnl_bps = trades_df["pnl_bps"].sum()
    sharpe = trades_df["actual_ret"].mean() / trades_df["actual_ret"].std() * np.sqrt(252 * 48) if trades_df["actual_ret"].std() > 0 else 0
    
    cumulative = trades_df["pnl_bps"].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    trades_df["hour"] = pd.to_datetime(trades_df["timestamp"]).dt.hour
    by_hour = trades_df.groupby("hour").agg(
        trades=("win", "count"),
        hit_rate=("win", "mean"),
        avg_ret=("actual_ret", "mean")
    ).sort_values("hit_rate", ascending=False)
    
    summary = {
        "threshold": threshold,
        "total_trades": int(total),
        "wins": int(wins),
        "losses": int(total - wins),
        "hit_rate": float(hit_rate),
        "avg_return_bps": float(avg_ret * 10000),
        "total_pnl_bps": float(total_pnl_bps),
        "sharpe_annualized": float(sharpe),
        "max_drawdown_bps": float(max_dd),
        "best_hour": int(by_hour.index[0]) if len(by_hour) > 0 else None,
        "worst_hour": int(by_hour.index[-1]) if len(by_hour) > 0 else None,
        "by_hour": by_hour.to_dict("index") if len(by_hour) > 0 else {},
        "use_mtf": use_mtf,
    }
    
    return trades_df, summary


def deflated_sharpe_ratio(sharpe_observed: float, n_trials: int, n_observations: int,
                          skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Compute the deflated Sharpe ratio (DSTR).
    
    Adjusts the observed Sharpe ratio for multiple testing bias.
    Based on Bailey & Lopez de Prado (2014).
    
    Args:
        sharpe_observed: The observed annualized Sharpe ratio
        n_trials: Number of independent strategies/trials tested
        n_observations: Number of return observations
        skew: Skewness of returns
        kurtosis: Kurtosis of returns (normal = 3)
    
    Returns:
        Deflated Sharpe ratio (probability that the observed Sharpe is not due to chance)
    """
    from scipy import stats
    
    # Expected Sharpe under null (multiple testing)
    e_max_sharpe = stats.norm.ppf(1 - 1/n_trials) if n_trials > 1 else 0
    
    # Standard error of Sharpe ratio
    se_sharpe = np.sqrt((1 + 0.5 * sharpe_observed**2 - 
                         skew * sharpe_observed + 
                         (kurtosis - 3) / 4 * sharpe_observed**2) / (n_observations - 1))
    
    # Deflated Sharpe ratio
    if se_sharpe > 0:
        z = (sharpe_observed - e_max_sharpe) / se_sharpe
        dstr = stats.norm.cdf(z)
    else:
        dstr = 0.5
    
    return dstr


def purged_cross_validation(trades_df: pd.DataFrame, n_splits: int = 5,
                            embargo_pct: float = 0.01) -> dict:
    """Perform purged K-fold cross-validation on trade results.
    
    Purging removes overlapping trades and embargo prevents information leakage.
    Based on de Prado (2018) "Advances in Financial Machine Learning".
    
    Args:
        trades_df: DataFrame with 'timestamp', 'win', 'next_ret' columns
        n_splits: Number of CV splits
        embargo_pct: Percentage of data to embargo between train/test
    
    Returns:
        Dictionary with CV statistics
    """
    if len(trades_df) < n_splits * 10:
        return {"error": "insufficient trades for CV", "n_trades": len(trades_df)}
    
    # Sort by timestamp
    df = trades_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    embargo_size = max(1, int(n * embargo_pct))
    
    # Generate purged splits
    fold_size = n // n_splits
    cv_results = []
    
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        
        # Embargo: remove embargo_size samples after test set from training
        train_end = max(0, test_start - embargo_size)
        train_start = min(n, test_end + embargo_size)
        
        # Get train and test indices
        test_idx = list(range(test_start, test_end))
        train_idx = list(range(0, train_start)) + list(range(train_end, n))
        
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        
        # Compute metrics on test set
        test_df = df.iloc[test_idx]
        hit_rate = test_df["win"].mean()
        avg_ret = test_df["next_ret"].mean()
        
        # Compute Sharpe on test set
        if test_df["next_ret"].std() > 0:
            sharpe = test_df["next_ret"].mean() / test_df["next_ret"].std() * np.sqrt(252 * 48)
        else:
            sharpe = 0
        
        cv_results.append({
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "hit_rate": hit_rate,
            "avg_ret_bps": avg_ret * 10000,
            "sharpe": sharpe,
        })
    
    if not cv_results:
        return {"error": "no valid CV folds"}
    
    # Aggregate results
    hit_rates = [r["hit_rate"] for r in cv_results]
    sharpes = [r["sharpe"] for r in cv_results]
    
    return {
        "n_folds": len(cv_results),
        "mean_hit_rate": np.mean(hit_rates),
        "std_hit_rate": np.std(hit_rates),
        "mean_sharpe": np.mean(sharpes),
        "std_sharpe": np.std(sharpes),
        "min_hit_rate": min(hit_rates),
        "max_hit_rate": max(hit_rates),
        "folds": cv_results,
    }


def run_validated_backtest(threshold: float = 0.55, lam: float = 0.3,
                           min_samples: int = 50, warmup_bars: int = 1000,
                           filename: str = None, use_mtf: bool = False,
                           atr_min_pct: float = 0.003, n_trials: int = 100):
    """Run walk-forward backtest with deflated Sharpe and purged CV validation.
    
    Returns comprehensive validation report.
    """
    print("=" * 60)
    print("VALIDATED BACKTEST WITH DEFLATED SHARPE & PURGED CV")
    print("=" * 60)
    
    # Run walk-forward backtest
    trades_df, summary = run_backtest_walkforward(
        threshold=threshold, lam=lam, min_samples=min_samples,
        warmup_bars=warmup_bars, filename=filename, use_mtf=use_mtf,
        atr_min_pct=atr_min_pct
    )
    
    if len(trades_df) == 0:
        print("[validated] No trades generated")
        return {"error": "no trades", "summary": summary}
    
    print(f"\n[validated] Walk-forward results:")
    print(f"  Trades: {summary['total_trades']}")
    print(f"  Hit rate: {summary['hit_rate']:.2%}")
    print(f"  Sharpe: {summary['sharpe_annualized']:.2f}")
    
    # Deflated Sharpe ratio
    print(f"\n[validated] Computing deflated Sharpe ratio (n_trials={n_trials})...")
    dstr = deflated_sharpe_ratio(
        sharpe_observed=summary['sharpe_annualized'],
        n_trials=n_trials,
        n_observations=summary['total_trades'],
        skew=trades_df["next_ret"].skew() if "next_ret" in trades_df else 0,
        kurtosis=trades_df["next_ret"].kurtosis() + 3 if "next_ret" in trades_df else 3,
    )
    print(f"  Deflated Sharpe ratio: {dstr:.4f}")
    print(f"  P(Sharpe is real): {dstr:.2%}")
    
    # Purged cross-validation
    print(f"\n[validated] Running purged 5-fold cross-validation...")
    cv_results = purged_cross_validation(trades_df, n_splits=5, embargo_pct=0.01)
    
    if "error" in cv_results:
        print(f"  CV failed: {cv_results['error']}")
    else:
        print(f"  CV mean hit rate: {cv_results['mean_hit_rate']:.2%} ± {cv_results['std_hit_rate']:.2%}")
        print(f"  CV mean Sharpe: {cv_results['mean_sharpe']:.2f} ± {cv_results['std_sharpe']:.2f}")
        print(f"  CV min/max hit rate: {cv_results['min_hit_rate']:.2%} / {cv_results['max_hit_rate']:.2%}")
    
    # Quality gate with deflation
    print(f"\n[validated] QUALITY GATE (with deflation adjustment):")
    gate_hit_rate = summary['hit_rate'] >= 0.53
    gate_trades = summary['total_trades'] >= 200
    gate_sharpe = summary['sharpe_annualized'] >= 1.0
    gate_dstr = dstr >= 0.95  # 95% confidence the Sharpe is real
    
    if "error" not in cv_results:
        gate_cv = cv_results['mean_hit_rate'] >= 0.52  # CV-adjusted threshold
    else:
        gate_cv = False
    
    print(f"  Hit rate >= 53%:      {'PASS' if gate_hit_rate else 'FAIL'} ({summary['hit_rate']:.2%})")
    print(f"  Trades >= 200:        {'PASS' if gate_trades else 'FAIL'} ({summary['total_trades']})")
    print(f"  Sharpe >= 1.0:        {'PASS' if gate_sharpe else 'FAIL'} ({summary['sharpe_annualized']:.2f})")
    print(f"  Deflated Sharpe >= 95%: {'PASS' if gate_dstr else 'FAIL'} ({dstr:.2%})")
    if "error" not in cv_results:
        print(f"  CV hit rate >= 52%:   {'PASS' if gate_cv else 'FAIL'} ({cv_results['mean_hit_rate']:.2%})")
    
    all_passed = gate_hit_rate and gate_trades and gate_sharpe and gate_dstr and gate_cv
    
    if all_passed:
        print(f"\n  [PASS] ALL QUALITY GATES PASSED")
    else:
        print(f"\n  [FAIL] ONE OR MORE QUALITY GATES FAILED")
    
    # Compile full report
    report = {
        "summary": summary,
        "deflated_sharpe": dstr,
        "cv_results": cv_results,
        "gates": {
            "hit_rate": gate_hit_rate,
            "trades": gate_trades,
            "sharpe": gate_sharpe,
            "deflated_sharpe": gate_dstr,
            "cv": gate_cv,
            "all_passed": all_passed,
        },
    }
    
    return report


def main():
    print("=" * 60)
    print("CEREBRUM TRADER BOT — BACKTEST RESULTS")
    print("=" * 60)
    
    # Run the fast vectorized version first (standard mode)
    print("\n--- FAST VECTORIZED BACKTEST (full-history edge, slight lookahead) ---")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    
    all_summaries = []
    
    for thresh in thresholds:
        print(f"\nThreshold: {thresh}")
        trades_df, summary = run_backtest_vectorized(threshold=thresh)
        if len(trades_df) == 0:
            print(f"  No trades triggered")
            continue
        all_summaries.append(summary)
        
        print(f"  Total trades:     {summary['total_trades']}")
        print(f"  Win/Loss:         {summary['wins']} / {summary['losses']}")
        print(f"  Hit rate:         {summary['hit_rate']:.2%}")
        print(f"  Avg return:       {summary['avg_return_bps']:.2f} bps")
        print(f"  Total PnL:        {summary['total_pnl_bps']:.0f} bps")
        print(f"  Sharpe (annual):  {summary['sharpe_annualized']:.2f}")
        print(f"  Max drawdown:     {summary['max_drawdown_bps']:.0f} bps")
        if summary['best_hour'] is not None:
            print(f"  Best hour:        {summary['best_hour']:02d}:00")
            print(f"  Worst hour:       {summary['worst_hour']:02d}:00")
    
    print("\n" + "=" * 60)
    print("QUALITY GATE CHECK (vectorized)")
    print("=" * 60)
    print("Hard gate: hit_rate >= 53% AND trades >= 200 at ANY threshold")
    
    gate_passed = False
    for s in all_summaries:
        if s["hit_rate"] >= 0.53 and s["total_trades"] >= 200:
            print(f"  [PASS] at threshold {s['threshold']}: hit_rate={s['hit_rate']:.2%}, trades={s['total_trades']}")
            gate_passed = True
        else:
            print(f"  [FAIL] at threshold {s['threshold']}: hit_rate={s['hit_rate']:.2%}, trades={s['total_trades']}")
    
    if not gate_passed:
        print("\n>>> VERDICT: Quality gate FAILED. Do NOT proceed to auto-execution.")
        print(">>> XAU/USD 4h is likely a random walk at retail timeframes with this model.")
    else:
        print("\n>>> VERDICT: Quality gate PASSED. Proceed to Phase 3 (auto-execution) if desired.")
    
    if all_summaries:
        best = max(all_summaries, key=lambda s: s["hit_rate"] * min(s["total_trades"], 500))
        print(f"\nBest threshold by risk-adjusted metric: {best['threshold']}")
    
    # Run TRUE walk-forward backtest
    print("\n" + "=" * 60)
    print("TRUE WALK-FORWARD BACKTEST (zero lookahead, rolling window)")
    print("=" * 60)
    best_thresh = best['threshold'] if all_summaries else 0.6
    print(f"Running walk-forward at threshold={best_thresh}...")
    trades_df, summary = run_backtest_walkforward(threshold=best_thresh)
    if len(trades_df) > 0:
        print(f"  Total trades:     {summary['total_trades']}")
        print(f"  Hit rate:         {summary['hit_rate']:.2%}")
        print(f"  Avg return:       {summary['avg_return_bps']:.2f} bps")
        print(f"  Sharpe:           {summary['sharpe_annualized']:.2f}")
        print(f"  Max drawdown:     {summary['max_drawdown_bps']:.0f} bps")

        if summary['hit_rate'] >= 0.53 and summary['total_trades'] >= 200:
            print("\n  [PASS] Walk-forward PASSED quality gate!")
        else:
            print("\n  [FAIL] Walk-forward FAILED quality gate - results may be overfit.")
    
    # Run validated backtest with deflated Sharpe and purged CV
    print("\n" + "=" * 60)
    print("VALIDATED BACKTEST (deflated Sharpe + purged CV)")
    print("=" * 60)
    report = run_validated_backtest(threshold=best_thresh, n_trials=len(thresholds))
    
    if report.get("gates", {}).get("all_passed"):
        print("\n>>> FINAL VERDICT: VALIDATED. Proceed to Phase 3 (auto-execution) if desired.")
    else:
        print("\n>>> FINAL VERDICT: NOT VALIDATED. Do NOT proceed to auto-execution.")


if __name__ == "__main__":
    main()