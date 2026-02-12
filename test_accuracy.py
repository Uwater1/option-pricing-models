"""
Test the accuracy of estimate_fair_value.py under production conditions.
Production constraint: only last_price (estimated fair price), strike, underlying, days, and type are available.
No bid, ask, volume, or open interest.
"""
import pandas as pd
import numpy as np
import glob
import os

from estimate_fair_value import estimate_fair_value

def test_on_file(file_path, ticker_label):
    """Test estimate_fair_value on a CSV file and return MAE stats."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  File not found: {file_path}")
        return None

    # Basic filtering (same as training)
    df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
    df = df[
        (df['bid'] > 0) & 
        (df['ask'] >= 0.01) & 
        (df['lastPrice'] > 0) &
        (df['volume'] >= 5) &
        (df['openInterest'] >= 5) &
        (df['lastTradeDate'] >= '2025-02-09')
    ].copy()
    
    if 'underlyingPrice' not in df.columns:
        df['underlyingPrice'] = df['strike'] * df['moneyness']
    
    # Moneyness filter (same as training)
    df = df[
        (df['moneyness'] > 0.5) & (df['moneyness'] < 1.5) &
        (df['lastPrice'] > 0.05)
    ].copy()
    
    df['actual_spread'] = df['ask'] - df['bid']
    df['mid_price'] = (df['ask'] + df['bid']) / 2.0
    
    # Filter to positive spreads
    df = df[df['actual_spread'] > 0].copy()
    
    if len(df) == 0:
        print(f"  No valid rows for {ticker_label}")
        return None
    
    # --- Production test: only last_price, no bid/ask/volume/oi ---
    pred_spreads = []
    methods = []
    skipped = 0
    
    for _, row in df.iterrows():
        res = estimate_fair_value(
            last_price=row['lastPrice'],
            strike=row['strike'],
            underlying_price=row['underlyingPrice'],
            days_to_expire=row['days_to_expire'],
            option_type=row['optionType']
            # No bid, ask, volume, or open_interest — production mode
        )
        if res is None:
            pred_spreads.append(np.nan)
            methods.append('N/A')
            skipped += 1
        else:
            pred_spreads.append(res['spread'])
            methods.append(res['method'])
    
    df['pred_spread'] = pred_spreads
    df['method'] = methods
    
    # Drop failed predictions
    df = df.dropna(subset=['pred_spread'])
    
    if len(df) == 0:
        return None
    
    # Calculate metrics
    df['error'] = (df['pred_spread'] - df['actual_spread']).abs()
    mae = df['error'].mean()
    median_ae = df['error'].median()
    mean_spread = df['actual_spread'].mean()
    rmse = np.sqrt(((df['pred_spread'] - df['actual_spread'])**2).mean())
    
    # Breakdown by method
    method_counts = df['method'].value_counts()
    
    print(f"\n  {ticker_label}: {len(df)} options tested (skipped {skipped})")
    print(f"    Mean Actual Spread:  ${mean_spread:.4f}")
    print(f"    MAE:                 ${mae:.4f}")
    print(f"    Median AE:           ${median_ae:.4f}")
    print(f"    RMSE:                ${rmse:.4f}")
    print(f"    MAE / Mean Spread:   {mae/mean_spread:.2%}")
    for method, count in method_counts.items():
        print(f"    Method: {method} ({count} rows)")
    
    return {
        'ticker': ticker_label,
        'n_options': len(df),
        'mean_actual_spread': mean_spread,
        'mae': mae,
        'median_ae': median_ae,
        'rmse': rmse,
        'error_pct': mae / mean_spread
    }

def main():
    print("=" * 60)
    print("  Fair Value Estimation — Production Accuracy Test")
    print("  (Only last_price provided, no bid/ask/volume/OI)")
    print("=" * 60)
    
    # Test on all available data
    tickers_files = [
        ("NVDA", "data/NVDA_options.csv"),
        ("QQQ", "data/QQQ_options.csv"),
        ("TSLA", "data/TSLA_options.csv"),
        ("IBIT", "data/IBIT_options.csv"),
        ("AAPL", "data/AAPL_options.csv"),
        ("AMZN", "data/AMZN_options.csv"),
        ("SPY", "data/SPY_options.csv"),
        ("NVDA_2", "data/NVDA_options_2.csv"),
        ("QQQ_2", "data/QQQ_options_2.csv"),
        ("TSLA_2", "data/TSLA_options_2.csv"),
    ]
    
    results = []
    for ticker, fpath in tickers_files:
        if os.path.exists(fpath):
            res = test_on_file(fpath, ticker)
            if res:
                results.append(res)
    
    if not results:
        print("\nNo test results generated.")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    res_df = pd.DataFrame(results)
    
    # Per-ticker table
    print(f"\n{'Ticker':<10} {'N':>6} {'Mean Spread':>12} {'MAE':>8} {'MedAE':>8} {'RMSE':>8} {'MAE%':>8}")
    print("-" * 62)
    for _, row in res_df.iterrows():
        print(f"{row['ticker']:<10} {row['n_options']:>6} ${row['mean_actual_spread']:>10.4f} ${row['mae']:>6.4f} ${row['median_ae']:>6.4f} ${row['rmse']:>6.4f} {row['error_pct']:>7.2%}")
    
    # Overall weighted averages
    total_n = res_df['n_options'].sum()
    weighted_mae = (res_df['mae'] * res_df['n_options']).sum() / total_n
    weighted_rmse = (res_df['rmse'] * res_df['n_options']).sum() / total_n
    weighted_mean_spread = (res_df['mean_actual_spread'] * res_df['n_options']).sum() / total_n
    
    print("-" * 62)
    print(f"{'OVERALL':<10} {total_n:>6} ${weighted_mean_spread:>10.4f} ${weighted_mae:>6.4f} {'':>9} ${weighted_rmse:>6.4f} {weighted_mae/weighted_mean_spread:>7.2%}")

if __name__ == "__main__":
    main()
