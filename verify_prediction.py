import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from estimate_fair_value import estimate_fair_value

def verify_on_ticker(ticker, file_path):
    """
    Verify estimate_fair_value under PRODUCTION conditions.
    Only last_price is provided — no bid, ask, volume, or open interest.
    """
    print(f"\n--- Verifying on {ticker} (Production Mode: last_price only) ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    # Apply same filters as training
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

    # Moneyness filter
    df = df[
        (df['moneyness'] > 0.5) & (df['moneyness'] < 1.5) &
        (df['lastPrice'] > 0.05)
    ].copy()
    
    df['actual_spread'] = df['ask'] - df['bid']
    df = df[df['actual_spread'] > 0].copy()
    
    if len(df) == 0:
        print(f"  No valid rows for {ticker}")
        return None

    # --- Production test: only last_price, no bid/ask/volume/oi ---
    pred_spreads = []
    skipped = 0
    
    for _, row in df.iterrows():
        res = estimate_fair_value(
            last_price=row['lastPrice'],
            strike=row['strike'],
            underlying_price=row['underlyingPrice'],
            days_to_expire=row['days_to_expire'],
            option_type=row['optionType']
        )
        if res is None:
            pred_spreads.append(np.nan)
            skipped += 1
        else:
            pred_spreads.append(res['spread'])
    
    df['pred_spread'] = pred_spreads
    df = df.dropna(subset=['pred_spread'])
    
    if len(df) == 0:
        return None
    
    df['error'] = (df['pred_spread'] - df['actual_spread']).abs()
    
    mae = df['error'].mean()
    median_ae = df['error'].median()
    mean_spread = df['actual_spread'].mean()
    rmse = np.sqrt(((df['pred_spread'] - df['actual_spread'])**2).mean())
    
    print(f"  Options tested: {len(df)} (skipped {skipped})")
    print(f"  Mean Spread:    ${mean_spread:.4f}")
    print(f"  MAE:            ${mae:.4f}")
    print(f"  Median AE:      ${median_ae:.4f}")
    print(f"  RMSE:           ${rmse:.4f}")
    print(f"  MAE % Spread:   {mae/mean_spread:.2%}")
    
    return df

def main():
    check_tickers = ["NVDA", "QQQ", "TSLA", "IBIT"]
    dfs = []
    
    for t in check_tickers:
        res = verify_on_ticker(t, f"data/{t}_options.csv")
        if res is not None:
            res['Ticker'] = t
            dfs.append(res)
            
    if dfs:
        combined = pd.concat(dfs)
        
        # Overall MAE
        overall_mae = combined['error'].mean()
        overall_mean_spread = combined['actual_spread'].mean()
        overall_rmse = np.sqrt(((combined['pred_spread'] - combined['actual_spread'])**2).mean())
        
        print(f"\n{'='*50}")
        print(f"  OVERALL ({len(combined)} options)")
        print(f"{'='*50}")
        print(f"  Mean Spread:    ${overall_mean_spread:.4f}")
        print(f"  MAE:            ${overall_mae:.4f}")
        print(f"  RMSE:           ${overall_rmse:.4f}")
        print(f"  MAE % Spread:   {overall_mae/overall_mean_spread:.2%}")
        
        # Plot
        os.makedirs("analysis_results", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.scatter(combined['actual_spread'], combined['pred_spread'], alpha=0.1, s=5)
        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.xlabel('Actual Spread')
        plt.ylabel('Predicted Spread')
        plt.title('Spread Prediction — Production Mode (last_price only)')
        plt.savefig("analysis_results/xgb_verification.png")
        print("\nSaved plot to analysis_results/xgb_verification.png")

if __name__ == "__main__":
    main()
