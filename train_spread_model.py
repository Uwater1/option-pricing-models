import pandas as pd
import numpy as np
import glob
import os
import ctypes
from scipy.optimize import curve_fit
import re

# Load C++ Library
lib = ctypes.CDLL('./libbs.so')
lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.implied_volatility.restype = ctypes.c_double

def fast_iv(price, S, K, T, r, option_type):
    # Determine Call/Put flag (1 for Call, 0 for Put)
    is_call = 1 if option_type.lower() == 'call' else 0
    return lib.implied_volatility(price, S, K, T, r, is_call)

def polynomial_model(X, b0, b1, b2, b3, b4, b5, b6):
    M, T, IV = X
    return b0 + b1*M + b2*T + b3*IV + b4*M**2 + b5*T**2 + b6*T*IV

def train_model():
    files = glob.glob("data/*.csv")
    print(f"Loading files: {files}")
    
    dfs = []
    for f in files:
        try:
            # ONLY use files with "options" in the name (e.g., AAPL_options.csv)
            # Ignore other files like AAPL-2026-02-20.csv
            if "options" not in f:
                continue

            # Determine group ID
            # _2.csv -> 2, _3.csv -> 3, _4.csv -> 4, else 1 (no suffix)
            match = re.search(r'_(\d+)\.csv$', f)
            if match:
                group_id = int(match.group(1))
            else:
                group_id = 1
            
            df = pd.read_csv(f)
            df['group_id'] = group_id
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        print("No 'options' data found.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    full_df['lastTradeDate'] = pd.to_datetime(full_df['lastTradeDate'])
    
    full_df = full_df[
        (full_df['bid'] > 0) & 
        (full_df['ask'] >= 0.01) & 
        (full_df['lastPrice'] > 0) &
        (full_df['volume'] >= 5) &
        (full_df['openInterest'] >= 5) &
        (full_df['lastTradeDate'] >= '2025-02-09')
    ].copy()
    
    full_df['spread'] = full_df['ask'] - full_df['bid']
    full_df['mid'] = (full_df['ask'] + full_df['bid']) / 2
    full_df['relative_spread'] = full_df['spread'] / full_df['mid']
    
    full_df = full_df[
        (full_df['moneyness'] > 0.5) & (full_df['moneyness'] < 1.5) &
        (full_df['relative_spread'] > 0) & (full_df['relative_spread'] < 0.2) &
        (full_df['mid'] > 0.5) 
    ].copy()

    print("Calculating Implied Volatility (Fast C++)...")
    
    if 'underlyingPrice' not in full_df.columns:
         full_df['underlyingPrice'] = full_df['strike'] * full_df['moneyness']

    r = 0.05
    
    # Vectorized application is tricky with ctypes without loop or vector inputs in C++
    # But loop in Python calling C++ is much faster than loop in Python calling Python
    
    ivs = []
    for _, row in full_df.iterrows():
        # Use Time in Years (Calendar Days is standard: 365)
        # Note: Research shows calendar days (365) is standard for Theta/Decay in BS models 
        # because interest is accrued over calendar time, not just trading days.
        # Volatility is sometimes scaled by sqrt(252), but T is usually 365. 
        # Let's stick to 365.0 for T input to BS.
        time_years = max(row['time_to_expire_years'], 0.001)
        
        iv = fast_iv(
            row['mid'], 
            row['underlyingPrice'], 
            row['strike'], 
            time_years, 
            r, 
            row['optionType']
        )
        ivs.append(iv)

    full_df['iv'] = ivs
    
    # Filter valid IV
    full_df = full_df[(full_df['iv'] > 0.01) & (full_df['iv'] < 5.0)].copy()
    
    
    # Split into Train (Group 1-3) and Test (Group 4)
    train_df = full_df[full_df['group_id'].isin([1, 2, 3])].copy()
    test_df = full_df[full_df['group_id'] == 4].copy()
    
    print(f"Total rows: {len(full_df)}")
    print(f"Training on {len(train_df)} rows (Groups 1-3).")
    print(f"Testing on {len(test_df)} rows (Group 4).")
    
    if len(train_df) == 0:
        print("Error: No training data available.")
        return

    X_train = (train_df['moneyness'].values, train_df['time_to_expire_years'].values, train_df['iv'].values)
    y_train = train_df['relative_spread'].values
    
    popt, pcov = curve_fit(polynomial_model, X_train, y_train)
    
    print("\n--- Model Coefficients ---")
    coeffs = {
        'b0': popt[0], 'b1': popt[1], 'b2': popt[2], 
        'b3': popt[3], 'b4': popt[4], 'b5': popt[5], 'b6': popt[6]
    }
    
    for k, v in coeffs.items():
        print(f"{k}: {v:.6f}")
    
    print("\nCopy this dictionary into your prediction script:")
    print(coeffs)
    
    # Training Error
    y_train_pred = polynomial_model(X_train, *popt)
    mae_train = np.mean(np.abs(y_train_pred - y_train))
    print(f"\nTraining MAE (Relative Spread): {mae_train:.4f}")
    
    # Testing Error
    if len(test_df) > 0:
        X_test = (test_df['moneyness'].values, test_df['time_to_expire_years'].values, test_df['iv'].values)
        y_test = test_df['relative_spread'].values
        y_test_pred = polynomial_model(X_test, *popt)
        mae_test = np.mean(np.abs(y_test_pred - y_test))
        print(f"Testing MAE (Group 4) (Relative Spread): {mae_test:.4f}")
    else:
        print("No testing data available for Group 4.")

if __name__ == "__main__":
    train_model()
