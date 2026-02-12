import pandas as pd
import numpy as np
import glob
import os
import ctypes
from scipy.optimize import curve_fit

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
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        print("No data found.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    full_df = full_df[
        (full_df['bid'] > 0) & 
        (full_df['ask'] > 0) & 
        (full_df['lastPrice'] > 0) &
        (full_df['volume'] >= 5)
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
            row['lastPrice'], 
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
    
    print(f"Training on {len(full_df)} rows.")
    
    X_data = (full_df['moneyness'].values, full_df['time_to_expire_years'].values, full_df['iv'].values)
    y_data = full_df['relative_spread'].values
    
    popt, pcov = curve_fit(polynomial_model, X_data, y_data)
    
    print("\n--- Model Coefficients ---")
    coeffs = {
        'b0': popt[0], 'b1': popt[1], 'b2': popt[2], 
        'b3': popt[3], 'b4': popt[4], 'b5': popt[5], 'b6': popt[6]
    }
    
    for k, v in coeffs.items():
        print(f"{k}: {v:.6f}")
    
    print("\nCopy this dictionary into your prediction script:")
    print(coeffs)
    
    y_pred = polynomial_model(X_data, *popt)
    mae = np.mean(np.abs(y_pred - y_data))
    print(f"\nMean Absolute Error (Relative Spread): {mae:.4f}")

if __name__ == "__main__":
    train_model()
