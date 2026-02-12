import pandas as pd
import numpy as np
import glob
import os
import ctypes
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load C++ Library
lib = ctypes.CDLL('./libbs.so')
# Define argument types
lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.implied_volatility.restype = ctypes.c_double
lib.black_scholes_vega.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.black_scholes_vega.restype = ctypes.c_double
lib.black_scholes_gamma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.black_scholes_gamma.restype = ctypes.c_double
lib.black_scholes_delta.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.black_scholes_delta.restype = ctypes.c_double
lib.black_scholes_theta.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.black_scholes_theta.restype = ctypes.c_double

def fast_iv(price, S, K, T, r, option_type):
    is_call = 1 if option_type.lower() == 'call' else 0
    return lib.implied_volatility(price, S, K, T, r, is_call)

def fast_vega(S, K, T, r, sigma):
    return lib.black_scholes_vega(S, K, T, r, sigma)

def fast_gamma(S, K, T, r, sigma):
    return lib.black_scholes_gamma(S, K, T, r, sigma)

def fast_delta(S, K, T, r, sigma, option_type):
    is_call = 1 if option_type.lower() == 'call' else 0
    return lib.black_scholes_delta(S, K, T, r, sigma, is_call)

def fast_theta(S, K, T, r, sigma, option_type):
    is_call = 1 if option_type.lower() == 'call' else 0
    return lib.black_scholes_theta(S, K, T, r, sigma, is_call)

def train_xgboost():
    # 1. Load Data
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
    
    # Filter outliers
    full_df = full_df[
        (full_df['moneyness'] > 0.5) & (full_df['moneyness'] < 1.5) &
        (full_df['spread'] > 0) & (full_df['spread'] < 25.0) & # Absolute spread sanity check
        (full_df['mid'] > 0.05) 
    ].copy()

    print("Calculating Features (IV, Gamma, Vega, 1/Price)...")
    
    if 'underlyingPrice' not in full_df.columns:
         full_df['underlyingPrice'] = full_df['strike'] * full_df['moneyness']

    r = 0.05
    
    ivs = []
    gammas = []
    vegas = []
    inv_prices = []
    
    for _, row in full_df.iterrows():
        time_years = max(row['time_to_expire_years'], 0.001)
        price = row['lastPrice']
        S = row['underlyingPrice']
        K = row['strike']
        
        # IV
        iv = fast_iv(price, S, K, time_years, r, row['optionType'])
        
        # Clamp IV for stability
        if iv < 0.001: iv = 0.001
        if iv > 5.0: iv = 5.0
            
        # Gamma & Vega
        gamma = fast_gamma(S, K, time_years, r, iv)
        vega = fast_vega(S, K, time_years, r, iv)
        
        ivs.append(iv)
        gammas.append(gamma)
        vegas.append(vega)
        inv_prices.append(1.0 / price if price > 0 else 0.0)

    full_df['iv'] = ivs
    full_df['gamma'] = gammas
    full_df['vega'] = vegas
    full_df['delta'] = [fast_delta(row['underlyingPrice'], row['strike'], max(row['time_to_expire_years'], 0.001), 0.05, iv, row['optionType']) for (_, row), iv in zip(full_df.iterrows(), ivs)]
    full_df['theta'] = [fast_theta(row['underlyingPrice'], row['strike'], max(row['time_to_expire_years'], 0.001), 0.05, iv, row['optionType']) for (_, row), iv in zip(full_df.iterrows(), ivs)]
    full_df['log_volume'] = np.log1p(full_df['volume'])
    full_df['log_oi'] = np.log1p(full_df['openInterest'])
    full_df['is_call'] = (full_df['optionType'] == 'call').astype(int)
    full_df['dist_from_atm'] = np.abs(full_df['moneyness'] - 1.0)
    full_df['inv_price'] = inv_prices
    
    # Filter valid IV
    full_df = full_df[(full_df['iv'] > 0.01) & (full_df['iv'] < 5.0)].copy()
    
    print(f"Training on {len(full_df)} rows.")
    
    # Features
    features = [
        'moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega', 
        'delta', 'theta', 'log_volume', 'log_oi', 'is_call', 'dist_from_atm',
        'inv_price', 'mid'
    ]
    # Target: Absolute Spread ($)
    target = 'spread'
    
    X = full_df[features]
    y = full_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mean_spread = y_test.mean()
    
    print(f"\n--- XGBoost Performance ---")
    print(f"Mean Absolute Error (Spread $): ${mae:.4f}")
    print(f"Mean Spread in Test Set:        ${mean_spread:.4f}")
    print(f"Error as % of Spread:           {mae/mean_spread:.2%}")
    
    # Feature Importance
    print("\nFeature Importance:")
    for name, imp in zip(features, model.feature_importances_):
        print(f"{name}: {imp:.4f}")
        
    # Save Model
    model.save_model("spread_xgb.json")
    print("\nModel saved to spread_xgb.json")

if __name__ == "__main__":
    train_xgboost()
