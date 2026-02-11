import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import ctypes
import os

# Load C++ Library
lib_path = os.path.join(os.path.dirname(__file__), 'libbs.so')
lib = ctypes.CDLL(lib_path)
lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.implied_volatility.restype = ctypes.c_double
lib.black_scholes_vega.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.black_scholes_vega.restype = ctypes.c_double
lib.black_scholes_gamma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.black_scholes_gamma.restype = ctypes.c_double

def fast_iv(price, S, K, T, r, is_call):
    return lib.implied_volatility(price, S, K, T, r, is_call)
def fast_vega(S, K, T, r, sigma):
    return lib.black_scholes_vega(S, K, T, r, sigma)
def fast_gamma(S, K, T, r, sigma):
    return lib.black_scholes_gamma(S, K, T, r, sigma)

def verify_on_ticker_xgb(ticker, file_path, model_path="spread_xgb.json"):
    print(f"\n--- Verifying XGBoost Model on {ticker} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['lastPrice'] > 0)].copy()
    
    if 'underlyingPrice' not in df.columns:
        df['underlyingPrice'] = df['strike'] * df['moneyness']
    
    print("Calculating Features...")
    r = 0.05
    ivs, gammas, vegas, inv_prices = [], [], [], []
    
    for _, row in df.iterrows():
        time_years = max(row['time_to_expire_years'], 0.001)
        price = row['lastPrice']
        S = row['underlyingPrice']
        K = row['strike']
        is_call = 1 if row['optionType'].lower() == 'call' else 0
        
        iv = fast_iv(price, S, K, time_years, r, is_call)
        if iv < 0.001: iv = 0.001
        
        gamma = fast_gamma(S, K, time_years, r, iv)
        vega = fast_vega(S, K, time_years, r, iv)
        
        ivs.append(iv)
        gammas.append(gamma)
        vegas.append(vega)
        inv_prices.append(1.0/price if price > 0 else 0)
        
    df['iv'] = ivs
    df['gamma'] = gammas
    df['vega'] = vegas
    df['inv_price'] = inv_prices
    
    # Predict
    features = pd.DataFrame({
        'moneyness': df['moneyness'],
        'time_to_expire_years': df['time_to_expire_years'],
        'iv': df['iv'],
        'gamma': df['gamma'],
        'vega': df['vega'],
        'inv_price': df['inv_price'],
        'mid': df['lastPrice'] # Using last price as 'mid' proxy for model feature, or mid_price if calculated
    })
    
    # Ideally should use exact same features as training.
    # Training used 'mid' as (ask+bid)/2. For verification we know ask/bid, so let's use actual mid.
    features['mid'] = df['mid_price']
    
    dtest = xgb.DMatrix(features)
    model = xgb.Booster()
    model.load_model(model_path)
    
    df['pred_spread'] = model.predict(dtest)
    df['actual_spread'] = df['ask'] - df['bid']
    df['error'] = (df['pred_spread'] - df['actual_spread']).abs()
    
    mae = df['error'].mean()
    mean_spread = df['actual_spread'].mean()
    
    print(f"MAE:            ${mae:.4f}")
    print(f"Mean Spread:    ${mean_spread:.4f}")
    print(f"Error % Spread: {mae/mean_spread:.2%}")
    
    return df

def main():
    check_tickers = ["NVDA", "QQQ", "TSLA", "IBIT"]
    dfs = []
    
    for t in check_tickers:
        res = verify_on_ticker_xgb(t, f"data/{t}_options.csv")
        if res is not None:
            res['Ticker'] = t
            dfs.append(res)
            
    if dfs:
        combined = pd.concat(dfs)
        plt.figure(figsize=(10, 5))
        plt.scatter(combined['actual_spread'], combined['pred_spread'], alpha=0.1, s=5)
        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.xlabel('Actual Spread')
        plt.ylabel('Predicted Spread')
        plt.title('XGBoost Spread Prediction (Absolute $)')
        plt.savefig("analysis_results/xgb_verification.png")
        print("\nSaved plot to analysis_results/xgb_verification.png")

if __name__ == "__main__":
    main()
