import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import ctypes

# 1. Setup C++ Library (for IV/Greeks)
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

# 2. Define Models
# Old Polynomial Model
POLY_COEFFS = {
    'b0': 0.07217830754411238, 'b1': -0.10741680766063832, 'b2': 0.024817628008708924, 
    'b3': 0.01673389789242793, 'b4': 0.04570831072599171, 'b5': -0.0031195961528276395, 
    'b6': -0.04091022400898136
}
def predict_poly(row):
    M = row['moneyness']
    T = row['time_to_expire_years']
    IV = row['iv']
    # Predicted RELATIVE spread
    rel_spread = (POLY_COEFFS['b0'] + POLY_COEFFS['b1']*M + POLY_COEFFS['b2']*T + 
                  POLY_COEFFS['b3']*IV + POLY_COEFFS['b4']*M**2 + POLY_COEFFS['b5']*T**2 + 
                  POLY_COEFFS['b6']*T*IV)
    rel_spread = max(0.001, min(0.5, rel_spread))
    # Convert to Absolute Prediction ($)
    # The old model predicted relative spread * mid
    start_price = row['mid_price'] if row['mid_price'] > 0 else row['lastPrice']
    return rel_spread * start_price

def predict_xgb(df_features, model_path="spread_xgb.json"):
    model = xgb.Booster()
    model.load_model(model_path)
    # Ensure correct column order
    cols = ['moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega', 'inv_price', 'mid']
    dtest = xgb.DMatrix(df_features[cols])
    return model.predict(dtest)

# 3. Comparison Logic
def compare_ticker(ticker):
    file_path = f"data/{ticker}_options.csv"
    if not os.path.exists(file_path):
        return None
        
    print(f"\nProcessing {ticker}...")
    df = pd.read_csv(file_path)
    
    # Filter
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['ask'] + df['bid']) / 2
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['lastPrice'] > 0)].copy()
    if 'underlyingPrice' not in df.columns:
        df['underlyingPrice'] = df['strike'] * df['moneyness']
        
    # Calculate Features
    r = 0.05
    ivs, gammas, vegas, inv_prices = [], [], [], []
    for _, row in df.iterrows():
        T = max(row['time_to_expire_years'], 0.001)
        is_call = 1 if row['optionType'].lower() == 'call' else 0
        iv = fast_iv(row['lastPrice'], row['underlyingPrice'], row['strike'], T, r, is_call)
        if iv < 0.001: iv = 0.001
        gamma = fast_gamma(row['underlyingPrice'], row['strike'], T, r, iv)
        vega = fast_vega(row['underlyingPrice'], row['strike'], T, r, iv)
        
        ivs.append(iv)
        gammas.append(gamma)
        vegas.append(vega)
        inv_prices.append(1.0/row['lastPrice'])
        
    df['iv'] = ivs
    df['gamma'] = gammas
    df['vega'] = vegas
    df['inv_price'] = inv_prices
    df['mid'] = df['mid_price'] # Feature for XGB
    
    # Predict Poly
    df['pred_poly'] = df.apply(predict_poly, axis=1)
    
    # Predict XGB
    df['pred_xgb'] = predict_xgb(df)
    
    # Actual
    df['actual_spread'] = df['ask'] - df['bid']
    
    # Errors
    mae_poly = (df['pred_poly'] - df['actual_spread']).abs().mean()
    mae_xgb = (df['pred_xgb'] - df['actual_spread']).abs().mean()
    
    improvement = (mae_poly - mae_xgb) / mae_poly * 100
    
    print(f"{ticker} Results:")
    print(f"  Old Poly MAE:  ${mae_poly:.4f}")
    print(f"  New XGB MAE:   ${mae_xgb:.4f}")
    print(f"  Improvement:   {improvement:.2f}%")
    
    return df, mae_poly, mae_xgb

def main():
    tickers = ["NVDA", "QQQ", "TSLA", "IBIT"]
    results = []
    
    for t in tickers:
        res = compare_ticker(t)
        if res:
            results.append({'Ticker': t, 'Poly MAE': res[1], 'XGB MAE': res[2]})
            
    # Summary Table
    res_df = pd.DataFrame(results)
    print("\n--- Final Comparison Summary ---")
    print(res_df.to_string(index=False))
    
    # Bar Chart
    res_df.plot(x='Ticker', y=['Poly MAE', 'XGB MAE'], kind='bar', figsize=(10, 6))
    plt.title("Model Comparison: Mean Absolute Error (Lower is Better)")
    plt.ylabel("MAE ($)")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("analysis_results/model_comparison.png")
    print("\nSaved comparison chart to analysis_results/model_comparison.png")

if __name__ == "__main__":
    main()
