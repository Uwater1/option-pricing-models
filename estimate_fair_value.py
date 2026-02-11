import argparse
import numpy as np
import os
import ctypes
import sys

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed. Spread prediction might be less accurate (falling back to polynomial model).")

# Load C++ Library
lib_path = os.path.join(os.path.dirname(__file__), 'libbs.so')
try:
    lib = ctypes.CDLL(lib_path)
    lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.implied_volatility.restype = ctypes.c_double
    lib.black_scholes_vega.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.black_scholes_vega.restype = ctypes.c_double
    lib.black_scholes_gamma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.black_scholes_gamma.restype = ctypes.c_double
except OSError:
    print("Error: Could not load C++ library libbs.so. Ensure it is compiled and in the same directory.")
    sys.exit(1)

def fast_iv(price, S, K, T, r, is_call):
    return lib.implied_volatility(price, S, K, T, r, is_call)

def fast_vega(S, K, T, r, sigma):
    return lib.black_scholes_vega(S, K, T, r, sigma)

def fast_gamma(S, K, T, r, sigma):
    return lib.black_scholes_gamma(S, K, T, r, sigma)

# Polynomial Coefficients (Fallback)
POLY_COEFFS = {
    'b0': 0.07217830754411238, 
    'b1': -0.10741680766063832, 
    'b2': 0.024817628008708924, 
    'b3': 0.01673389789242793, 
    'b4': 0.04570831072599171, 
    'b5': -0.0031195961528276395, 
    'b6': -0.04091022400898136
}

def predict_spread_poly(moneyness, time_to_expire_years, iv, last_price):
    M, T, IV = moneyness, time_to_expire_years, iv
    rel_spread = (POLY_COEFFS['b0'] + 
                  POLY_COEFFS['b1']*M + 
                  POLY_COEFFS['b2']*T + 
                  POLY_COEFFS['b3']*IV + 
                  POLY_COEFFS['b4']*M**2 + 
                  POLY_COEFFS['b5']*T**2 + 
                  POLY_COEFFS['b6']*T*IV)
    return max(0.001, min(0.5, rel_spread)) * last_price

def predict_spread_xgb(moneyness, time_to_expire_years, iv, gamma, vega, last_price, model_path="spread_xgb.json"):
    if not XGB_AVAILABLE or not os.path.exists(model_path):
        return None
    
    # Features: ['moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega', 'inv_price', 'mid']
    inv_price = 1.0 / last_price if last_price > 0 else 0.0
    
    # Create DMatrix for single prediction
    # Note: feature names must match training
    import pandas as pd
    features = pd.DataFrame([{
        'moneyness': moneyness,
        'time_to_expire_years': time_to_expire_years,
        'iv': iv,
        'gamma': gamma,
        'vega': vega,
        'inv_price': inv_price,
        'mid': last_price
    }])
    
    model = xgb.Booster()
    model.load_model(model_path)
    dtest = xgb.DMatrix(features)
    
    pred_spread = model.predict(dtest)[0]
    return max(0.01, pred_spread) # Minimum spread 1 cent

def estimate_fair_value(last_price, strike, underlying_price, days_to_expire, option_type='call'):
    # 1. Calculate Factors
    if days_to_expire <= 0:
        return None
        
    time_years = days_to_expire / 365.0
    moneyness = underlying_price / strike
    
    # 2. Calculate IV and Greeks
    r = 0.05
    is_call = 1 if option_type.lower() == 'call' else 0
    iv = fast_iv(last_price, underlying_price, strike, time_years, r, is_call)
    
    if iv <= 0.001: iv = 0.3 # Fallback if IV fails
    if iv > 5.0: iv = 5.0    # Clamp
        
    gamma = fast_gamma(underlying_price, strike, time_years, r, iv)
    vega = fast_vega(underlying_price, strike, time_years, r, iv)
    
    # 3. Predict Spread
    # Try XGBoost first
    predicted_spread = predict_spread_xgb(moneyness, time_years, iv, gamma, vega, last_price)
    method = "XGBoost (Advanced)"
    
    if predicted_spread is None:
        # Fallback to Polynomial
        predicted_spread = predict_spread_poly(moneyness, time_years, iv, last_price)
        method = "Polynomial (Basic)"
    
    # 4. Derive Bid/Ask
    fair_bid = last_price - (predicted_spread / 2)
    fair_ask = last_price + (predicted_spread / 2)
    
    return {
        "strike": strike,
        "underlying": underlying_price,
        "moneyness": moneyness,
        "days": days_to_expire,
        "iv": iv,
        "gamma": gamma,
        "vega": vega,
        "spread": predicted_spread,
        "bid": fair_bid,
        "ask": fair_ask,
        "method": method
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Fair Bid/Ask using Advanced Machine Learning")
    parser.add_argument("last_price", type=float, help="Option Last Traded Price")
    parser.add_argument("strike", type=float, help="Option Strike Price")
    parser.add_argument("underlying", type=float, help="Underlying Asset Price")
    parser.add_argument("days_to_expire", type=float, help="Days to Expiration")
    parser.add_argument("--type", type=str, default="call", choices=['call', 'put'], help="Option Type")
    
    args = parser.parse_args()
    
    res = estimate_fair_value(args.last_price, args.strike, args.underlying, args.days_to_expire, args.type)
    
    if res:
        print("\n--- Advanced Fair Value Estimation ---")
        print(f"Option:            {args.type.upper()} Strike ${args.strike:.2f}")
        print(f"Underlying:        ${res['underlying']:.2f} (Moneyness: {res['moneyness']:.2f})")
        print(f"Time:              {res['days']} days")
        print("-" * 30)
        print(f"Implied Vol (IV):  {res['iv']:.2%}")
        print(f"Gamma:             {res['gamma']:.4f}")
        print(f"Vega:              {res['vega']:.4f}")
        print("-" * 30)
        print(f"Method Used:       {res['method']}")
        print(f"Predicted Spread:  ${res['spread']:.2f}")
        print("-" * 30)
        print(f"Fair Bid:          ${res['bid']:.2f}")
        print(f"Fair Ask:          ${res['ask']:.2f}")
