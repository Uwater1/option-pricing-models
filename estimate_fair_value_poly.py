import argparse
import numpy as np
from datetime import datetime
import ctypes

# Load C++ Library
lib = ctypes.CDLL('./libbs.so')
lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.implied_volatility.restype = ctypes.c_double

def fast_iv(price, S, K, T, r, is_call):
    return lib.implied_volatility(price, S, K, T, r, is_call)

# Model Coefficients (Trained on SPY, QQQ, NVDA)
# Model: RelativeSpread = b0 + b1*M + b2*T + b3*IV + b4*M^2 + b5*T^2 + b6*T*IV
COEFFS = {
    'b0': 0.08435910157616279, 
    'b1': -0.14138231593843822, 
    'b2': 0.015237941763599355, 
    'b3': 0.016146873728418405, 
    'b4': 0.06674618886979494, 
    'b5': 0.0012808918801879127, 
    'b6': -0.04291380581474171
}

def predict_relative_spread(moneyness, time_to_expire_years, iv):
    """
    Predicts the relative spread (spread / mid_price) based on Moneyness, Time, and IV.
    """
    M = moneyness
    T = time_to_expire_years
    IV = iv
    
    # Polynomial Model
    rel_spread = (COEFFS['b0'] + 
                  COEFFS['b1']*M + 
                  COEFFS['b2']*T + 
                  COEFFS['b3']*IV + 
                  COEFFS['b4']*M**2 + 
                  COEFFS['b5']*T**2 + 
                  COEFFS['b6']*T*IV)
    
    # Clamp result
    return max(0.001, min(0.5, rel_spread))

def estimate_fair_value(last_price, strike, underlying_price, days_to_expire, option_type='call'):
    """
    Estimates Fair Bid and Ask based on Last Price and factors.
    """
    # 1. Calculate Factors
    if days_to_expire <= 0:
        print("Error: Option has expired or expires today.")
        return None
        
    time_years = days_to_expire / 365.0
    
    # Moneyness (S / K)
    moneyness = underlying_price / strike
    
    # 2. Calculate IV
    r = 0.05
    is_call = 1 if option_type.lower() == 'call' else 0
    iv = fast_iv(last_price, underlying_price, strike, time_years, r, is_call)
    
    if iv <= 0.001:
        print("Warning: Could not calculate valid IV (likely illiquid or arbitrage bounds). Using default 0.3")
        iv = 0.3 # Fallback
    
    # 3. Predict Relative Spread
    pred_rel_spread = predict_relative_spread(moneyness, time_years, iv)
    
    # 4. Calculate Spread Value
    predicted_spread = pred_rel_spread * last_price
    
    # 5. Derive Bid/Ask
    fair_bid = last_price - (predicted_spread / 2)
    fair_ask = last_price + (predicted_spread / 2)
    
    return {
        "strike": strike,
        "underlying": underlying_price,
        "moneyness": moneyness,
        "days_to_expire": days_to_expire,
        "time_years": time_years,
        "iv": iv,
        "predicted_rel_spread": pred_rel_spread,
        "predicted_spread_value": predicted_spread,
        "fair_bid": fair_bid,
        "fair_ask": fair_ask
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Fair Bid/Ask for Options")
    parser.add_argument("last_price", type=float, help="Option Last Traded Price")
    parser.add_argument("strike", type=float, help="Option Strike Price")
    parser.add_argument("underlying", type=float, help="Underlying Asset Price")
    parser.add_argument("days_to_expire", type=float, help="Days to Expiration (e.g. 30)")
    parser.add_argument("--type", type=str, default="call", choices=['call', 'put'], help="Option Type (call/put)")
    
    args = parser.parse_args()
    
    result = estimate_fair_value(args.last_price, args.strike, args.underlying, args.days_to_expire, args.type)
    
    if result:
        print("\n--- Fair Value Estimation ---")
        print(f"Input Last Price:  ${args.last_price:.2f}")
        print(f"Underlying Price:  ${result['underlying']:.2f}")
        print(f"Strike Price:      ${result['strike']:.2f}")
        print(f"Detail:            {args.type.upper()} Expiring in {result['days_to_expire']} days")
        print("-" * 30)
        print(f"Moneyness (S/K):   {result['moneyness']:.2f}")
        print(f"Implied Volatility:{result['iv']:.2%}")
        print("-" * 30)
        print(f"Predicted Rel Spread: {result['predicted_rel_spread']:.2%}")
        print(f"Predicted Spread ($): ${result['predicted_spread_value']:.2f}")
        print("-" * 30)
        print(f"Fair Bid:  ${result['fair_bid']:.2f}")
        print(f"Fair Ask:  ${result['fair_ask']:.2f}")
