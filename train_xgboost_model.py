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

# Latest Polynomial Coefficients for Stacking
POLY_COEFFS = {
    'b0': 0.084359, 'b1': -0.141382, 'b2': 0.015238, 
    'b3': 0.016147, 'b4': 0.067123, 'b5': 0.002284, 'b6': -0.039646
}

def get_poly_pred(M, T, IV):
    rel_spread = (POLY_COEFFS['b0'] + POLY_COEFFS['b1']*M + POLY_COEFFS['b2']*T + 
                  POLY_COEFFS['b3']*IV + POLY_COEFFS['b4']*M**2 + 
                  POLY_COEFFS['b5']*T**2 + POLY_COEFFS['b6']*T*IV)
    return rel_spread

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
    
    # Filter outliers
    full_df = full_df[
        (full_df['moneyness'] > 0.5) & (full_df['moneyness'] < 1.5) &
        (full_df['spread'] > 0) & (full_df['spread'] < 25.0) & # Absolute spread sanity check
        (full_df['mid'] > 0.05) 
    ].copy()

    print("Calculating Features (IV, Greeks — production-available only)...")
    
    if 'underlyingPrice' not in full_df.columns:
         full_df['underlyingPrice'] = full_df['strike'] * full_df['moneyness']

    r = 0.05
    
    ivs = []
    gammas = []
    vegas = []
    deltas = []
    thetas = []
    
    for _, row in full_df.iterrows():
        time_years = max(row['time_to_expire_years'], 0.001)
        price = row['lastPrice']
        S = row['underlyingPrice']
        K = row['strike']
        
        # IV — computed from lastPrice (production-available)
        iv = fast_iv(price, S, K, time_years, r, row['optionType'])
        
        # Clamp IV for stability
        if iv < 0.001: iv = 0.001
        if iv > 5.0: iv = 5.0
            
        # Greeks
        gamma = fast_gamma(S, K, time_years, r, iv)
        vega = fast_vega(S, K, time_years, r, iv)
        delta = fast_delta(S, K, time_years, r, iv, row['optionType'])
        theta = fast_theta(S, K, time_years, r, iv, row['optionType'])
        
        ivs.append(iv)
        gammas.append(gamma)
        vegas.append(vega)
        deltas.append(delta)
        thetas.append(theta)

    full_df['iv'] = ivs
    full_df['gamma'] = gammas
    full_df['vega'] = vegas
    full_df['delta'] = deltas
    full_df['theta'] = thetas
    
    # Calculate Polynomial Spread "Hint" for Stacking (uses lastPrice, not mid)
    full_df['poly_hint'] = [get_poly_pred(row['moneyness'], max(row['time_to_expire_years'], 0.001), iv) * row['lastPrice'] for (_, row), iv in zip(full_df.iterrows(), ivs)]
    
    # --- PRODUCTION-AVAILABLE features only ---
    # Basic features (derivable from the 5 production inputs)
    full_df['is_call'] = (full_df['optionType'] == 'call').astype(int)
    full_df['dist_from_atm'] = np.abs(full_df['moneyness'] - 1.0)
    full_df['inv_price'] = 1.0 / full_df['lastPrice'].clip(lower=0.01)
    
    # NEW engineered features to compensate for missing volume/OI/mid
    full_df['log_price'] = np.log(full_df['lastPrice'].clip(lower=0.01))
    full_df['sqrt_time'] = np.sqrt(full_df['time_to_expire_years'].clip(lower=0.001))
    full_df['iv_moneyness'] = full_df['iv'] * full_df['moneyness']
    full_df['iv_squared'] = full_df['iv'] ** 2
    full_df['time_iv'] = full_df['time_to_expire_years'] * full_df['iv']
    
    # Intrinsic value ratio: how deep ITM/OTM the option is
    full_df['intrinsic'] = np.where(
        full_df['is_call'] == 1,
        np.maximum(0, full_df['underlyingPrice'] - full_df['strike']),
        np.maximum(0, full_df['strike'] - full_df['underlyingPrice'])
    )
    full_df['intrinsic_ratio'] = full_df['intrinsic'] / full_df['lastPrice'].clip(lower=0.01)
    full_df['intrinsic_ratio'] = full_df['intrinsic_ratio'].clip(upper=10.0)
    
    # Time value ratio: fraction of price that is extrinsic/time value
    full_df['time_value'] = (full_df['lastPrice'] - full_df['intrinsic']).clip(lower=0)
    full_df['time_value_ratio'] = full_df['time_value'] / full_df['lastPrice'].clip(lower=0.01)
    
    # --- NEW MICROSTRUCTURE FEATURES (from ideas.md) ---
    # 1. Strike Roundness (Psychological Liquidity)
    full_df['strike_mod_5'] = (full_df['strike'] % 5 == 0).astype(int)
    full_df['strike_mod_1'] = (full_df['strike'] % 1 == 0).astype(int)

    # 2. Tick Size Regime Proxy (Options < $3.00 often have different tick constraints)
    full_df['is_low_price'] = (full_df['lastPrice'] < 3.0).astype(int)

    # 3. Advanced Greeks (Vanna / Volga / Cash Gamma)
    # Re-calculate d1/d2 using numpy for speed
    T_vec = np.maximum(full_df['time_to_expire_years'], 0.001)
    S_vec = full_df['underlyingPrice']
    K_vec = full_df['strike']
    sigma_vec = np.maximum(full_df['iv'], 0.001) # Avoid division by zero
    sqrt_T = np.sqrt(T_vec)
    
    d1 = (np.log(S_vec / K_vec) + (r + 0.5 * sigma_vec ** 2) * T_vec) / (sigma_vec * sqrt_T)
    d2 = d1 - sigma_vec * sqrt_T
    pdf_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 ** 2)
    
    # Vanna: dDelta/dSigma (Skew risk)
    full_df['vanna'] = -pdf_d1 * d2 / sigma_vec
    
    # Volga: dVega/dSigma (Vol-of-Vol risk)
    full_df['volga'] = full_df['vega'] * d1 * d2 / sigma_vec
    
    # Cash Gamma: Dollar risk exposure (Gamma * S^2 / 100)
    full_df['cash_gamma'] = full_df['gamma'] * (S_vec ** 2) / 100.0

    # Fill NaNs with 0 (safe default for regression) instead of dropping rows
    full_df = full_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Filter valid IV
    full_df = full_df[(full_df['iv'] > 0.01) & (full_df['iv'] < 5.0)].copy()
    
    print(f"Training on {len(full_df)} rows.")
    
    # PRODUCTION-ONLY features: everything derivable from
    # (lastPrice, strike, underlyingPrice, days_to_expire, optionType)
    features = [
        'moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega', 
        'delta', 'theta', 'is_call', 'dist_from_atm', 'inv_price',
        'log_price', 'sqrt_time', 'iv_moneyness', 'iv_squared', 'time_iv',
        'intrinsic_ratio', 'time_value_ratio',
        # NEW FEATURES
        'strike_mod_5', 'strike_mod_1', 'is_low_price', 
        'vanna', 'volga', 'cash_gamma'
    ]
    
    # Target: Residual (Actual Spread - Polynomial Baseline)
    full_df['residual'] = full_df['spread'] - full_df['poly_hint']
    target = 'residual'
    
    X = full_df[features]
    y = full_df[target]
    
    # Keep track of poly_hint for evaluation
    X_train, X_test, y_train, y_test, poly_train, poly_test = train_test_split(
        X, y, full_df['poly_hint'], test_size=0.2, random_state=42
    )
    
    # Custom Asymmetric SQUARED Loss Function
    # Avoids constant gradients of MAE which caused convergence failure (0 feature importance)
    # Still penalizes underestimation (y_true > y_pred) heavily
    def asymmetric_squared_loss(y_true, y_pred):
        residual = y_true - y_pred # Positive = Underestimation (Bad)
        
        # Penalty factor for underestimation
        alpha = 5.0 
        
        # Gradient = dL/dy_pred = -2 * residual * weight
        grad = -2.0 * residual * np.where(residual > 0, alpha, 1.0)
        
        # Hessian = d^2L/dy_pred^2 = 2 * weight
        hess = 2.0 * np.where(residual > 0, alpha, 1.0)
        
        return grad, hess

    # Train XGBoost with early stopping for better generalization
    model = xgb.XGBRegressor(
        objective=asymmetric_squared_loss, # Use stable squared loss
        n_estimators=1000,                 # More estimators for smoother convergence
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"Best iteration: {model.best_iteration}")
    
    # Evaluate Layered Model
    residual_pred = model.predict(X_test)
    final_pred = poly_test + residual_pred
    actual_spread = poly_test + y_test
    
    mae = mean_absolute_error(actual_spread, final_pred)
    mean_spread = actual_spread.mean()
    
    print(f"\n--- Production-Ready Model (v2 Microstructure Features) ---")
    print(f"Mean Absolute Error (Spread $): ${mae:.4f}")
    print(f"Mean Spread in Test Set:        ${mean_spread:.4f}")
    print(f"Error as % of Spread:           {mae/mean_spread:.2%}")
    
    # Baseline comparison
    poly_mae = mean_absolute_error(actual_spread, poly_test)
    print(f"Baseline Poly-only MAE:         ${poly_mae:.4f}")
    print(f"Improvement over Poly-only:    {(poly_mae - mae)/poly_mae:.2%}")
    
    # Feature Importance
    print("\nFeature Importance:")
    importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"  {name:25s}: {imp:.4f} {bar}")
        
    # Save Model
    model.save_model("spread_xgb.json")
    print("\nModel saved to spread_xgb.json")

if __name__ == "__main__":
    train_xgboost()
