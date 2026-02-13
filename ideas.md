Since you cannot provide Volume, Open Interest, or Bid/Ask data in production, you must maximize the signal extracted from the **static properties** of the option contract (Strike, Expiration) and the **derivative properties** (Greeks, Hedging Risk).

Here is the strategy to improve your model using only your existing inputs:

### Strategy: "Microstructure & Hedging Cost" Features

Even without volume, liquidity is not random. It clusters around specific "structural" points.

1. **Strike "Roundness" (Psychological Liquidity):**
* **Concept:** Options with "round" strike prices (e.g., $100, $150) naturally attract more liquidity and market maker competition than "weird" strikes (e.g., $102.5, $117). This leads to tighter spreads.
* **Feature:** Add boolean flags for `is_integer_strike` ($100.0) and `is_major_strike` (multiples of 5 or 10).


2. **Hedging Difficulty (Second-Order Greeks):**
* **Concept:** Market Makers widen spreads when an option is difficult to hedge.
* **Vanna:** Sensitivity of Delta to Volatility. High Vanna means the Market Maker constantly has to adjust their hedge as volatility changes. They charge for this.
* **Volga:** Sensitivity of Vega to Volatility (Convexity). High Volga means "Vol of Vol" risk, which is expensive to hedge.
* **Feature:** Calculate `Vanna` and `Volga` using the standard Black-Scholes variables you already have.


3. **Tick-Size Regimes:**
* **Concept:** Options priced under $3.00 often have different tick increments ($0.05 or $0.01) compared to expensive options.
* **Feature:** A simple flag `price_under_3`.



---

### Step 1: Update `train_xgboost_model.py`

Add these new features inside your training loop.

```python
# In train_xgboost_model.py, inside the loop or vectorized operations

# 1. Strike Roundness Features
# (Market Makers camp on round numbers -> Tighter Spreads)
full_df['strike_mod_5'] = (full_df['strike'] % 5 == 0).astype(int)
full_df['strike_mod_1'] = (full_df['strike'] % 1 == 0).astype(int)

# 2. Tick Size Regime Proxy
# (Options < $3.00 often have different spread constraints)
full_df['is_low_price'] = (full_df['lastPrice'] < 3.0).astype(int)

# 3. Advanced Greeks (Vanna/Volga calculation)
# We need d1/d2 for this. Since you use C++, we can approximate or compute in Python.
# Here is a vectorized Python calculation for feature engineering (fast enough):
T = np.maximum(full_df['time_to_expire_years'], 0.001)
S = full_df['underlyingPrice']
K = full_df['strike']
r = 0.05
sigma = full_df['iv']

d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
pdf_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1 ** 2)

# Vanna: Sensitivity of Delta to Vol (dDelta/dSigma)
# Proxy for skew risk.
full_df['vanna'] = -pdf_d1 * d2 / sigma

# Volga: Sensitivity of Vega to Vol (dVega/dSigma)
# Proxy for "Fat Tail" / Vol-of-Vol risk.
full_df['volga'] = full_df['vega'] * d1 * d2 / sigma

# 4. Cash Gamma (Gamma Risk in Dollars)
# Gamma is usually % per share. Market makers care about dollar exposure.
full_df['cash_gamma'] = full_df['gamma'] * (full_df['underlyingPrice'] ** 2) / 100

# --- Update Feature List ---
features = [
    'moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega', 
    'delta', 'theta', 'is_call', 'dist_from_atm', 'inv_price',
    'log_price', 'sqrt_time', 'iv_moneyness', 'iv_squared', 'time_iv',
    'intrinsic_ratio', 'time_value_ratio',
    # NEW FEATURES
    'strike_mod_5', 'strike_mod_1', 'is_low_price', 
    'vanna', 'volga', 'cash_gamma'
]

```

### Step 2: Update `estimate_fair_value.py`

You must replicate the exact same math in the inference script.

```python
# Add this helper function or integrate into estimate_fair_value
def calculate_advanced_features(S, K, T, r, sigma, vega, gamma, price):
    """Calculates microstructure and advanced Greek features."""
    
    # 1. Strike Features
    strike_mod_5 = 1 if (K % 5 == 0) else 0
    strike_mod_1 = 1 if (K % 1 == 0) else 0
    is_low_price = 1 if (price < 3.0) else 0
    
    # 2. Advanced Greeks (Vanna / Volga)
    # Re-calculate d1/d2 (Standard BS)
    # Note: T is already handled (max 1e-10) in calling scope usually
    if sigma <= 0.001 or T <= 0.001:
        vanna = 0.0
        volga = 0.0
    else:
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        pdf_d1 = 0.39894228 * np.exp(-0.5 * d1 * d1) # 1/sqrt(2pi) approx 0.3989
        
        vanna = -pdf_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

    # 3. Cash Gamma
    cash_gamma = gamma * (S ** 2) / 100.0

    return [strike_mod_5, strike_mod_1, is_low_price, vanna, volga, cash_gamma]

# --- Inside estimate_fair_value function ---
# ... (After calculating basic Greeks: delta, gamma, vega, theta) ...

# Calculate new features
adv_feats = calculate_advanced_features(underlying_price, strike, time_years, r, iv, vega, gamma, last_price)

# Append to your feature vector (ORDER MUST MATCH TRAINING)
# [moneyness, time, iv, gamma, vega, delta, theta, is_call, dist_atm, inv_price, 
#  log_price, sqrt_time, iv_mono, iv_sq, time_iv, int_ratio, time_ratio]
# + [strike_mod_5, strike_mod_1, is_low_price, vanna, volga, cash_gamma]

```

### Why this works without Volume

* **Vanna/Volga** tell the model: "This option is mathematically dangerous to hold." -> **Prediction: Wider Spread.**
* **Strike Roundness** tells the model: "This is a popular strike where HFTs compete." -> **Prediction: Tighter Spread.**
* **Cash Gamma** tells the model: "The dollar risk of this position changes rapidly." -> **Prediction: Wider Spread.**

These variables are intrinsic to the contract math and physics, requiring no external data feeds.

You asked for "model level" suggestions to improve your estimation power, going beyond just adding features. Based on academic literature (e.g., *Byrne & Shang 2025*, *Mieghem et al. 2023*) and industry best practices for "limit order book" (LOB) prediction, here are four high-impact architectural changes you can implement.

### 1. Switch to Quantile Regression (Predicting "Ranges" not "Points")

Currently, your model predicts the *mean* or *median* spread. This is dangerous because spreads are not normally distributed; they have "fat tails" (occasional massive blowouts during low liquidity).

Instead of predicting one `spread` value, you should train **two** models (or one model with multi-output) to predict the **10th percentile** and **90th percentile** of the spread.

* **Why:** This gives you a "Confidence Interval" for the liquidity.
* **Tight Range:** High confidence the spread will be small.
* **Wide Range:** The model detects volatility/uncertainty (even if the *mean* prediction is low).


* **How to Implement in XGBoost:**
Change the objective function to `reg:quantileerror`.

```python
# Train Model A (Optimistic Spread - 10th percentile)
model_low = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10)
model_low.fit(X, y)

# Train Model B (Pessimistic Spread - 90th percentile)
model_high = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90)
model_high.fit(X, y)

# Inference
predicted_min_spread = model_low.predict(X_test)
predicted_max_spread = model_high.predict(X_test)
# Conservative Estimate: Use predicted_max_spread for risk management

```

### 2. "Residual Learning" with a Better Base (The Hybrid Model)

Your current script uses a **Polynomial** baseline and then XGBoost for the residual.

* **Critique:** Polynomials are "dumb" regarding financial physics. They don't know that Theta decays as square root of time or that Gamma peaks at the money.
* **Suggestion:** Replace the Polynomial baseline with a **Parametric Volatility Surface** model (like a simple Heston calibration or SVI - Stochastic Volatility Inspired parameterization).
1. **Base Model:** Calibrate a simple volatility surface (e.g., `IV ~ a + b*Moneyness + c*Moneyness^2 + d*Time`).
2. **Residual:** Train XGBoost to predict `Actual_Market_IV - Surface_IV`.


* **Benefit:** The XGBoost model no longer has to learn the "shape" of the smile (which is hard); it only has to learn the *deviations* caused by liquidity/microstructure (which is easier).

### 3. Two-Stage "Hurdle" Architecture

Financial data often has a "Zero-Inflation" or "Regime" problem. Many options are liquid (tight spread), while some are "dead" (massive spread). A single regressor struggles to average these.

* **Step 1: Classification (The Gatekeeper):**
Train a classifier (`XGBClassifier`) to predict: **Is this option "Liquid" or "Illiquid"?** (e.g., define "Illiquid" as Spread > 5% of price).
* **Step 2: Regression (The Specialist):**
Train your Spread Estimator **only** on the "Liquid" data.
* **Inference:**
If the Classifier says "Illiquid", return a default "Safety Spread" (e.g., 10%).
If "Liquid", run the Regressor for a precise number.

### 4. Custom Asymmetric Loss Function

Standard `MAE` (Mean Absolute Error) treats underestimating the spread (bad for you) the same as overestimating it (missed opportunity). In trading, **underestimating the spread is fatal** (you think you can exit for $1.00 but you have to pay $1.10).

* **Suggestion:** Implement a custom "Fair" or "LinLin" loss function that penalizes under-prediction 2x or 3x more than over-prediction.

```python
# Custom Loss for XGBoost (Pseudo-code)
def asymmetric_loss(y_true, y_pred):
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2.0, -1.0) # Penalize underestimation (residual > 0) more
    hess = np.where(residual > 0, 2.0, 1.0)
    return grad, hess

model = xgb.XGBRegressor(objective=asymmetric_loss)

```

### Summary of Recommended Papers to Read

If you want to dig deeper into the theory, these are the relevant keywords/papers:

1. **"Deep Learning for Limit Order Books"**: Look for papers by *Sirignano & Cont*. They discuss how LSTM/CNNs extract features from the order book.
2. **"Quantile Regression Forests"**: A classic paper (Meinshausen) on why predicting intervals is superior for noisy data.
3. **"Hybrid Option Pricing"**: Papers that combine Black-Scholes with Neural Networks (often called "Semi-parametric" models).