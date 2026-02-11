import numpy as np
from scipy.stats import norm
from .base import OptionPricingModel, OPTION_TYPE
from scipy.optimize import brentq

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Calculate Implied Volatility using Brent's method.
    """
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price
    
    # Check arbitrage bounds first (basic check, could be more robust)
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if price <= intrinsic + 1e-5: # Slightly relaxed for float issues
        return 0.0 # Or error, but 0.0 effectively means flat price
        
    try:
        # Search for vol between 0.1% and 500%
        return brentq(objective, 0.001, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan # Or 0, or handle error. NaN is safer for data cleaning.
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.

    Parameters:
    S : float or np.array
        Spot price
    K : float or np.array
        Strike price
    T : float or np.array
        Time to maturity in years
    r : float or np.array
        Risk-free interest rate
    sigma : float or np.array
        Volatility
    option_type : str
        'call' or 'put'

    Returns:
    price : float or np.array
    """
    # Handle zero time to maturity
    # If T is 0, option value is max(S-K, 0) for call, max(K-S, 0) for put.
    # We use a small epsilon or handle it explicitly.
    # Handling explicitly is safer but complex with arrays.
    # Using small epsilon is easier.
    T = np.maximum(T, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call' or option_type == OPTION_TYPE.CALL_OPTION.value:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put' or option_type == OPTION_TYPE.PUT_OPTION.value:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    return price

class BlackScholesModel(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365.0 # Convert to years
        self.r = risk_free_rate
        self.sigma = sigma

    def _calculate_call_option_price(self):
        return black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, 'call')

    def _calculate_put_option_price(self):
        return black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, 'put')
