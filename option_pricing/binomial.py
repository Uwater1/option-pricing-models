import numpy as np
from .base import OptionPricingModel, OPTION_TYPE

def binomial_price(S, K, T, r, sigma, option_type='call', steps=1000):
    """
    Calculate option price using Binomial Tree model.
    """
    T = np.maximum(T, 1e-10)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral probability
    a = np.exp(r * dt)
    p = (a - d) / (u - d)
    q = 1.0 - p

    # Initialize asset prices at maturity (steps)
    # S_T = S * u^j * d^(N-j)
    j = np.arange(steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))

    # Initialize option values at maturity
    if option_type.lower() == 'call' or option_type == OPTION_TYPE.CALL_OPTION.value:
        V = np.maximum(ST - K, 0)
    elif option_type.lower() == 'put' or option_type == OPTION_TYPE.PUT_OPTION.value:
        V = np.maximum(K - ST, 0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    # Backward induction
    discount = np.exp(-r * dt)

    for i in range(steps):
        # Calculate V at step (steps - 1 - i)
        # V has size (steps + 1 - i)
        # New V will have size (steps - i)
        V = discount * (p * V[1:] + q * V[:-1])

    return V[0]

class BinomialTreeModel(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365.0
        self.r = risk_free_rate
        self.sigma = sigma
        self.steps = number_of_time_steps

    def _calculate_call_option_price(self):
        return binomial_price(self.S, self.K, self.T, self.r, self.sigma, 'call', self.steps)

    def _calculate_put_option_price(self):
        return binomial_price(self.S, self.K, self.T, self.r, self.sigma, 'put', self.steps)
