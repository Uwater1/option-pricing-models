import numpy as np
import matplotlib.pyplot as plt
from .base import OptionPricingModel, OPTION_TYPE

def monte_carlo_price(S, K, T, r, sigma, option_type='call', N=10000):
    """
    Calculate option price using Monte Carlo simulation (European option).
    Only simulates the final price distribution, which is sufficient for European options.
    """
    # Ensure T is not zero to avoid division/multiplication errors
    T = np.maximum(T, 1e-10)

    # Vectorized simulation of end prices
    # ST = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    Z = np.random.standard_normal(N)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type.lower() == 'call' or option_type == OPTION_TYPE.CALL_OPTION.value:
        payoff = np.maximum(ST - K, 0)
    elif option_type.lower() == 'put' or option_type == OPTION_TYPE.PUT_OPTION.value:
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    price = np.exp(-r * T) * np.mean(payoff)
    return price

class MonteCarloPricing(OptionPricingModel):
    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations):
        self.S = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365.0
        self.r = risk_free_rate
        self.sigma = sigma
        self.N = number_of_simulations
        self.steps = int(days_to_maturity)
        if self.steps == 0: self.steps = 1
        self.simulation_results_S = None

    def simulate_prices(self):
        """
        Simulate full price paths for visualization.
        """
        dt = self.T / self.steps

        # Vectorized path generation
        Z = np.random.standard_normal((self.steps, self.N))
        W = np.cumsum(Z * np.sqrt(dt), axis=0)

        time_steps = np.linspace(dt, self.T, self.steps).reshape(-1, 1)

        drift = (self.r - 0.5 * self.sigma**2) * time_steps
        diffusion = self.sigma * W

        paths = self.S * np.exp(drift + diffusion)
        self.simulation_results_S = np.vstack([np.full((1, self.N), self.S), paths])

    def _calculate_call_option_price(self):
        # We can use the fast function if simulation hasn't run,
        # but if the user wants consistent results with the plot, we should use stored paths.
        # However, the fast function uses different random numbers.
        # If simulation_results_S exists, use it.
        if self.simulation_results_S is not None:
            ST = self.simulation_results_S[-1]
            payoff = np.maximum(ST - self.K, 0)
            return np.exp(-self.r * self.T) * np.mean(payoff)
        else:
            return monte_carlo_price(self.S, self.K, self.T, self.r, self.sigma, 'call', self.N)

    def _calculate_put_option_price(self):
        if self.simulation_results_S is not None:
            ST = self.simulation_results_S[-1]
            payoff = np.maximum(self.K - ST, 0)
            return np.exp(-self.r * self.T) * np.mean(payoff)
        else:
            return monte_carlo_price(self.S, self.K, self.T, self.r, self.sigma, 'put', self.N)

    def plot_simulation_results(self, num_of_movements):
        if self.simulation_results_S is None:
            self.simulate_prices()

        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results_S[:, :num_of_movements])
        plt.axhline(self.K, c='k', label='Strike Price')
        plt.xlim([0, self.steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Steps')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        plt.show()
