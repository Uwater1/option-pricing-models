# Option Pricing Models

## Introduction
This repository contains tools for calculating European option prices using three different methods:

1. Black-Scholes model
2. Monte Carlo simulation
3. Binomial model

It consists of two main parts:
1. **Python Library (`option_pricing/`)**: A flexible and efficient Python package for option pricing, designed to be easily plugged into backtesting systems.
2. **Web Simulation (`docs/`)**: A browser-based simulation for visualizing option pricing (hosted on GitHub Pages).

## Option Pricing Methods

### 1. Black-Scholes Model
A mathematical model used to calculate the theoretical price of European-style options.

### 2. Monte Carlo Simulation
A probabilistic method that uses random sampling to estimate option prices. The Python implementation is vectorized for high performance.

### 3. Binomial Model
A discrete-time model (Binomial Tree) for calculating option prices.

## Project Structure

- `option_pricing/`: Python package containing model implementations.
  - `black_scholes.py`: Black-Scholes model (class and function).
  - `monte_carlo.py`: Monte Carlo simulation (class and vectorized function).
  - `binomial.py`: Binomial Tree model (class and function).
- `docs/`: Web-based simulation (HTML/JS/CSS) for GitHub Pages.
- `verify_real_data.py`: Script to verify models using real-world data from Yahoo Finance.
- `requirements.txt`: Python dependencies.

## Usage

### Python Library

Install dependencies:
```bash
pip install -r requirements.txt
```

Example usage:
```python
from option_pricing import black_scholes_price, monte_carlo_price

S = 100  # Spot Price
K = 100  # Strike Price
T = 1.0  # Time to Maturity (Years)
r = 0.05 # Risk-free Rate
sigma = 0.2 # Volatility

price_bs = black_scholes_price(S, K, T, r, sigma, 'call')
print(f"Black-Scholes Price: {price_bs}")

price_mc = monte_carlo_price(S, K, T, r, sigma, 'call')
print(f"Monte Carlo Price: {price_mc}")
```

### Verification with Real Data

You can verify the models against real market data (fetched via `yfinance`) by running:
```bash
python verify_real_data.py
```

### Bid-Ask Spread Prediction

To estimate the fair Bid and Ask prices for an option based on its Last Price, Strike, and Underlying Price, use `estimate_fair_value.py`. This uses a polynomial model trained on market data to predict the spread.

```bash
python estimate_fair_value.py <LAST_PRICE> <STRIKE> <UNDERLYING> <Calendar_DAYS_TO_EXPIRE> [optional: --type call/put]
```

**Note on Days to Expiration:**
For option pricing models (Black-Scholes), it is standard to use **Calendar Days** (365 days/year) because interest accumulates daily. While trading days (252) are often used for volatility, using calendar days for time-to-decay is the convention. Please input the number of calendar days until expiration (e.g., 30 for one month).

**Example:**
```bash
python estimate_fair_value.py 10.50 200 205 30 --type call
```

### Web Simulation

The web-based simulation is automatically deployed to GitHub Pages. You can access it here:
**[Option Pricing Simulation](https://hallop.github.io/option-pricing-models/)**

*(Note: Replace `hallop` with your GitHub username if different.)*

To run locally, simply open `docs/index.html` in your web browser.
