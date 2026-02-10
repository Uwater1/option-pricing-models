import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from option_pricing import black_scholes_price, monte_carlo_price, binomial_price

def verify_models(ticker_symbol='SPY'):
    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    # 1. Get Spot Price
    history = ticker.history(period="1y") # 1 year history for vol calculation
    if history.empty:
        print(f"Could not fetch history for {ticker_symbol}")
        return

    spot_price = history['Close'].iloc[-1]
    print(f"Spot Price: {spot_price:.2f}")

    # 2. Calculate Historical Volatility
    returns = history['Close'].pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)
    print(f"Historical Volatility (1y): {sigma:.2%}")

    # 3. Get Option Chain
    expirations = ticker.options
    if not expirations:
        print("No options found.")
        return

    # Pick an expiration ~1 month away
    target_date = datetime.now() + timedelta(days=30)
    best_expiry = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
    print(f"Selected Expiry: {best_expiry}")

    # Get chain
    chain = ticker.option_chain(best_expiry)
    calls = chain.calls
    puts = chain.puts

    # Pick ATM strike
    calls['abs_diff'] = abs(calls['strike'] - spot_price)
    atm_call = calls.loc[calls['abs_diff'].idxmin()]

    puts['abs_diff'] = abs(puts['strike'] - spot_price)
    atm_put = puts.loc[puts['abs_diff'].idxmin()]

    # Parameters
    S = spot_price
    K_call = atm_call['strike']
    K_put = atm_put['strike']

    T = (datetime.strptime(best_expiry, '%Y-%m-%d') - datetime.now()).days / 365.0
    r = 0.045 # Assumption: 4.5% risk free rate

    print(f"\n--- Verifying CALL Option (Strike: {K_call}) ---")
    market_price = atm_call['lastPrice']
    print(f"Market Price: {market_price:.2f}")

    bs_price = black_scholes_price(S, K_call, T, r, sigma, 'call')
    mc_price = monte_carlo_price(S, K_call, T, r, sigma, 'call')
    bin_price = binomial_price(S, K_call, T, r, sigma, 'call')

    print(f"Black-Scholes: {bs_price:.2f}")
    print(f"Monte Carlo:   {mc_price:.2f}")
    print(f"Binomial:      {bin_price:.2f}")

    print(f"\n--- Verifying PUT Option (Strike: {K_put}) ---")
    market_price_put = atm_put['lastPrice']
    print(f"Market Price: {market_price_put:.2f}")

    bs_price_put = black_scholes_price(S, K_put, T, r, sigma, 'put')
    mc_price_put = monte_carlo_price(S, K_put, T, r, sigma, 'put')
    bin_price_put = binomial_price(S, K_put, T, r, sigma, 'put')

    print(f"Black-Scholes: {bs_price_put:.2f}")
    print(f"Monte Carlo:   {mc_price_put:.2f}")
    print(f"Binomial:      {bin_price_put:.2f}")

    print("\nNote: Differences are expected as Historical Volatility != Implied Volatility used by market.")

if __name__ == "__main__":
    verify_models()
