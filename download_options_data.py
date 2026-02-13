import yfinance as yf
import pandas as pd
import sys
import os
from datetime import datetime

def download_options_data(ticker_symbol, output_dir="data"):
    """
    Downloads option chain for all expiration dates for a given ticker.
    Calculates Days to Expiration and Moneyness.
    Saves the result to a CSV file.
    """
    print(f"Fetching data for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return

    if not expirations:
        print(f"No option expirations found for {ticker_symbol}.")
        return

    # Get current stock price for moneyness calculation
    try:
        # Try fast_info first
        current_price = ticker.fast_info.get('last_price', None)
        if current_price is None:
             # Fallback to history
             hist = ticker.history(period="1d")
             if not hist.empty:
                 current_price = hist['Close'].iloc[-1]
    except Exception:
         # Fallback to history if fast_info fails
         hist = ticker.history(period="1d")
         if not hist.empty:
             current_price = hist['Close'].iloc[-1]
         else:
             print("Could not retrieve current stock price. Skipping moneyness calculation.")
             current_price = None

    print(f"Current price for {ticker_symbol}: {current_price}")

    all_options = []
    today = datetime.now()

    for expiry in expirations:
        print(f"Processing expiration: {expiry}")
        try:
            opt = ticker.option_chain(expiry)
            calls = opt.calls
            puts = opt.puts
            
            # Add metadata
            calls['optionType'] = 'call'
            puts['optionType'] = 'put'
            
            df = pd.concat([calls, puts])
            df['expirationDate'] = expiry
            
            # Calculate Time to Expiration (in years, or days)
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            days_to_expire = (expiry_date - today).days
            # If expired today or passed, set to 0 or negative
            df['days_to_expire'] = days_to_expire
            df['time_to_expire_years'] = days_to_expire / 365.0

            # Calculate Moneyness
            # For Calls: S / K (or K / S, different conventions exist. Standard is often S/K for calls)
            # Let's use simple moneyness: S / K
            if current_price:
                df['underlyingPrice'] = current_price
                df['moneyness'] = current_price / df['strike']
            
            all_options.append(df)
            
        except Exception as e:
            print(f"Failed to fetch data for expiration {expiry}: {e}")

    if all_options:
        final_df = pd.concat(all_options, ignore_index=True)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{ticker_symbol}_options_4.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(final_df)} rows to {output_file}")
    else:
        print("No option data collected.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_options_data.py <TICKER_SYMBOL>")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    download_options_data(ticker)
