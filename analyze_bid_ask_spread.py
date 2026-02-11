import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def analyze_spread(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # 1. Data Cleaning and Preparation
    initial_count = len(df)
    
    # Ensure required columns exist
    required_cols = ['bid', 'ask', 'volume', 'strike', 'expirationDate', 'days_to_expire', 'moneyness']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    # Filter out invalid bid/ask
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]
    
    # Calculate Spread
    df['spread'] = df['ask'] - df['bid']
    df['mid_price'] = (df['ask'] + df['bid']) / 2
    df['relative_spread'] = df['spread'] / df['mid_price']
    
    # remove outliers or bad data (e.g. negative spread)
    df = df[df['spread'] >= 0]
    
    print(f"Data cleaned: {len(df)} rows (from {initial_count})")
    
    # 2. Analysis
    
    # Summary Statistics
    print("\n--- Summary Statistics ---")
    print(df[['spread', 'relative_spread', 'volume', 'moneyness', 'days_to_expire']].describe())
    
    # Correlation Matrix
    corr = df[['spread', 'relative_spread', 'volume', 'moneyness', 'days_to_expire']].corr()
    print("\n--- Correlation Matrix ---")
    print(corr)

    # 3. Visualization
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set figure size
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Relative Spread vs Volume (Log Scale)
    plt.subplot(2, 2, 1)
    # Use log scale for volume as it spans orders of magnitude
    plt.scatter(df['volume'], df['relative_spread'], alpha=0.3, s=10)
    plt.xscale('log')
    plt.xlabel('Volume (Log Scale)')
    plt.ylabel('Relative Bid-Ask Spread')
    plt.title('Spread vs Volume')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 2: Relative Spread vs Moneyness
    plt.subplot(2, 2, 2)
    plt.scatter(df['moneyness'], df['relative_spread'], alpha=0.3, s=10, c='orange')
    plt.xlabel('Moneyness (Spot / Strike)')
    plt.ylabel('Relative Bid-Ask Spread')
    plt.title('Spread vs Moneyness')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # Highlight ATM (around 1.0)
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='ATM')
    plt.legend()

    # Plot 3: Relative Spread vs Time to Expiration
    plt.subplot(2, 2, 3)
    plt.scatter(df['days_to_expire'], df['relative_spread'], alpha=0.3, s=10, c='green')
    plt.xlabel('Days to Expiration')
    plt.ylabel('Relative Bid-Ask Spread')
    plt.title('Spread vs Time to Expiration')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 4: Average Relative Spread by Moneyness Bins
    plt.subplot(2, 2, 4)
    # Bin Moneyness
    bins = np.linspace(0.5, 1.5, 21) # 0.5 to 1.5 in steps of 0.05
    df['moneyness_bin'] = pd.cut(df['moneyness'], bins=bins)
    avg_spread_by_moneyness = df.groupby('moneyness_bin')['relative_spread'].mean()
    avg_spread_by_moneyness.plot(kind='bar', color='purple', alpha=0.7)
    plt.xlabel('Moneyness Bin')
    plt.ylabel('Avg Relative Spread')
    plt.title('Avg Spread by Moneyness')
    plt.grid(axis='y', alpha=0.2)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "spread_analysis.png")
    plt.savefig(plot_file)
    print(f"\nSaved analysis plot to {plot_file}")
    # plt.show() # Cannot show GUI in this environment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_bid_ask_spread.py <CSV_FILE>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_spread(file_path)
