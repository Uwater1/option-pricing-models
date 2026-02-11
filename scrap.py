import sys
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
DRIVER_PATH = "/usr/bin/chromedriver"
CHROME_PATH = "/usr/bin/chromium"

def scrape_option_chain(ticker, expiration_date):
    output_file = f"{ticker}-{expiration_date}.csv"
    
    # 1. Setup Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.binary_location = CHROME_PATH
    
    # Fake a user agent to avoid basic bot detection
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    service = Service(DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 2. Go to the URL
        url = f"https://optioncharts.io/options/{ticker}/option-chain?option_type=all&expiration_dates={expiration_date}:w&view=straddle&strike_range=all"
        print(f"Loading {url}...")
        driver.get(url)

        # 3. Wait for the page to load the specific element
        wait = WebDriverWait(driver, 15)
        
        print("Waiting for form data to render...")
        
        # We wait for the table body to appear, confirming data is loaded
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))

        # --- PART A: SCRAPE THE "FORM" (The Filters) ---
        print("\n--- Form / Filter Options ---")
        
        selects = driver.find_elements(By.TAG_NAME, "select")
        for select in selects:
            name = select.get_attribute("name") or select.get_attribute("id") or "Unknown Dropdown"
            print(f"\nFound Dropdown: {name}")
            
            opts = select.find_elements(By.TAG_NAME, "option")
            for opt in opts[:5]: 
                print(f"  - Value: {opt.get_attribute('value')} | Text: {opt.text}")

        # --- PART B: SCRAPE THE DATA (The Table) ---
        print("\n--- Option Chain Data ---")
        
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-sm.table-hover")))
        
        rows = driver.find_elements(By.CSS_SELECTOR, "table.table.table-sm.table-hover tbody tr.border:not(.header-row)")
        
        all_data = []
        for i, row in enumerate(rows):
            cols = row.find_elements(By.TAG_NAME, "td")
            if not cols: continue
            
            row_data = [col.text.strip() for col in cols]
            all_data.append(row_data)
            if i < 5:
                print(f"Row {i+1}: {row_data}")

        if all_data:
            columns = [
                'Call_Last', 'Call_Bid', 'Call_Ask', 'Call_Vol', 'Call_OI', 
                'Strike', 
                'Put_Last', 'Put_Bid', 'Put_Ask', 'Put_Vol', 'Put_OI'
            ]
            
            num_cols = len(all_data[0])
            if num_cols == len(columns):
                df = pd.DataFrame(all_data, columns=columns)
            else:
                df = pd.DataFrame(all_data)
                print(f"Warning: Unexpected column count ({num_cols}). Saving without headers.")

            df.to_csv(output_file, index=False)
            print(f"\nSuccessfully saved {len(all_data)} rows to {output_file}")
        else:
            print("No data found in the table.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scrap.py <TICKER> <EXPIRATION_DATE>")
        print("Example: python scrap.py NVDA 2026-02-11")
        sys.exit(1)
    
    input_ticker = sys.argv[1].upper()
    input_date = sys.argv[2]
    
    scrape_option_chain(input_ticker, input_date)