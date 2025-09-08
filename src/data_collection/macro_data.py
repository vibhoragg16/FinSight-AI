# src/data_collection/macro_data.py

import yfinance as yf
import pandas as pd
import os
import logging
from src.utils.config import PROCESSED_DATA_PATH

# Define key macroeconomic indicators available through yfinance
MACRO_INDICATORS = {
    "10_YR_TREASURY": "^TNX",
    "VIX": "^VIX",
    "US_DOLLAR_INDEX": "DX-Y.NYB",
    "CRUDE_OIL": "CL=F",
    "GOLD": "GC=F"
}

def fetch_macro_data(period="5y"):
    """
    Fetches historical data for key macroeconomic indicators.
    """
    logging.info("Fetching macroeconomic data...")
    output_dir = os.path.join(PROCESSED_DATA_PATH, "macro")
    os.makedirs(output_dir, exist_ok=True)

    for name, ticker in MACRO_INDICATORS.items():
        try:
            data = yf.download(ticker, period=period)
            if data.empty:
                logging.warning(f"No data found for {name} ({ticker}).")
                continue

            data.reset_index(inplace=True)
            output_file = os.path.join(output_dir, f"{name}.csv")
            data.to_csv(output_file, index=False)
            logging.info(f"Successfully fetched and saved data for {name}.")
        except Exception as e:
            logging.error(f"Could not fetch data for {name} ({ticker}): {e}")

    logging.info("Macroeconomic data fetching complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetch_macro_data()
