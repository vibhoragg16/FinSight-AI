import os
import logging
from sec_edgar_downloader import Downloader
from src.utils.config import SEC_API_USER_AGENT, RAW_DATA_PATH, TARGET_COMPANIES

def download_sec_filings(tickers, filing_types=['10-K', '10-Q'], limit=5):
    """
    Downloads the latest SEC filings for a list of companies.
    """
    if not SEC_API_USER_AGENT or "YourName" in SEC_API_USER_AGENT:
        logging.error("SEC_API_USER_AGENT is not configured. Please set it in config.py or .env file.")
        return

    logging.info("Starting SEC filing download...")
    dl_path = os.path.join(RAW_DATA_PATH)
    os.makedirs(dl_path, exist_ok=True)
    
    dl = Downloader("MyCompanyName", SEC_API_USER_AGENT, dl_path)

    for ticker in tickers:
        try:
            for filing_type in filing_types:
                logging.info(f"Downloading {limit} most recent {filing_type} filings for {ticker}...")
                dl.get(filing_type, ticker, limit=limit, download_details=True)
            logging.info(f"Successfully downloaded filings for {ticker}")
        except Exception as e:
            logging.error(f"Could not download filings for {ticker}: {e}")
            
    logging.info("SEC filing download complete.")