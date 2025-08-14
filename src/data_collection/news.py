import requests
import json
import os
import logging
from src.utils.config import NEWS_API_KEY, NEWS_DATA_PATH, TARGET_COMPANIES

def fetch_company_news(tickers):
    """
    Fetches recent news articles for a list of companies using the NewsAPI.
    """
    if not NEWS_API_KEY:
        logging.error("NEWS_API_KEY is not configured. Cannot fetch news.")
        return

    logging.info("Fetching company news...")
    os.makedirs(NEWS_DATA_PATH, exist_ok=True)

    for ticker in tickers:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=50&language=en"
        try:
            response = requests.get(url)
            response.raise_for_status()
            news_data = response.json()

            output_file = os.path.join(NEWS_DATA_PATH, f'{ticker}_news.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, indent=4)
            logging.info(f"Successfully fetched and saved {len(news_data.get('articles', []))} articles for {ticker}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not fetch news for {ticker}: {e}")
