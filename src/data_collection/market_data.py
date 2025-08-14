import yfinance as yf
import os
import logging
import pandas as pd
import ta
from src.utils.config import MARKET_DATA_PATH, TARGET_COMPANIES

def _add_technical_indicators(df):
    """Adds a comprehensive set of technical indicators to the dataframe."""
    if df.empty:
        return df
    
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    
    df['Volatility_30'] = df['Close'].rolling(window=30).std() * (252**0.5)

    df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    return df.fillna(0)

def fetch_market_data(tickers, period="5y"):
    """
    Fetches historical market data and adds technical indicators.
    """
    logging.info(f"Fetching market data for the last {period}...")
    os.makedirs(MARKET_DATA_PATH, exist_ok=True)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)

            if hist_data.empty:
                logging.warning(f"No market data found for {ticker}.")
                continue

            hist_data_with_ta = _add_technical_indicators(hist_data.copy())
            hist_data_with_ta.reset_index(inplace=True)
            
            output_file = os.path.join(MARKET_DATA_PATH, f'{ticker}_market_data.csv')
            hist_data_with_ta.to_csv(output_file, index=False)
            logging.info(f"Successfully fetched and enhanced market data for {ticker}")
        except Exception as e:
            logging.error(f"Could not fetch market data for {ticker}: {e}")
            
    logging.info("Market data fetching complete.")