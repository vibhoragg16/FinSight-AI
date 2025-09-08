# src/ai_core/macro_brain.py

import pandas as pd
import os
import logging
from src.utils.config import PROCESSED_DATA_PATH

class MacroBrain:
    """
    Handles analysis related to macroeconomic trends and their impact on stocks.
    """
    def __init__(self):
        self.macro_data = self._load_macro_data()

    def _load_macro_data(self):
        """Loads all available macroeconomic data into a dictionary of DataFrames."""
        macro_dir = os.path.join(PROCESSED_DATA_PATH, "macro")
        data = {}
        if not os.path.exists(macro_dir):
            logging.warning("Macroeconomic data directory not found.")
            return data

        for file in os.listdir(macro_dir):
            if file.endswith(".csv"):
                try:
                    name = os.path.splitext(file)[0]
                    df = pd.read_csv(os.path.join(macro_dir, file))
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    data[name] = df
                except Exception as e:
                    logging.error(f"Failed to load macro data from {file}: {e}")
        return data

    def get_correlation_analysis(self, stock_data: pd.DataFrame):
        """
        Analyzes the correlation between a stock's performance and key macro indicators.
        """
        if stock_data.empty or not self.macro_data:
            return pd.DataFrame()

        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        stock_returns = stock_data['Close'].pct_change().rename("Stock_Returns")

        correlations = {}
        for name, df in self.macro_data.items():
            indicator_returns = df['Close'].pct_change().rename(name)
            combined = pd.concat([stock_returns, indicator_returns], axis=1).dropna()
            correlation = combined.corr().iloc[0, 1]
            correlations[name] = correlation

        corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
        return corr_df
