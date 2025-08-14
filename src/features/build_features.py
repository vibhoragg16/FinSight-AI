# File: src/features/build_features.py

import pandas as pd
import numpy as np

def prepare_features_for_prediction(market_df: pd.DataFrame, ratios_df: pd.DataFrame, news_sentiment: float) -> pd.DataFrame:
    """
    Prepares a combined feature set for the stock prediction model.
    It merges market data with financial ratios and engineers technical indicators.
    """
    if market_df.empty:
        return pd.DataFrame()

    # --- 1. Prepare and Merge DataFrames ---
    df = market_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
    df.set_index('Date', inplace=True)

    if ratios_df is not None and not ratios_df.empty:
        ratios_df['Date'] = pd.to_datetime(ratios_df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
        ratios_df.set_index('Date', inplace=True)
        df = pd.merge_asof(df.sort_index(), ratios_df.sort_index(), on='Date', direction='backward')

    # --- 2. Calculate Technical Indicators ---
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility_30'] = df['Close'].rolling(window=30).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- 3. Incorporate Other Features ---
    df['news_sentiment'] = news_sentiment

    # --- 4. Define the Target Variable ---
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

    # --- 5. Final Assembly and Cleanup ---
    final_feature_cols = [
        'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Volatility_30', 'Volume',
        'CurrentRatio', 'DebtToEquity', 'ReturnOnEquity', 'NetProfitMargin',
        'AssetTurnover', 'news_sentiment'
    ]

    for col in final_feature_cols:
        if col not in df.columns:
            df[col] = 0

    final_df = df[final_feature_cols + ['Target']].copy() # Use .copy() to prevent SettingWithCopyWarning

    # FIX: A more robust cleaning process
    # Step A: Drop only rows where the essential 'Target' column is missing.
    final_df.dropna(subset=['Target'], inplace=True)
    # Step B: Fill any remaining NaNs in feature columns (from rolling windows) with 0.
    final_df.fillna(0, inplace=True)

    return final_df