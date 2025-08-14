import pandas as pd
import numpy as np

def calculate_all_ratios(financials_df):
    """
    Calculates a comprehensive set of financial ratios from a DataFrame
    of financial statement data. The input DataFrame should be pivoted.
    """
    if financials_df.empty:
        return pd.DataFrame()

    ratios = pd.DataFrame()
    if 'Date' in financials_df.columns:
        ratios['Date'] = financials_df['Date']
    
    # --- Liquidity Ratios ---
    if 'CurrentAssets' in financials_df and 'CurrentLiabilities' in financials_df:
        ratios['CurrentRatio'] = financials_df['CurrentAssets'] / financials_df['CurrentLiabilities']
    
    # --- Leverage Ratios ---
    if 'TotalLiabilities' in financials_df and 'StockholdersEquity' in financials_df:
        ratios['DebtToEquity'] = financials_df['TotalLiabilities'] / financials_df['StockholdersEquity']

    # --- Profitability Ratios ---
    if 'NetIncome' in financials_df and 'Revenues' in financials_df:
        ratios['NetProfitMargin'] = financials_df['NetIncome'] / financials_df['Revenues']
    
    if 'NetIncome' in financials_df and 'TotalAssets' in financials_df:
        ratios['ReturnOnAssets'] = financials_df['NetIncome'] / financials_df['TotalAssets']
        
    if 'NetIncome' in financials_df and 'StockholdersEquity' in financials_df:
        ratios['ReturnOnEquity'] = financials_df['NetIncome'] / financials_df['StockholdersEquity']

    # --- Efficiency Ratios ---
    if 'Revenues' in financials_df and 'TotalAssets' in financials_df:
        ratios['AssetTurnover'] = financials_df['Revenues'] / financials_df['TotalAssets']

    # Clean up any potential infinity values
    ratios = ratios.replace([np.inf, -np.inf], np.nan)
    
    return ratios