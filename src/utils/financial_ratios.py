import pandas as pd
import numpy as np

def _get_column(df, column_options):
    """Helper function to find the first available column from a list of options."""
    for col in column_options:
        if col in df.columns:
            return df[col]
    return pd.Series(0, index=df.index) # Return a series of zeros if no column is found

def calculate_all_ratios(financials_df):
    """
    Calculates a comprehensive set of financial ratios from a DataFrame.
    This version is robust and checks for multiple common names for financial metrics.
    """
    if financials_df.empty:
        return pd.DataFrame()

    ratios = pd.DataFrame()
    if 'Date' in financials_df.columns:
        ratios['Date'] = financials_df['Date']
    
    # --- Define data columns using the helper function for robustness ---
    revenues = _get_column(financials_df, ['Revenues', 'totalRevenue', 'revenue'])
    gross_profit = _get_column(financials_df, ['GrossProfit', 'grossProfit'])
    op_income = _get_column(financials_df, ['OperatingIncome', 'operatingIncome'])
    net_income = _get_column(financials_df, ['NetIncome', 'netIncome'])
    
    total_assets = _get_column(financials_df, ['TotalAssets', 'totalAssets'])
    current_assets = _get_column(financials_df, ['CurrentAssets', 'totalCurrentAssets'])
    inventory = _get_column(financials_df, ['Inventory', 'inventory'])
    cash = _get_column(financials_df, ['Cash', 'cash', 'cashAndCashEquivalents'])
    
    total_liabilities = _get_column(financials_df, ['TotalLiabilities', 'totalLiab'])
    current_liabilities = _get_column(financials_df, ['CurrentLiabilities', 'totalCurrentLiabilities'])
    
    equity = _get_column(financials_df, ['StockholdersEquity', 'totalStockholderEquity', 'totalEquity'])

    # --- Liquidity Ratios ---
    ratios['CurrentRatio'] = current_assets / current_liabilities
    ratios['QuickRatio'] = (current_assets - inventory) / current_liabilities
    
    # --- Leverage Ratios ---
    ratios['DebtToEquity'] = total_liabilities / equity
    ratios['DebtToAssets'] = total_liabilities / total_assets
    
    # --- Profitability Ratios ---
    ratios['GrossProfitMargin'] = gross_profit / revenues
    ratios['OperatingMargin'] = op_income / revenues
    ratios['NetProfitMargin'] = net_income / revenues
    ratios['ReturnOnAssets'] = net_income / total_assets
    ratios['ReturnOnEquity'] = net_income / equity
    
    # --- Efficiency Ratios ---
    ratios['AssetTurnover'] = revenues / total_assets

    # Clean up any potential infinity values from division by zero
    ratios = ratios.replace([np.inf, -np.inf], np.nan)
    
    return ratios
