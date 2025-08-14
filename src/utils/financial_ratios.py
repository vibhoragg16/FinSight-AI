# File: financial_ratios.py

import pandas as pd
import numpy as np

def _get_column(df, column_options):
    """Helper function to find the first available column from a list of options."""
    for col in column_options:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return df[col]
    return None # Return None if no valid column is found

def calculate_all_ratios(financials_df):
    """
    Calculates financial ratios and also reports which underlying data points were missing.
    Returns a tuple: (ratios_df, missing_columns)
    """
    if financials_df.empty:
        return pd.DataFrame(), []

    ratios = pd.DataFrame()
    if 'Date' in financials_df.columns:
        ratios['Date'] = financials_df['Date']
    
    # --- Define required data points and track missing ones ---
    all_required = {
        'revenues': ['Revenues', 'totalRevenue', 'revenue'],
        'gross_profit': ['GrossProfit', 'grossProfit'],
        'op_income': ['OperatingIncome', 'operatingIncome'],
        'net_income': ['NetIncome', 'netIncome'],
        'total_assets': ['TotalAssets', 'totalAssets'],
        'current_assets': ['CurrentAssets', 'totalCurrentAssets'],
        'inventory': ['Inventory', 'inventory'],
        'total_liabilities': ['TotalLiabilities', 'totalLiab'],
        'current_liabilities': ['CurrentLiabilities', 'totalCurrentLiabilities'],
        'equity': ['StockholdersEquity', 'totalStockholderEquity', 'totalEquity']
    }
    
    data = {}
    missing_columns = []
    
    for key, options in all_required.items():
        column_data = _get_column(financials_df, options)
        if column_data is not None:
            data[key] = column_data
        else:
            missing_columns.append(options[0]) # Report the primary name as missing
            data[key] = 0 # Default to 0 to avoid crashing calculations

    # --- Perform calculations using the fetched data ---
    ratios['CurrentRatio'] = data['current_assets'] / data['current_liabilities']
    ratios['QuickRatio'] = (data['current_assets'] - data['inventory']) / data['current_liabilities']
    ratios['DebtToEquity'] = data['total_liabilities'] / data['equity']
    ratios['DebtToAssets'] = data['total_liabilities'] / data['total_assets']
    ratios['GrossProfitMargin'] = data['gross_profit'] / data['revenues']
    ratios['OperatingMargin'] = data['op_income'] / data['revenues']
    ratios['NetProfitMargin'] = data['net_income'] / data['revenues']
    ratios['ReturnOnAssets'] = data['net_income'] / data['total_assets']
    ratios['ReturnOnEquity'] = data['net_income'] / data['equity']
    ratios['AssetTurnover'] = data['revenues'] / data['total_assets']

    ratios = ratios.replace([np.inf, -np.inf], np.nan)
    
    return ratios, missing_columns