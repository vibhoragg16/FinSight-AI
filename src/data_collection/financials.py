import os
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.config import PROCESSED_DATA_PATH


YF_BS_KEYS = {
    'CurrentAssets': 'Total Current Assets',
    'CurrentLiabilities': 'Total Current Liabilities',
    'TotalLiabilities': 'Total Liab',
    'StockholdersEquity': 'Total Stockholder Equity',
    'TotalAssets': 'Total Assets',
}

YF_IS_KEYS = {
    'NetIncome': 'Net Income',
    'Revenues': 'Total Revenue',
}


def _extract_series(frame: pd.DataFrame, yf_row_key: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    if yf_row_key not in frame.index:
        return pd.Series(dtype=float)
    series = frame.loc[yf_row_key]
    # Ensure Series with datetime index
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
    return series


def fetch_financials_dataframe(ticker: str) -> pd.DataFrame:
    """
    Fetches quarterly financials for a ticker from yfinance and maps them
    to a unified DataFrame used by the ratios pipeline.

    Columns: Date, CurrentAssets, CurrentLiabilities, TotalLiabilities,
             StockholdersEquity, TotalAssets, NetIncome, Revenues
    """
    stock = yf.Ticker(ticker)
    bs_q = getattr(stock, 'quarterly_balance_sheet', None)
    if bs_q is None or (hasattr(bs_q, 'empty') and bs_q.empty):
        bs_q = pd.DataFrame()
    is_q = getattr(stock, 'quarterly_financials', None)
    if is_q is None or (hasattr(is_q, 'empty') and is_q.empty):
        is_q = pd.DataFrame()

    # Normalize to use columns as datetimes
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        try:
            df.columns = pd.to_datetime(df.columns, utc=True, errors='coerce').tz_convert(None)
        except Exception:
            pass
        return df

    bs_q = _normalize_cols(bs_q)
    is_q = _normalize_cols(is_q)

    dates: List[pd.Timestamp] = []
    if not bs_q.empty:
        dates.extend(list(bs_q.columns))
    if not is_q.empty:
        dates.extend(list(is_q.columns))
    dates = sorted({d for d in dates if pd.notna(d)})

    records = []
    for d in dates:
        row = {
            'Date': pd.to_datetime(d, errors='coerce')
        }
        for out_key, yf_key in YF_BS_KEYS.items():
            row[out_key] = _extract_series(bs_q, yf_key).get(d, np.nan)
        for out_key, yf_key in YF_IS_KEYS.items():
            row[out_key] = _extract_series(is_q, yf_key).get(d, np.nan)
        records.append(row)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df = df.sort_values('Date').reset_index(drop=True)

    # Persist a copy for transparency/debugging
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, ticker), exist_ok=True)
    output_csv = os.path.join(PROCESSED_DATA_PATH, ticker, f"{ticker}_financials_quarterly.csv")
    try:
        df.to_csv(output_csv, index=False)
    except Exception:
        pass

    return df


