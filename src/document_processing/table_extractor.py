import camelot
import pandas as pd
import logging
import os
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def extract_tables_from_pdf(pdf_path: str, pages: str = 'all', flavor: str = 'lattice'):
    """
    Extracts all tables from a given PDF file using Camelot.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return []

    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor, suppress_stdout=True)
        logging.info(f"Found {tables.n} tables in {os.path.basename(pdf_path)} using '{flavor}' method.")
        return [table.df for table in tables]
    except Exception as e:
        logging.error(f"Failed to extract tables from {pdf_path}: {e}")
        return []

def clean_financial_table(df: pd.DataFrame):
    """
    Performs basic cleaning on an extracted financial table.
    """
    if df.empty:
        return df

    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    
    for col in df.columns:
        if df[col].astype(str).str.contains(r'[\$,]', regex=True).any():
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df[col] = df[col].astype(str).str.replace(r'\(', '-', regex=True).str.replace(r'\)', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df