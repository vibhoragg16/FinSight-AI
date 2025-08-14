import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from src.utils.config import MODEL_SAVE_PATH as MODELS_PATH
import os
import streamlit as st
from huggingface_hub import hf_hub_download
from src.utils.config import HF_REPO_ID

@st.cache_resource
def load_model_from_hub(repo_id, filename):
    """Downloads a model from Hugging Face Hub and caches it."""
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return joblib.load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model {filename} from {repo_id}: {e}")
        st.error(f"Critical model file '{filename}' could not be loaded.")
        return None

class PCAHealthScorer:
    """
    A dedicated class for calculating the fundamental health score using PCA.
    """
    def __init__(self):
        """Initializes by loading models from Hugging Face Hub."""
        self.scaler = load_model_from_hub(repo_id=HF_REPO_ID, filename="models/saved/health_scaler.pkl")
        self.pca = load_model_from_hub(repo_id=HF_REPO_ID, filename="models/saved/health_pca.pkl")
        if self.scaler and self.pca:
            logging.info("PCAHealthScorer initialized with models from Hugging Face Hub.")

    def train(self, ratios_df):
        """Trains and saves the scaler and PCA models."""
        numeric_cols = ratios_df.select_dtypes(include=np.number).columns
        features = ratios_df[numeric_cols].fillna(0)
        # Require at least 3 samples for a stable PCA model
        if len(features) < 3:
            self.scaler = None
            self.pca = None
            return
        
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        
        self.pca = PCA(n_components=1)
        self.pca.fit(scaled_features)
        
        joblib.dump(self.scaler, self.scaler_file)
        joblib.dump(self.pca, self.pca_file)
        print("PCA Health Scorer models trained and saved.")

    def calculate_score(self, ratios_df):
        """Calculates the health score for the latest data point."""
        if ratios_df is None or ratios_df.empty:
            return 50.0

        # If we don't have enough history for PCA, compute a heuristic score
        numeric_cols = ratios_df.select_dtypes(include=np.number).columns
        latest_row = ratios_df[numeric_cols].tail(1).fillna(0)
        if len(ratios_df) < 3:
            return self._heuristic_score(latest_row)

        if self.scaler is None or self.pca is None:
            self.train(ratios_df)
            if self.scaler is None or self.pca is None:
                # Training skipped due to insufficient samples
                return self._heuristic_score(latest_row)
        
        latest_features = latest_row

        if latest_features.empty:
            return 50.0

        scaled_features = self.scaler.transform(latest_features)
        principal_component = self.pca.transform(scaled_features)
        
        score = 50 + (principal_component[0][0] * 10)
        return max(0, min(100, score))

    def _heuristic_score(self, latest_row: pd.DataFrame) -> float:
        """A simple rule-based score when PCA cannot be used reliably."""
        row = latest_row.squeeze()
        # Extract ratios with fallbacks
        current_ratio = float(row.get('CurrentRatio', 0) or 0)
        debt_to_equity = float(row.get('DebtToEquity', 0) or 0)
        return_on_equity = float(row.get('ReturnOnEquity', 0) or 0)
        return_on_assets = float(row.get('ReturnOnAssets', 0) or 0)

        # Normalize to 0-100 intuitive scales
        def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
            return max(lo, min(hi, x))

        # Current ratio: 1.5-2.5 is healthy
        cr_score = clamp(100 * min(current_ratio / 2.0, 1.0))
        # Debt-to-equity: lower is better, >3 is risky
        dte_score = clamp(100 * (1.0 - min(debt_to_equity / 3.0, 1.0)))
        # ROE: 0.0-0.25 (0-25%) maps to 0-100
        roe_score = clamp(100 * min(return_on_equity, 0.25) / 0.25)
        # ROA: 0.0-0.10 (0-10%) maps to 0-100
        roa_score = clamp(100 * min(return_on_assets, 0.10) / 0.10)

        # Weighted blend
        score = 0.30 * cr_score + 0.25 * dte_score + 0.30 * roe_score + 0.15 * roa_score
        return float(round(score, 1))

