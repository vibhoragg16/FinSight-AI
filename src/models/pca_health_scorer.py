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
        model_path = hf_hub_download(repo_id=repo_id, filename=filename,repo_type="dataset")
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

