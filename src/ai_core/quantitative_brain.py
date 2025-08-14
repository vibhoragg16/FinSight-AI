# File: src/ai_core/quantitative_brain.py

import pandas as pd
import os
import joblib
import logging

from src.models.pca_health_scorer import PCAHealthScorer
from src.features.build_features import prepare_features_for_prediction
from src.utils.financial_ratios import calculate_all_ratios
from src.utils.config import MODEL_SAVE_PATH as MODELS_PATH

class QuantitativeBrain:
    """
    A high-level coordinator for all quantitative analysis. It uses specialized
    models for scoring, prediction, and feature engineering.
    """
    def __init__(self):
        """Initializes the QuantitativeBrain with its models."""
        self.health_scorer = PCAHealthScorer()
        
        try:
            # This correctly loads the trained stock predictor model
            model_path = os.path.join(MODELS_PATH, 'stock_predictor_model.pkl')
            self.predictor = joblib.load(model_path)
            logging.info("QuantitativeBrain initialized with trained stock predictor.")
        except FileNotFoundError:
            logging.error("Stock predictor model not found. Please run the training script.")
            self.predictor = None

    def get_analysis(self, market_data, financials_data, news_sentiment, ticker: str | None = None):
            """
            Performs a full quantitative analysis and returns key insights.
            """
            # --- Health Score Calculation ---
            
            # FIX: Unpack the tuple returned by calculate_all_ratios
            if not financials_data.empty:
                ratios_df, missing_cols = calculate_all_ratios(financials_data)
                # The 'missing_cols' variable is now available if needed for logging
            else:
                ratios_df = pd.DataFrame()

            health_score = self.health_scorer.calculate_score(ratios_df)

            # --- Prediction Calculation ---
            features_df = prepare_features_for_prediction(market_data, ratios_df, news_sentiment)
            
            prediction = "N/A"
            confidence = 0.0

            if self.predictor is not None and not features_df.empty:
                latest_features = features_df.tail(1).drop(columns=['Target'], errors='ignore')
                
                if hasattr(self.predictor, 'feature_names_in_'):
                    latest_features = latest_features.reindex(columns=self.predictor.feature_names_in_, fill_value=0)

                pred_proba = self.predictor.predict_proba(latest_features)[0]
                prediction = "Bullish" if pred_proba[1] > 0.5 else "Bearish"
                confidence = max(pred_proba)
            else:
                logging.warning("Predictor model not loaded or no features available, skipping prediction.")
                
            return {
                'health_score': health_score,
                'prediction': prediction,
                'confidence': confidence,
                'news_sentiment': news_sentiment
            }