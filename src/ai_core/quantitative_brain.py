import pandas as pd
from src.models.pca_health_scorer import PCAHealthScorer
from src.models.xgboost_predictor import XGBoostPredictor
from src.ai_core.feature_engineering import prepare_features_for_prediction
from src.utils.financial_ratios import calculate_all_ratios
from src.data_collection.financials import fetch_financials_dataframe

class QuantitativeBrain:
    """
    A high-level coordinator for all quantitative analysis. It uses specialized
    modules for scoring, prediction, and feature engineering.
    """
    def __init__(self):
        self.health_scorer = PCAHealthScorer()
        self.predictor = XGBoostPredictor()

    def get_analysis(self, market_data, financials_data, news_sentiment, ticker: str | None = None):
        """
        Performs a full quantitative analysis and returns key insights.
        """
        # If no financials provided, fetch from yfinance as a fallback
        if financials_data is None or financials_data.empty:
            inferred_ticker = ticker
            fetched_df = fetch_financials_dataframe(inferred_ticker) if inferred_ticker else pd.DataFrame()
            financials_df = fetched_df if not fetched_df.empty else pd.DataFrame()
        else:
            financials_df = financials_data

        ratios_df = calculate_all_ratios(financials_df) if not financials_df.empty else pd.DataFrame()
        health_score = self.health_scorer.calculate_score(ratios_df)
        features_df = prepare_features_for_prediction(market_data, ratios_df, news_sentiment)

        if not features_df.empty:
            # Ensure we have enough samples to train a model
            if 'target' in features_df.columns:
                labeled = features_df.dropna(subset=['target'])
            else:
                labeled = features_df

            if len(labeled) >= 50 and labeled['target'].nunique() > 1:
                X = labeled.drop(columns=['target'])
                y = labeled['target']
                latest_features = X.tail(1)

                if not self.predictor.model_exists():
                    self.predictor.train(X, y)

                prediction, confidence = self.predictor.predict(latest_features)
            else:
                prediction, confidence = "N/A", 0.0
        else:
            prediction, confidence = "N/A", 0.0

        return {
            "health_score": health_score,
            "prediction": prediction,
            "confidence": confidence,
            "ratios": ratios_df.tail(1).to_dict('records')[0] if not ratios_df.empty else {},
            "news_sentiment": news_sentiment
        }