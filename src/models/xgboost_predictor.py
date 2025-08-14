import pandas as pd
import xgboost as xgb
from pathlib import Path
from src.utils.config import MODEL_SAVE_PATH

class XGBoostPredictor:
    """
    A dedicated class for training and using the XGBoost prediction model.
    """
    def __init__(self):
        self.model_path = Path(MODEL_SAVE_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_file = self.model_path / "xgb_stock_predictor.json"
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the XGBoost model if it exists."""
        if self.model_file.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_file)
            print("XGBoost predictor model loaded.")

    def model_exists(self):
        return self.model is not None

    def train(self, X, y):
        """Trains the XGBoost model."""
        print("Training XGBoost prediction model...")
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0
        )
        
        self.model.fit(X, y)
        self.model.save_model(self.model_file)
        print(f"Model trained and saved to {self.model_file}")

    def predict(self, features):
        """Makes a prediction using the loaded model."""
        if not self.model_exists():
            return "N/A", 0.0

        if hasattr(self.model, 'feature_names_in_'):
            model_cols = self.model.feature_names_in_
            features = features.reindex(columns=model_cols, fill_value=0)

        try:
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
        except Exception:
            return "N/A", 0.0

        direction = "Bullish" if prediction == 1 else "Bearish"
        confidence = max(prediction_proba)
        
        return direction, confidence
