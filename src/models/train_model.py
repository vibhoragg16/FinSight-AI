# File: src/models/train_model.py

import pandas as pd
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sys

# Add project root to path to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, PROCESSED_DATA_PATH, MARKET_DATA_PATH,MODEL_SAVE_PATH as MODELS_PATH
from src.features.build_features import prepare_features_for_prediction
from src.utils.financial_ratios import calculate_all_ratios

# Ensure models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

def train_health_scorer():
    """Trains and saves the PCA-based health scorer model."""
    logging.info("Starting training for the PCA Health Scorer...")
    all_ratios = []
    
    for company in TARGET_COMPANIES:
        fin_path = os.path.join(PROCESSED_DATA_PATH, company, f'{company}_financials_quarterly.csv')
        if os.path.exists(fin_path):
            financials_df = pd.read_csv(fin_path)
            ratios_df = calculate_all_ratios(financials_df)
            if not ratios_df.empty:
                all_ratios.append(ratios_df)

    if not all_ratios:
        logging.warning("No financial ratios generated. Skipping health scorer training.")
        return

    combined_ratios_df = pd.concat(all_ratios).drop(columns=['Date']).reset_index(drop=True)
    
    # GOOD PRACTICE: Update to modern .ffill() and .bfill() to resolve FutureWarning
    combined_ratios_df = combined_ratios_df.ffill().bfill()
    
    # FIX 1: Fill any remaining NaN values with 0 instead of dropping rows.
    combined_ratios_df.fillna(0, inplace=True)

    if combined_ratios_df.empty:
        logging.warning("Ratio data is empty after cleaning. Skipping health scorer training.")
        return

    scaler = StandardScaler()
    scaler.fit(combined_ratios_df)
    joblib.dump(scaler, os.path.join(MODELS_PATH, 'health_scaler.pkl'))
    logging.info(f"Health score scaler trained on {len(combined_ratios_df)} samples and saved.")

    scaled_data = scaler.transform(combined_ratios_df)
    pca = PCA(n_components=1)
    pca.fit(scaled_data)
    joblib.dump(pca, os.path.join(MODELS_PATH, 'health_pca.pkl'))
    logging.info("PCA model for health score trained and saved.")


def train_stock_predictor():
    """Trains and saves the stock price movement prediction model (XGBoost)."""
    logging.info("Starting training for the Stock Predictor...")
    all_features = []
    
    for company in TARGET_COMPANIES:
        market_path = os.path.join(MARKET_DATA_PATH, f'{company}_market_data.csv')
        fin_path = os.path.join(PROCESSED_DATA_PATH, company, f'{company}_financials_quarterly.csv')
        
        if os.path.exists(market_path) and os.path.exists(fin_path):
            market_df = pd.read_csv(market_path)
            financials_df = pd.read_csv(fin_path)
            ratios_df = calculate_all_ratios(financials_df)
            
            # FIX 2: Pass the required 'news_sentiment' argument with a neutral value for training.
            features = prepare_features_for_prediction(market_df, ratios_df, news_sentiment=0.0)
            if not features.empty:
                all_features.append(features)

    if not all_features:
        logging.warning("No features generated. Skipping stock predictor training.")
        return

    combined_features_df = pd.concat(all_features).reset_index(drop=True)
    combined_features_df.dropna(inplace=True)

    if combined_features_df.empty or 'Target' not in combined_features_df.columns:
        logging.warning("Feature data is empty or 'Target' column is missing. Skipping training.")
        return

    X = combined_features_df.drop(columns=['Target'])
    y = combined_features_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    logging.info(f"Stock predictor model trained. Test Accuracy: {accuracy:.2f}")

    joblib.dump(model, os.path.join(MODELS_PATH, 'stock_predictor_model.pkl'))
    logging.info("Stock predictor model saved.")


def train_all_models():
    """Main function to run all model training processes."""
    logging.info("--- Starting Model Training Suite ---")
    train_health_scorer()
    train_stock_predictor()
    logging.info("--- Model Training Suite Finished ---")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_all_models()