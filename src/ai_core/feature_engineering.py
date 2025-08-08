import pandas as pd

def prepare_features_for_prediction(market_data, ratios_df, news_sentiment):
    """
    Prepares a comprehensive feature set for the prediction model.
    """
    if market_data.empty:
        return pd.DataFrame()
    
    features_df = market_data.copy()
    # Ensure timezone-aware parsing and then remove timezone for consistency
    features_df['Date'] = (
        pd.to_datetime(features_df['Date'], utc=True, errors='coerce')
        .dt.tz_convert(None)
    )

    if ratios_df is not None and not ratios_df.empty:
        if 'Date' in ratios_df.columns:
            ratios_df['Date'] = (
                pd.to_datetime(ratios_df['Date'], utc=True, errors='coerce')
                .dt.tz_convert(None)
            )
            features_df = pd.merge_asof(
                features_df.sort_values('Date'),
                ratios_df.sort_values('Date'),
                on='Date',
                direction='backward'
            )

    features_df['news_sentiment'] = news_sentiment
    features_df['target'] = (features_df['Close'].shift(-5) > features_df['Close']).astype('float')
    
    feature_columns = [
        'RSI', 'MACD', 'SMA_20', 'SMA_50', 'Volatility_30', 'Volume_Ratio',
        'CurrentRatio', 'DebtToEquity', 'ReturnOnEquity', 'news_sentiment', 'target'
    ]
    
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Build feature matrix and clean NaNs conservatively: keep rows where label is present
    result = features_df[feature_columns].copy()
    if 'target' in result.columns:
        result = result.dropna(subset=['target'])
    # Fill remaining NaNs in feature columns (except target) with 0
    non_label_cols = [c for c in result.columns if c != 'target']
    result[non_label_cols] = result[non_label_cols].fillna(0)
    return result