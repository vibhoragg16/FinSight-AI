import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH
from src.rag.query_processor import query_rag_system
from src.ai_core.qualitative_brain import QualitativeBrain
from src.ai_core.quantitative_brain import QuantitativeBrain
from src.data_collection.financials import fetch_financials_dataframe

# --- Page Config ---
st.set_page_config(page_title="AI Corporate Intelligence", layout="wide", page_icon="🤖")

# --- Load CSS ---
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css('src/dashboard/components/style.css')

# --- App State & Initialization ---
@st.cache_resource
def init_brains():
    try:
        return QualitativeBrain(), QuantitativeBrain()
    except ValueError as e:
        st.error(f"Initialization Error: {e}. Please set your GROQ_API_KEY.")
        st.stop()

qual_brain, quant_brain = init_brains()

# --- Sidebar ---
with st.sidebar:
    st.header("🏢 Company Selection")
    selected_company = st.selectbox("Select a Company", TARGET_COMPANIES)

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_company_data(ticker):
    market_file = os.path.join(MARKET_DATA_PATH, f'{ticker}_market_data.csv')
    news_file = os.path.join(NEWS_DATA_PATH, f'{ticker}_news.json')
    market_df = pd.read_csv(market_file) if os.path.exists(market_file) else pd.DataFrame()
    if not market_df.empty:
        market_df['Date'] = (
            pd.to_datetime(market_df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
        )
    # Load raw JSON and normalize to a flat DataFrame for caching stability
    news_df = pd.read_json(news_file, typ='frame') if os.path.exists(news_file) else pd.DataFrame()
    if not news_df.empty:
        # If the JSON contains an 'articles' list or dict objects, normalize to rows
        if 'articles' in news_df:
            articles = news_df['articles']
            try:
                articles_df = pd.json_normalize(articles)
            except Exception:
                # Fallback: ensure iterable of dicts
                articles_df = pd.json_normalize(list(articles) if isinstance(articles, (list, tuple)) else [])
            # Coerce published date with tz handling
            if 'publishedAt' in articles_df:
                articles_df['publishedAt'] = (
                    pd.to_datetime(articles_df['publishedAt'], utc=True, errors='coerce').dt.tz_convert(None)
                )
            news_df = articles_df
        else:
            # If already tabular with nested fields, try best-effort normalization
            if any(news_df.dtypes.apply(lambda dt: dt == 'object')):
                try:
                    news_df = pd.json_normalize(news_df.to_dict(orient='records'))
                except Exception:
                    pass
    # Fetch financials (cached by this function)
    financials_df = fetch_financials_dataframe(ticker)
    return market_df, news_df, financials_df

market_data, news_data, financials_data = load_company_data(selected_company)

# --- AI Analysis ---
@st.cache_data(ttl=3600)
def run_ai_analysis(ticker, market_df, news_df, financials_df):
    # Compute news sentiment from a flat articles dataframe
    news_sentiment = 0.0
    if not news_df.empty:
        titles_series = None
        if 'title' in news_df.columns:
            titles_series = news_df['title']
        elif 'articles' in news_df.columns:
            try:
                titles_series = pd.json_normalize(news_df['articles']).get('title')
            except Exception:
                titles_series = None
        if titles_series is not None and not titles_series.empty:
            sentiments = titles_series.astype(str).apply(lambda x: qual_brain.analyze_text_sentiment(x))
            if not sentiments.empty:
                news_sentiment = float(sentiments.mean())

    analysis_results = quant_brain.get_analysis(market_df, financials_df, news_sentiment, ticker=ticker)
    return analysis_results

analysis = run_ai_analysis(selected_company, market_data, news_data, financials_data)
health_score = analysis['health_score']
prediction = analysis['prediction']
confidence = analysis['confidence']
avg_sentiment = analysis.get('news_sentiment', 0)

# --- Main Dashboard ---
st.markdown(f'<h1 class="main-header">AI Corporate Intelligence: {selected_company}</h1>', unsafe_allow_html=True)

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    score_class = "health-score-good" if health_score >= 70 else "health-score-warning" if health_score >= 50 else "health-score-danger"
    st.markdown(f'<div class="metric-card"><h4>🧠 Fundamental Health</h4><div class="{score_class}">{health_score:.1f}/100</div></div>', unsafe_allow_html=True)
with col2:
    direction_emoji = "📈" if prediction == "Bullish" else "📉"
    st.markdown(f'<div class="metric-card"><h4>🎯 Stock Forecast</h4><div class="forecast-text">{direction_emoji} {prediction}</div><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
with col3:
    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😟" if avg_sentiment < -0.1 else "😐"
    st.markdown(f'<div class="metric-card"><h4>📰 News Sentiment</h4><div class="sentiment-text">{sentiment_emoji} {avg_sentiment:.2f}</div></div>', unsafe_allow_html=True)
with col4:
    if not market_data.empty:
        latest_market = market_data.iloc[-1]
        price_delta = latest_market['Close'] - latest_market['Open']
        delta_color = "green" if price_delta >= 0 else "red"
        st.markdown(f'<div class="metric-card"><h4>💰 Latest Price</h4><div class="price-text">${latest_market["Close"]:.2f}</div><p style="color:{delta_color};">{price_delta:+.2f}</p></div>', unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🤖 AI Analyst Chat", "📰 News Analysis", "💡 Deep Dive"])

with tab1:
    st.header("Market Performance")
    if not market_data.empty:
        fig = go.Figure(data=[go.Candlestick(x=market_data['Date'], open=market_data['Open'], high=market_data['High'], low=market_data['Low'], close=market_data['Close'], name="Price")])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Chat with the AI Analyst")
    if 'messages' not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input(f"Ask about {selected_company}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = query_rag_system(selected_company, prompt)
                st.markdown(response)

with tab3:
    st.header("Recent Company News")
    if not news_data.empty:
        display_df = news_data.copy()
        # Attempt to align schema
        if 'publishedAt' in display_df:
            display_df['publishedAt'] = (
                pd.to_datetime(display_df['publishedAt'], utc=True, errors='coerce').dt.tz_convert(None)
            )
        for _, article in display_df.head(5).iterrows():
            title = article.get('title', 'Untitled')
            url = article.get('url', '')
            source = article.get('source.name', article.get('source', 'Unknown'))
            published = article.get('publishedAt')
            published_str = pd.to_datetime(published, errors='coerce').strftime('%Y-%m-%d') if pd.notna(published) else ''
            description = article.get('description', '')
            st.write(f"**[{title}]({url})**")
            st.write(f"_{source} - {published_str}_")
            if description:
                st.write(description)
            st.divider()

with tab4:
    st.header("💡 AI-Powered Deep Dive")
    with st.spinner("Generating AI executive summary..."):
        summary_prompt = f"""
        Generate an executive summary for {selected_company} based on:
        - Health Score: {health_score:.1f}/100
        - Forecast: {prediction} ({confidence:.1%} confidence)
        - News Sentiment: {avg_sentiment:.2f}
        Identify key strengths and concerns.
        """
        try:
            response = qual_brain.groq_client.chat.completions.create(
                model= "llama3-8b-8192",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Could not generate summary: {e}")
