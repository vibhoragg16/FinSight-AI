# src/dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH
from src.rag.query_processor import query_rag_system
from src.ai_core.qualitative_brain import QualitativeBrain
from src.ai_core.quantitative_brain import QuantitativeBrain

# --- Page Config ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="💡")

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
        st.error(f"Initialization Error: {e}. Please set your GROQ_API_KEY in the Streamlit secrets.")
        st.stop()

qual_brain, quant_brain = init_brains()

# --- Sidebar ---
with st.sidebar:
    st.header("💡 FinSight AI")
    selected_company = st.selectbox("Select a Company", TARGET_COMPANIES)

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_company_data(ticker):
    market_file = os.path.join(MARKET_DATA_PATH, f'{ticker}_market_data.csv')
    news_file = os.path.join(NEWS_DATA_PATH, f'{ticker}_news.json')
    market_df = pd.read_csv(market_file) if os.path.exists(market_file) else pd.DataFrame()
    if not market_df.empty:
        market_df['Date'] = pd.to_datetime(market_df['Date'])
    news_df = pd.read_json(news_file) if os.path.exists(news_file) else pd.DataFrame()
    return market_df, news_df

market_data, news_data = load_company_data(selected_company)

# --- AI Analysis ---
@st.cache_data(ttl=3600)
def run_ai_analysis(market_df, news_df):
    news_sentiment = 0
    if not news_df.empty and 'articles' in news_df:
        try:
            articles_df = pd.json_normalize(news_df['articles'])
            if not articles_df.empty and 'title' in articles_df.columns:
                articles_df['sentiment'] = articles_df['title'].astype(str).apply(qual_brain.analyze_text_sentiment)
                news_sentiment = articles_df['sentiment'].mean()
        except Exception as e:
            logging.error(f"Error processing news data for sentiment: {e}")
            news_sentiment = 0

    financials_df = pd.DataFrame()
    analysis_results = quant_brain.get_analysis(market_df, financials_df, news_sentiment)
    return analysis_results

# --- Main Dashboard ---
st.markdown(f'<h1 class="main-header">AI Corporate Intelligence: {selected_company}</h1>', unsafe_allow_html=True)

if market_data.empty and news_data.empty:
     st.warning(f"No data found for {selected_company}. Please run the collection and processing scripts first using 'python main.py collect' and 'python main.py process'.")

if not market_data.empty:
    analysis = run_ai_analysis(market_data, news_data)
    health_score = analysis['health_score']
    prediction = analysis['prediction']
    confidence = analysis['confidence']
    avg_sentiment = analysis.get('news_sentiment', 0)

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
        latest_market = market_data.iloc[-1]
        price_delta = latest_market['Close'] - latest_market['Open']
        delta_color = "green" if price_delta >= 0 else "red"
        st.markdown(f'<div class="metric-card"><h4>💰 Latest Price</h4><div class="price-text">${latest_market["Close"]:.2f}</div><p style="color:{delta_color};">{price_delta:+.2f}</p></div>', unsafe_allow_html=True)
else:
    st.warning("Market data for this company is not available. Please run the scheduler to collect data.")


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🤖 AI Analyst Chat", "📰 News Analysis", "💡 Deep Dive"])

with tab1:
    if not market_data.empty:
        st.header("Market Performance")
        fig = go.Figure(data=[go.Candlestick(x=market_data['Date'], open=market_data['Open'], high=market_data['High'], low=market_data['Low'], close=market_data['Close'], name="Price")])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No market data to display.")

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
    if not news_data.empty and 'articles' in news_data:
        articles_df = pd.json_normalize(news_data['articles'])
        for _, article in articles_df.head(5).iterrows():
            st.write(f"**[{article['title']}]({article['url']})**")
            st.write(f"_{article['source.name']} - {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d')}_")
            st.write(article['description'])
            st.divider()
    else:
        st.info("No news data to display.")

with tab4:
    st.header("💡 AI-Powered Deep Dive")
    if not market_data.empty:
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
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Could not generate summary: {e}")
    else:
        st.info("Not enough data available to generate a deep dive analysis.")
