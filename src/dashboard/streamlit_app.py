# src/dashboard/streamlit_app.py

import sys
import subprocess
import importlib.util

# Check if pysqlite3 is available
spec = importlib.util.find_spec("pysqlite3")
if spec is None:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
        print("Successfully installed pysqlite3-binary")
    except Exception as e:
        print(f"Failed to install pysqlite3-binary: {e}")

# Replace sqlite3 with pysqlite3
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("SQLite3 module successfully replaced with pysqlite3")
except ImportError as e:
    print(f"Failed to import pysqlite3: {e}")
    # Try alternative approach
    try:
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
        print("SQLite3 module replaced using alternative method")
    except ImportError:
        print("Could not replace sqlite3 module - ChromaDB may fail")

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import sys
import logging
import re
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH, GROQ_LLM_MODEL, HF_REPO_ID
from src.rag.query_processor import query_rag_system
from src.ai_core.qualitative_brain import QualitativeBrain
from src.ai_core.quantitative_brain import QuantitativeBrain
from src.data_collection.financials import fetch_financials_dataframe
from src.utils.financial_ratios import calculate_all_ratios # Import the new function

# --- Page Config ---
st.set_page_config(page_title="FinSight AI", layout="wide", page_icon="💡")

# --- Load CSS ---
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css('src/dashboard/components/style.css')

@st.cache_data(ttl=3600) # Cache for 1 hour
def load_csv_from_hub(repo_id, filename):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to load CSV {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_json_from_hub(repo_id, filename):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return pd.read_json(file_path)
    except Exception as e:
        logging.error(f"Failed to load JSON {filename}: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=3600)
def get_filing_content_from_hub(ticker, filename):
    """Downloads a raw filing file from HF Hub and returns its content."""
    try:
        # Construct the path as it exists in the HF repo
        repo_file_path = f"data/raw/sec-edgar-filings/{ticker}/{filename.split('/')[0]}/{filename.split('/')[1]}"

        file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_file_path,
            repo_type="dataset"
        )
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load filing {filename} for {ticker}: {e}")
        return None
# Add this to your streamlit_app.py to replace the current init_brains function

@st.cache_resource
def init_brains():
    """Initialize AI brains with robust error handling."""
    try:
        # Check if GROQ_API_KEY is available from Streamlit secrets or environment
        groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not groq_key:
            st.error("🔑 GROQ_API_KEY is required. Please add it to your Streamlit secrets.")
            st.info("Go to your app settings and add GROQ_API_KEY to the secrets section.")
            st.stop()
        
        # Set the environment variable if it's from secrets so the client can find it
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = groq_key
        
        # Initialize the brains
        qual_brain = QualitativeBrain()
        quant_brain = QuantitativeBrain()
        
        return qual_brain, quant_brain
        
    except ValueError as e:
        st.error(f"🚫 Configuration Error: {e}")
        st.info("Please check your API key configuration in Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Initialization Error: Could not initialize AI components.")
        st.error(f"Details: {e}")
        
        if "proxies" in str(e).lower():
            st.warning("🔧 This looks like a package compatibility issue. The fix is likely in the 'qualitative_brain.py' file.")
        
        st.stop()

# --- App State & Initialization ---
# This line now calls your new, single function
qual_brain, quant_brain = init_brains()
# --- Sidebar ---
with st.sidebar:
    st.header("💡 FinSight AI")
    selected_company = st.selectbox("Select a Company", TARGET_COMPANIES)

# --- Data Loading ---
def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def load_company_data(ticker):
    """Loads all necessary data for a given ticker from Hugging Face Hub."""
    market_df = load_csv_from_hub(HF_REPO_ID, f"data/market_data/{ticker}_market_data.csv")
    news_df_raw = load_json_from_hub(HF_REPO_ID, f"data/news/{ticker}_news.json")
    financials_df = load_csv_from_hub(HF_REPO_ID, f"data/processed/{ticker}/{ticker}_financials_quarterly.csv")
    
    # --- Data Processing ---
    if not market_df.empty:
        market_df['Date'] = pd.to_datetime(market_df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
    
    news_df = pd.DataFrame()
    if not news_df_raw.empty and 'articles' in news_df_raw:
        try:
            articles_df = pd.json_normalize(news_df_raw['articles'])
            if 'publishedAt' in articles_df:
                articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'], utc=True, errors='coerce').dt.tz_convert(None)
            news_df = articles_df
        except Exception as e:
            logging.error(f"Error normalizing news data for {ticker}: {e}")
            
    return market_df, news_df, financials_df

market_data, news_data, financials_data = load_company_data(selected_company)
# --- AI Analysis ---
@st.cache_data(ttl=3600)
def run_ai_analysis(ticker, market_df, news_df, financials_df):
    news_sentiment = 0
    if not news_df.empty:
        try:
            if 'title' in news_df.columns:
                sentiments = news_df['title'].astype(str).apply(qual_brain.analyze_text_sentiment)
                if not sentiments.empty:
                    news_sentiment = float(sentiments.mean())
        except Exception as e:
            logging.error(f"Error processing news data for sentiment: {e}")
            news_sentiment = 0.0

    analysis_results = quant_brain.get_analysis(market_df, financials_df, news_sentiment, ticker=ticker)
    return analysis_results

# --- Enhanced Source Display Functions ---
def get_sec_link(filename, selected_company):
    """Generate SEC EDGAR link if possible"""
    company_ciks = {
        'AAPL': '0000320193', 'MSFT': '0000789019', 'GOOGL': '0001652044',
        'AMZN': '0001018724', 'TSLA': '0001318605', 'META': '0001326801',
        'NVDA': '0001045810', 'BRK.B': '0001067983', 'JNJ': '0000200406',
        'V': '0001403161',
    }
    accession_match = re.search(r'(\d{10}-\d{2}-\d{6})', filename)
    if accession_match and selected_company in company_ciks:
        cik = company_ciks.get(selected_company)
        accession = accession_match.group(1)
        if cik:
            return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"
    return None

def extract_relevant_paragraphs(content, query_keywords, max_paragraphs=3):
    """Extract paragraphs most relevant to the query using BeautifulSoup for cleaning."""
    if not query_keywords:
        return []

    # Use BeautifulSoup to parse and get clean text
    soup = BeautifulSoup(content, 'html.parser')

    # Get all text and split into potential paragraphs
    all_text = soup.get_text(separator='\n', strip=True)
    # Filter for longer paragraphs that are more likely to contain prose
    paragraphs = [p.strip() for p in all_text.split('\n') if len(p.strip()) > 150]

    scored_paragraphs = []
    for para in paragraphs:
        # Simple scoring: count keyword occurrences
        score = sum(1 for keyword in query_keywords if keyword.lower() in para.lower())
        # Penalize paragraphs that look like code/links
        if 'http' in para or '/' in para or len(para.split()) < 15:
            score -= 5

        if score > 0:
            scored_paragraphs.append((score, para))

    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    return [para for score, para in scored_paragraphs[:max_paragraphs]]


def display_enhanced_sources(sources, prompt=""):
    """
    Displays sources and re-enables advanced analysis by fetching full 
    document content on-demand from Hugging Face Hub.
    """
    st.markdown("---")
    st.markdown("### 📚 Sources & Direct Access")

    if not sources:
        st.info("No specific document sources were found for this answer.")
        return

    # Group sources by the document they came from
    doc_sources = {}
    for s in sources:
        source_path = s.metadata.get("source", "Unknown")
        if source_path == "Unknown":
            continue
        
        parts = source_path.split(os.sep)
        if 'sec-edgar-filings' in parts:
            # Create a unique key like: AAPL/10-K/0000320193-23-000106/primary-document.html
            key_parts = parts[parts.index('sec-edgar-filings')+1:]
            doc_key = "/".join(key_parts)
            
            if doc_key not in doc_sources:
                doc_sources[doc_key] = []
            doc_sources[doc_key].append(s.page_content)

    query_keywords = prompt.lower().split() if prompt else []

    for doc_key, snippets in doc_sources.items():
        view_state_key = f'active_view_{hash(doc_key)}'
        st.session_state.setdefault(view_state_key, None)
        
        # Extract ticker and filename for display and fetching
        key_parts = doc_key.split('/')
        ticker_from_key = key_parts[0]
        filename_for_display = " / ".join(key_parts[1:])
        filename_for_download = "/".join(key_parts[1:])

        with st.expander(f"📄 **{filename_for_display}**", expanded=True):
            sec_link = get_sec_link(os.path.basename(doc_key), ticker_from_key)
            if sec_link:
                st.markdown(f'**Source:** <a href="{sec_link}" target="_blank">View on SEC EDGAR</a>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**Relevant Excerpts Found:**")
            for snippet in snippets:
                st.markdown(f"> _{snippet}_")

            # Advanced Options
            st.markdown("---")
            st.markdown("##### Advanced Analysis")
            col1, col2, col3 = st.columns(3)

            if col1.button("🤖 Generate AI Summary", key=f"summary_{doc_key}"):
                st.session_state[view_state_key] = 'summary' if st.session_state[view_state_key] != 'summary' else None
            if col2.button("📖 View Complete Content", key=f"full_{doc_key}"):
                st.session_state[view_state_key] = 'content' if st.session_state[view_state_key] != 'content' else None
            if col3.button("🎯 Extract Key Paragraphs", key=f"para_{doc_key}"):
                st.session_state[view_state_key] = 'paragraphs' if st.session_state[view_state_key] != 'paragraphs' else None
            
            active_view = st.session_state.get(view_state_key)
            if active_view:
                with st.spinner("Fetching and processing full document from the data hub..."):
                    content = get_filing_content_from_hub(ticker_from_key, filename_for_download)

                    if not content:
                        st.error("Could not retrieve the full document content.")
                    else:
                        if active_view == 'summary':
                            soup = BeautifulSoup(content, 'html.parser')
                            clean_content = soup.get_text(strip=True)[:20000] # Truncate for performance
                            summary_prompt = f"Please provide a concise, professional executive summary of the following financial document content. Focus on financial performance, key business segments, risk factors, and future outlook. Use bullet points.\n\nDOCUMENT CONTENT:\n\n{clean_content}"
                            response = qual_brain.groq_client.chat.completions.create(model=GROQ_LLM_MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0.2)
                            ai_summary = response.choices[0].message.content
                            st.markdown(f'<div style="background-color: #1E293B; padding: 20px; border-radius: 8px;">{ai_summary}</div>', unsafe_allow_html=True)

                        elif active_view == 'content':
                            soup = BeautifulSoup(content, 'html.parser')
                            clean_content = soup.get_text(separator='\n', strip=True)
                            st.code(clean_content, language=None)
                            
                        elif active_view == 'paragraphs':
                            relevant_paragraphs = extract_relevant_paragraphs(content, query_keywords)
                            if relevant_paragraphs:
                                for para in relevant_paragraphs:
                                    st.info(para)
                            else:
                                st.info("💡 No highly relevant paragraphs found based on your query.")
# --- Main Dashboard ---
st.markdown(f'<h1 class="main-header">AI Corporate Intelligence: {selected_company}</h1>', unsafe_allow_html=True)

if market_data.empty and news_data.empty:
     st.warning(f"No data found for {selected_company}. Please run data collection and processing scripts.")

if not market_data.empty:
    analysis = run_ai_analysis(selected_company, market_data, news_data, financials_data)
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
tab_labels = ["📈 Market Analysis", "🤖 AI Analyst Chat", "📰 News Analysis", "💡 Deep Dive"]
selected_tab = st.tabs(tab_labels)
tab_market, tab_chat, tab_news, tab_deep = selected_tab[0], selected_tab[1], selected_tab[2], selected_tab[3]

st.markdown("---")
st.markdown(f"""
<div style="padding: 15px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 8px; margin: 20px 0;">
<h3 style="color: white; margin: 0; text-align: center;">🚀 Ready to Analyze {selected_company}?</h3>
<p style="color: white; margin: 10px 0 0 0; text-align: center; opacity: 0.9;">
Use the tabs above to explore different insights. The <strong>AI Analyst Chat</strong> tab provides detailed financial analysis with direct source access.
</p>
</div>
""", unsafe_allow_html=True)

with tab_market:
    if not market_data.empty:
        st.header("📈 Market Performance Analysis")
        st.markdown("*Real-time market data and performance metrics*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            latest = market_data.iloc[-1]
            st.metric("Current Price", f"${latest['Close']:.2f}", f"{latest['Close'] - market_data.iloc[-2]['Close']:+.2f}")
        with col2: st.metric("Day Range", f"${latest['Low']:.2f} - ${latest['High']:.2f}")
        with col3: st.metric("Volume", f"{latest['Volume']:,}")
        
        fig = go.Figure(data=[go.Candlestick(x=market_data['Date'], open=market_data['Open'], high=market_data['High'], low=market_data['Low'], close=market_data['Close'], name="Price")])
        fig.update_layout(title=f"{selected_company} Stock Price", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No market data available. Please run the data collection script.")

with tab_chat:
    st.header("🤖 AI Financial Analyst")
    st.markdown("*Your intelligent companion for financial analysis and SEC filing insights*")
    
    st.markdown("""
    <div style="padding: 20px; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); border-radius: 10px; margin: 20px 0; border: 2px solid #3498db; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 25px;">
        <div style="text-align: center; padding: 15px; background: rgba(52, 152, 219, 0.2); border-radius: 8px; border: 1px solid #3498db;"><div style="font-size: 24px; margin-bottom: 8px;">📊</div><strong style="color: #ecf0f1; font-size: 16px;">Financial Metrics</strong><br><small style="color: #bdc3c7;">Ask about specific numbers</small></div>
        <div style="text-align: center; padding: 15px; background: rgba(46, 204, 113, 0.2); border-radius: 8px; border: 1px solid #2ecc71;"><div style="font-size: 24px; margin-bottom: 8px;">📄</div><strong style="color: #ecf0f1; font-size: 16px;">SEC Filings</strong><br><small style="color: #bdc3c7;">Get direct EDGAR links</small></div>
        <div style="text-align: center; padding: 15px; background: rgba(155, 89, 182, 0.2); border-radius: 8px; border: 1px solid #9b59b6;"><div style="font-size: 24px; margin-bottom: 8px;">📈</div><strong style="color: #ecf0f1; font-size: 16px;">Smart Analysis</strong><br><small style="color: #bdc3c7;">Direct source paragraphs</small></div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    session_key = f"messages_{selected_company}"
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    
    # Display chat messages from history
    for message in st.session_state[session_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                display_enhanced_sources(message["sources"], message.get("prompt", ""))

    # Chat input
    chat_key = f"chat_input_{selected_company}"
    prompt = st.chat_input(f"Ask about {selected_company}'s financials...", key=chat_key)
    
    if prompt:
        st.session_state[session_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing financial data and SEC filings..."):
                try:
                    response, sources = query_rag_system(selected_company, prompt)
                    st.markdown(response)
                    if sources:
                        display_enhanced_sources(sources, prompt)
                    
                    st.session_state[session_key].append({
                        "role": "assistant", "content": response, 
                        "sources": sources, "prompt": prompt
                    })
                except Exception as e:
                    error_response = f"Sorry, I couldn't complete the analysis: {e}"
                    st.error(error_response)
                    st.session_state[session_key].append({"role": "assistant", "content": error_response, "sources": []})

with tab_news:
    st.header("📰 Recent Company News")
    st.markdown("*Latest news and sentiment analysis for market insights*")
    
    if not news_data.empty:
        display_df = news_data.copy()
        
        # --- News Metrics (Unchanged) ---
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Articles", len(display_df))
        with col2:
            latest_news = pd.to_datetime(display_df['publishedAt']).max() if 'publishedAt' in display_df else None
            if latest_news: st.metric("Latest News", latest_news.strftime('%Y-%m-%d'))
        with col3: st.metric("News Sources", display_df['source.name'].nunique() if 'source.name' in display_df else 0)
        
        st.markdown("---")

        # --- New Two-Column News Card Layout ---
        for i, article in display_df.head(5).iterrows():
            title = article.get('title', 'Untitled')
            url = article.get('url', '')
            image_url = article.get('urlToImage')
            source = article.get('source.name', 'Unknown')
            published_str = pd.to_datetime(article.get('publishedAt')).strftime('%Y-%m-%d %H:%M') if pd.notna(article.get('publishedAt')) else ''
            description = article.get('description', 'No description available.')

            st.markdown(f"""
            <div style="background-color: #262730; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #333;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    {f'<img src="{image_url}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 8px;">' if image_url else ''}
                    <div style="flex: 1;">
                        <p style="color: #aaa; margin: 0; font-size: 0.8em;">{source.upper()} | {published_str}</p>
                        <h4 style="margin: 5px 0;"><a href="{url}" target="_blank" style="color: #e0e0e0; text-decoration: none;">{title}</a></h4>
                        <p style="color: #ccc; margin: 0; font-size: 0.9em;">{description}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📰 No news data available. Please run the data collection script.")
with tab_deep:
    st.header("💡 AI-Powered Deep Dive Analysis")
    st.markdown("*Comprehensive AI-generated insights and executive summary*")
    
    if not market_data.empty:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("AI Health Score", f"{health_score:.1f}/100", "🟢 Strong" if health_score >= 70 else "🟡 Moderate" if health_score >= 50 else "🔴 Weak")
        with col2: st.metric("AI Forecast", prediction, f"{confidence:.1%} confidence")
        with col3:
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("News Sentiment", sentiment_label, f"{avg_sentiment:.2f}")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Executive Summary")
        
        with st.spinner("🤖 AI is generating an executive summary..."):
            summary_prompt = f"Generate a comprehensive executive summary for {selected_company} based on: Health Score: {health_score:.1f}/100, Forecast: {prediction} ({confidence:.1%} confidence), and News Sentiment: {avg_sentiment:.2f}. Please provide key financial highlights, market position, risk factors, strategic recommendations, and future outlook in clear sections."
            try:
                from src.utils.config import GROQ_LLM_MODEL
                response = qual_brain.groq_client.chat.completions.create(model=GROQ_LLM_MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0.3)
                ai_summary = response.choices[0].message.content
                st.markdown(f'<div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 20px 0;"><h4 style="margin: 0 0 15px 0; text-align: center;">🎯 AI Analysis Results</h4><div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; line-height: 1.6;">{ai_summary}</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Could not generate AI summary: {e}")

        st.markdown("---")
        st.markdown("### 🛠️ Advanced Analysis Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Financial Ratios Analysis", key="financial_ratios"):
                with st.spinner("Calculating financial ratios..."):
                    if financials_data.empty:
                        st.warning("❌ No financial data available to calculate ratios.")
                    else:
                        try:
                            # Unpack the ratios DataFrame AND the list of missing columns
                            ratios_df, missing_cols = calculate_all_ratios(financials_data)
                            
                            latest_valid_row = ratios_df.dropna(how='all', subset=ratios_df.columns.drop('Date')).head(1)
                            
                            if not latest_valid_row.empty:
                                st.success("✅ Financial Ratios Calculated Successfully!")
                                latest_date = latest_valid_row['Date'].iloc[0]
                                st.markdown(f"#### Latest Available Financial Ratios (as of {pd.to_datetime(latest_date).strftime('%Y-%m-%d')})")
                                
                                display_df = latest_valid_row.set_index('Date').T
                                st.dataframe(display_df.style.format("{:.2f}", na_rep="N/A"))

                                # If there were missing columns, display a warning to the user
                                if missing_cols:
                                    st.warning(f"**Data Quality Alert:** Some ratios could not be calculated because the following data points were missing from the source financial statements: **{', '.join(missing_cols)}**")

                                st.markdown("#### Ratio Trends Over Time")
                                st.line_chart(ratios_df.set_index('Date')[['DebtToEquity', 'NetProfitMargin', 'ReturnOnEquity']])
                            else:
                                st.error("❌ Could not calculate any valid ratios from the available data.")
                        except Exception as e:
                            st.error(f"An error occurred during ratio calculation: {e}")

        with col2:
            if st.button("📈 Generate Peer Comparison", key="peer_comparison"):
                st.info("💡 Peer comparison feature coming soon!")
        
        st.markdown("---")
        st.markdown("### ⚠️ Risk Assessment Dashboard")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            market_risk = "High" if confidence < 0.6 else "Medium" if confidence < 0.8 else "Low"
            risk_color = "#dc3545" if market_risk == "High" else "#ffc107" if market_risk == "Medium" else "#28a745"
            st.markdown(f'<div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;"><h4 style="color: {risk_color}; margin: 0;">📊 Market Risk</h4><p style="margin: 5px 0 0 0; font-weight: bold;">{market_risk}</p><small>Based on prediction confidence</small></div>', unsafe_allow_html=True)
        with risk_col2:
            health_risk = "Low" if health_score >= 70 else "Medium" if health_score >= 50 else "High"
            risk_color = "#28a745" if health_risk == "Low" else "#ffc107" if health_risk == "Medium" else "#dc3545"
            st.markdown(f'<div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;"><h4 style="color: {risk_color}; margin: 0;">💰 Financial Risk</h4><p style="margin: 5px 0 0 0; font-weight: bold;">{health_risk}</p><small>Based on health score</small></div>', unsafe_allow_html=True)
        with risk_col3:
            sentiment_risk = "Low" if avg_sentiment > 0.1 else "High" if avg_sentiment < -0.1 else "Medium"
            risk_color = "#28a745" if sentiment_risk == "Low" else "#ffc107" if sentiment_risk == "Medium" else "#dc3545"
            st.markdown(f'<div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;"><h4 style="color: {risk_color}; margin: 0;">📰 Sentiment Risk</h4><p style="margin: 5px 0 0 0; font-weight: bold;">{sentiment_risk}</p><small>Based on news sentiment</small></div>', unsafe_allow_html=True)
    else:
        st.info("📊 Not enough data available to generate a deep dive analysis.")

# --- Footer ---













