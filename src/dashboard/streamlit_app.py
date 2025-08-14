# src/dashboard/streamlit_app.py
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
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to load CSV {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_json_from_hub(repo_id, filename):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return pd.read_json(file_path)
    except Exception as e:
        logging.error(f"Failed to load JSON {filename}: {e}")
        return pd.DataFrame()

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
    Enhanced source display with direct content access, full-width views,
    and AI-powered analysis capabilities.
    """
    st.markdown("---")
    st.markdown("""
    <div style="padding: 15px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); border-radius: 8px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; text-align: center;">📚 Sources & Direct Access</h3>
    <p style="color: white; margin: 10px 0 0 0; text-align: center; opacity: 0.9;">
    Direct links, full content, and AI-powered analysis of official sources
    </p>
    </div>
    """, unsafe_allow_html=True)

    doc_sources = {}
    for s in sources:
        doc_key = s.metadata.get("source", "Unknown")
        if doc_key not in doc_sources:
            doc_sources[doc_key] = []
        doc_sources[doc_key].append(s)

    query_keywords = prompt.lower().split() if prompt else []

    for doc_path, doc_refs in doc_sources.items():
        if doc_path != "Unknown":
            # Initialize a session state for each document expander to manage views
            view_state_key = f'active_view_{hash(doc_path)}'
            st.session_state.setdefault(view_state_key, None)

            filename = os.path.basename(doc_path)
            doc_type = "SEC Filing" if doc_path.endswith('.html') else "Financial Data"
            
            with st.expander(f"📄 {filename} - {doc_type}", expanded=True):
                # Header Information (SEC Link, etc.)
                filing_type = None
                if doc_path.endswith('.html'):
                    if '10-k' in filename.lower(): filing_type = "10-K Annual Report"
                    elif '10-q' in filename.lower(): filing_type = "10-Q Quarterly Report"
                    elif '8-k' in filename.lower(): filing_type = "8-K Current Report"
                sec_link = get_sec_link(filename, selected_company)
                if filing_type:
                    st.markdown(f'<div style="padding: 12px; background-color: #007bff; color: white; border-radius: 5px; margin-bottom: 15px;"><strong>📋 {filing_type.upper()}</strong></div>', unsafe_allow_html=True)
                if sec_link:
                    st.markdown(f'<div style="padding: 15px; background-color: #e3f2fd; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;"><h4 style="margin: 0 0 10px 0; color: #1976d2;">🔗 Direct SEC EDGAR Link</h4><a href="{sec_link}" target="_blank" style="color: #1976d2; text-decoration: none; font-weight: bold;">📊 View Official Filing on SEC.gov</a></div>', unsafe_allow_html=True)

                # Initial Relevant Snippets
                st.markdown("### 📝 Relevant Content from This Document")
                for i, ref in enumerate(doc_refs, 1):
                    snippet = ref.page_content
                    st.markdown(f"""
                    <div style="border-left: 4px solid #007bff; padding-left: 15px; margin: 15px 0; background-color: rgba(0, 123, 255, 0.05); border-radius: 5px;">
                        <h5 style="color: #00aaff; margin-top: 5px; margin-bottom: 10px;">📍 Reference {i}</h5>
                        <p style="margin: 0; line-height: 1.6; color: #e0e0e0; font-size: 16px;">{snippet}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Advanced Options Button Bar
                st.markdown("---")
                st.markdown("### 📁 Advanced Analysis & Access Options")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"🤖 Generate AI Summary", key=f"summary_{hash(doc_path)}"):
                        st.session_state[view_state_key] = 'summary' if st.session_state[view_state_key] != 'summary' else None
                with col2:
                    if st.button(f"📖 Extract Key Paragraphs", key=f"paragraphs_{hash(doc_path)}"):
                        st.session_state[view_state_key] = 'paragraphs' if st.session_state[view_state_key] != 'paragraphs' else None
                with col3:
                    if st.button(f"📄 View Complete Content", key=f"fullcontent_{hash(doc_path)}"):
                        st.session_state[view_state_key] = 'content' if st.session_state[view_state_key] != 'content' else None
                with col4:
                    if os.path.exists(doc_path):
                        with open(doc_path, 'rb') as f: file_data = f.read()
                        st.download_button(label="💾 Download Full File", data=file_data, file_name=filename, mime="text/html", key=f"download_{hash(doc_path)}")

                # Full-Width Display Area based on Session State
                active_view = st.session_state.get(view_state_key)


                if active_view == 'summary':
                    with st.spinner("🤖 AI is reading and summarizing the document..."):
                        if os.path.exists(doc_path):
                            with open(doc_path, 'r', encoding='utf-8') as f: content = f.read()
                            soup = BeautifulSoup(content, 'html.parser')
                            clean_content = soup.get_text(strip=True)

                            # Truncate content to fit model context window if necessary
                            max_chars = 20000 # Approx 7k-8k tokens for Llama3 8b
                            if len(clean_content) > max_chars:
                                st.warning(f"⚠️ Document is very long. Summarizing the first {max_chars} characters.")
                                clean_content = clean_content[:max_chars]

                            summary_prompt = f"Please provide a concise, professional executive summary of the following financial document content. Focus on the most critical information, such as financial performance, key business segments, risk factors, and future outlook. Use bullet points for clarity.\n\nDOCUMENT CONTENT:\n\n{clean_content}"
                            
                            try:
                                response = qual_brain.groq_client.chat.completions.create(
                                    model=GROQ_LLM_MODEL,
                                    messages=[{"role": "user", "content": summary_prompt}],
                                    temperature=0.2
                                )
                                ai_summary = response.choices[0].message.content
                                st.markdown("#### 🤖 AI-Generated Executive Summary")
                                st.markdown(f'<div style="background-color: #1E293B; padding: 20px; border-radius: 8px; border: 1px solid #334155;">{ai_summary}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Could not generate AI summary: {e}")
                        else:
                            st.error("❌ File not found")


                # --- Full-Width Display Area ---
                elif active_view == 'paragraphs':
                    with st.spinner("Extracting and cleaning relevant paragraphs..."):
                        if os.path.exists(doc_path):
                            with open(doc_path, 'r', encoding='utf-8') as f: content = f.read()
                            relevant_paragraphs = extract_relevant_paragraphs(content, query_keywords)
                            if relevant_paragraphs:
                                st.markdown("**🎯 Most Relevant Paragraphs:**")
                                for j, para in enumerate(relevant_paragraphs, 1):
                                    st.markdown(f"""
                                    <div style="padding: 15px; background-color: rgba(255, 193, 7, 0.1); color: #e0e0e0; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107;">
                                        <strong style="color: #ffc107;">Paragraph {j}:</strong>
                                        <div style="line-height: 1.6; margin-top: 8px;">{para}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else: st.info("💡 No highly relevant paragraphs found.")
                        else: st.error("❌ File not found")
                
                elif active_view == 'content':
                    if os.path.exists(doc_path):
                        with st.spinner("Loading and cleaning full document..."):
                    
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                            
                            # Use BeautifulSoup to get a clean text representation
                            soup = BeautifulSoup(full_content, 'html.parser')
                            clean_content = soup.get_text(separator='\n', strip=True)
                            
                            st.markdown("### 📄 Complete File Content (Raw Text)")

                            # Use st.markdown with a styled <pre> tag for a code-block style view
                            st.markdown(f"""
                            <div style="background-color: #1a1a2e; 
                                         border: 1px solid #3a3a5e; 
                                         border-radius: 8px; 
                                         padding: 15px; 
                                         height: 500px; 
                                         overflow-y: scroll; 
                                         font-family: 'Courier New', Courier, monospace; 
                                         color: #e0e0e0;
                                         font-size: 0.9em;">
                                <pre style="white-space: pre-wrap; margin: 0; word-wrap: break-word;">{clean_content}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ File not found")
                        
                # --- File Details ---
                if os.path.exists(doc_path):
                    file_size = os.path.getsize(doc_path)
                    file_modified = os.path.getmtime(doc_path)
                    st.markdown(f"""
                    <div style="padding: 12px; background-color: rgba(108, 117, 125, 0.1); border-radius: 5px; margin: 15px 0; border: 1px solid #444;">
                        <strong>ℹ️ File Details:</strong><br>
                        <span style="color: #ccc;">📁 Path:</span> <code>{doc_path}</code><br>
                        <span style="color: #ccc;">📏 Size:</span> <span style="color: #e0e0e0;">{file_size:,} bytes</span><br>
                        <span style="color: #ccc;">🕐 Last Modified:</span> <span style="color: #e0e0e0;">{pd.to_datetime(file_modified, unit='s').strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    """, unsafe_allow_html=True)

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



