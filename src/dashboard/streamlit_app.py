# src/dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import sys
import logging
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH
from src.rag.query_processor import query_rag_system
from src.ai_core.qualitative_brain import QualitativeBrain
from src.ai_core.quantitative_brain import QuantitativeBrain
from src.data_collection.financials import fetch_financials_dataframe
from src.utils.financial_ratios import calculate_all_ratios

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
def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def load_company_data(ticker, market_sig: float, news_sig: float, fin_sig: float):
    market_file = os.path.join(MARKET_DATA_PATH, f'{ticker}_market_data.csv')
    news_file = os.path.join(NEWS_DATA_PATH, f'{ticker}_news.json')
    market_df = pd.read_csv(market_file) if os.path.exists(market_file) else pd.DataFrame()
    if not market_df.empty:
        market_df['Date'] = pd.to_datetime(market_df['Date'], utc=True, errors='coerce').dt.tz_convert(None)
    # Normalize news JSON to a flat dataframe to avoid caching hash issues
    news_df = pd.read_json(news_file) if os.path.exists(news_file) else pd.DataFrame()
    if not news_df.empty and 'articles' in news_df:
        try:
            articles_df = pd.json_normalize(news_df['articles'])
            if 'publishedAt' in articles_df:
                articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'], utc=True, errors='coerce').dt.tz_convert(None)
            news_df = articles_df
        except Exception as e:
            logging.error(f"Error normalizing news data: {e}")
    # Fetch financials for health scoring
    financials_df = fetch_financials_dataframe(ticker)
    return market_df, news_df, financials_df

_market_file = os.path.join(MARKET_DATA_PATH, f'{selected_company}_market_data.csv')
_news_file = os.path.join(NEWS_DATA_PATH, f'{selected_company}_news.json')
_fin_file = os.path.join(os.environ.get('PROCESSED_DATA_PATH', 'data/processed'), selected_company, f'{selected_company}_financials_quarterly.csv')

market_data, news_data, financials_data = load_company_data(
    selected_company,
    _file_mtime(_market_file),
    _file_mtime(_news_file),
    _file_mtime(_fin_file),
)

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
    # Company CIK mapping
    company_ciks = {
        'AAPL': '0000320193',
        'MSFT': '0000789019',
        'GOOGL': '0001652044',
        'AMZN': '0001018724',
        'TSLA': '0001318605',
        'META': '0001326801',
        'NVDA': '0001045810',
        'BRK.B': '0001067983',
        'JNJ': '0000200406',
        'V': '0001403161',
        # Add more as needed
    }
    
    # Extract accession number from filename
    accession_match = re.search(r'(\d{10}-\d{2}-\d{6})', filename)
    if accession_match and selected_company in company_ciks:
        cik = company_ciks.get(selected_company)
        accession = accession_match.group(1)
        if cik:
            return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"
    return None

def extract_relevant_paragraphs(content, query_keywords, max_paragraphs=3):
    """Extract paragraphs most relevant to the query"""
    if not query_keywords:
        return []
    
    # Split content into paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    # Score paragraphs based on keyword matches
    scored_paragraphs = []
    for para in paragraphs:
        score = sum(1 for keyword in query_keywords if keyword.lower() in para.lower())
        if score > 0:
            scored_paragraphs.append((score, para))
    
    # Sort by relevance and return top paragraphs
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    return [para for score, para in scored_paragraphs[:max_paragraphs]]

# In streamlit_app.py, replace the entire function with this one

def display_enhanced_sources(sources, prompt=""):
    """Enhanced source display with direct content access"""
    
    st.markdown("---")
    st.markdown("""
    <div style="padding: 15px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); border-radius: 8px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; text-align: center;">📚 Sources & Direct Access</h3>
    <p style="color: white; margin: 10px 0 0 0; text-align: center; opacity: 0.9;">
    Direct links, full content, and relevant paragraphs from official sources
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Group sources by document
    doc_sources = {}
    for s in sources:
        # CORRECTED: Access the 'metadata' dictionary
        doc_key = s.metadata.get("source", "Unknown")
        if doc_key not in doc_sources:
            doc_sources[doc_key] = []
        doc_sources[doc_key].append(s)
    
    query_keywords = prompt.lower().split() if prompt else []
    
    for doc_path, doc_refs in doc_sources.items():
        if doc_path != "Unknown":
            filename = os.path.basename(doc_path)
            doc_type = "SEC Filing" if doc_path.endswith('.html') else "Financial Data"
            
            # (The rest of the function for determining filing type and SEC links remains the same)
            filing_type = None
            if doc_path.endswith('.html'):
                if '10-k' in filename.lower(): filing_type = "10-K Annual Report"
                elif '10-q' in filename.lower(): filing_type = "10-Q Quarterly Report"
                elif '8-k' in filename.lower(): filing_type = "8-K Current Report"
                elif 'def 14a' in filename.lower(): filing_type = "DEF 14A Proxy Statement"
            
            sec_link = get_sec_link(filename, selected_company) if doc_type == "SEC Filing" else None
            
            with st.expander(f"📄 {filename} - {doc_type}", expanded=False):
                if filing_type:
                    st.markdown(f'<div style="padding: 12px; background-color: #007bff; color: white; border-radius: 5px; margin-bottom: 15px;"><strong>📋 {filing_type.upper()}</strong><br><small>Official SEC Filing for {selected_company}</small></div>', unsafe_allow_html=True)
                
                if sec_link:
                    st.markdown(f'<div style="padding: 15px; background-color: #e3f2fd; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;"><h4 style="margin: 0 0 10px 0; color: #1976d2;">🔗 Direct SEC EDGAR Link</h4><a href="{sec_link}" target="_blank" style="color: #1976d2; text-decoration: none; font-weight: bold; font-size: 16px;">📊 View Official Filing on SEC.gov</a><br><small style="color: #666; margin-top: 5px; display: block;">Click to access the original document on the SEC website</small></div>', unsafe_allow_html=True)
                
                st.markdown("### 📝 Relevant Content from This Document")
                for i, ref in enumerate(doc_refs, 1):
                    # CORRECTED: Get the main text from the 'page_content' attribute
                    snippet = ref.page_content
                    
                    if snippet and snippet.strip():
                        st.markdown(f'<div style="padding: 20px; background-color: #f8f9fa; border-left: 4px solid #007bff; border-radius: 5px; margin: 15px 0; border: 1px solid #e9ecef;"><h5 style="color: #007bff; margin-top: 0;">📍 Reference {i}</h5><div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0;"><p style="margin: 0; line-height: 1.6; color: #495057; font-size: 16px;">{snippet}</p></div></div>', unsafe_allow_html=True)
                        
                        context_info = []
                        # CORRECTED: Access metadata for all context info
                        if ref.metadata.get("section"):
                            context_info.append(f'📂 Section: {ref.metadata.get("section")}')
                        if ref.metadata.get("page"):
                            context_info.append(f'📄 Page: {ref.metadata.get("page")}')
                        if ref.metadata.get("source_id"):
                            context_info.append(f'🏷️ Source ID: {ref.metadata.get("source_id")}')
                        
                        if context_info:
                            st.markdown(f"<small style='color: #6c757d; font-style: italic;'>{' | '.join(context_info)}</small>", unsafe_allow_html=True)
                
                # File access options
                st.markdown("---")
                st.markdown("### 📁 Advanced Access Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # View specific paragraphs
                    if st.button(f"📖 Extract Key Paragraphs", key=f"paragraphs_{hash(doc_path)}"):
                        if os.path.exists(doc_path):
                            try:
                                with open(doc_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                relevant_paragraphs = extract_relevant_paragraphs(content, query_keywords)
                                
                                if relevant_paragraphs:
                                    st.markdown("**🎯 Most Relevant Paragraphs:**")
                                    for j, para in enumerate(relevant_paragraphs, 1):
                                        # Clean HTML tags if present
                                        clean_para = re.sub(r'<[^>]+>', '', para)
                                        st.markdown(f"""
                                        <div style="padding: 15px; background-color: #fff3cd; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107;">
                                        <strong>Paragraph {j}:</strong><br><br>
                                        <div style="line-height: 1.6;">{clean_para[:800]}{'...' if len(clean_para) > 800 else ''}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("💡 No highly relevant paragraphs found. Try refining your query with specific financial terms.")
                            except Exception as e:
                                st.error(f"❌ Could not extract paragraphs: {e}")
                        else:
                            st.error("❌ File not found")
                
                with col2:
                    # Download file
                    if os.path.exists(doc_path):
                        try:
                            with open(doc_path, 'rb') as f:
                                file_data = f.read()
                            
                            st.download_button(
                                label="💾 Download Full File",
                                data=file_data,
                                file_name=filename,
                                mime="text/html" if doc_path.endswith('.html') else "text/plain",
                                key=f"download_{hash(doc_path)}"
                            )
                        except Exception as e:
                            st.error(f"❌ Could not prepare download: {e}")
                    else:
                        st.error("❌ File not available for download")
                
                with col3:
                    # Show full content in expandable section
                    if st.button(f"📄 View Complete Content", key=f"fullcontent_{hash(doc_path)}"):
                        if os.path.exists(doc_path):
                            try:
                                with open(doc_path, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                                
                                st.markdown("**📄 Complete File Content:**")
                                # Clean content for better display
                                if doc_path.endswith('.html'):
                                    # Remove HTML tags for cleaner reading
                                    clean_content = re.sub(r'<[^>]+>', '', full_content)
                                    clean_content = re.sub(r'\n\s*\n', '\n\n', clean_content)
                                else:
                                    clean_content = full_content
                                
                                st.text_area(
                                    f"Content of {filename}:",
                                    clean_content,
                                    height=400,
                                    key=f"content_area_{hash(doc_path)}"
                                )
                            except Exception as e:
                                st.error(f"❌ Could not read file: {e}")
                        else:
                            st.error("❌ File not found")
                
                # File information
                if os.path.exists(doc_path):
                    file_size = os.path.getsize(doc_path)
                    file_modified = os.path.getmtime(doc_path)
                    st.markdown(f"""
                    <div style="padding: 12px; background-color: #e9ecef; border-radius: 5px; margin: 15px 0;">
                    <strong>ℹ️ File Details:</strong><br>
                    📁 Path: <code>{doc_path}</code><br>
                    📏 Size: {file_size:,} bytes<br>
                    🕐 Last Modified: {pd.to_datetime(file_modified, unit='s').strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Document structure for HTML files
                if doc_path.endswith('.html') and os.path.exists(doc_path):
                    with st.expander("🏗️ Document Structure Analysis", expanded=False):
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Extract headings
                            headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', content, re.IGNORECASE | re.DOTALL)
                            if headings:
                                st.markdown("**📋 Document Sections:**")
                                for j, heading in enumerate(headings[:15], 1):  # Show first 15 headings
                                    clean_heading = re.sub(r'<[^>]+>', '', heading).strip()
                                    if clean_heading and len(clean_heading) > 3:
                                        st.markdown(f"{j}. {clean_heading}")
                            else:
                                # Try to find other structural elements
                                tables = len(re.findall(r'<table', content, re.IGNORECASE))
                                divs = len(re.findall(r'<div', content, re.IGNORECASE))
                                st.markdown(f"**📊 Document Elements:**")
                                st.markdown(f"• Tables found: {tables}")
                                st.markdown(f"• Div sections: {divs}")
                        except Exception as e:
                            st.error(f"Could not analyze document structure: {e}")

# --- Main Dashboard ---
st.markdown(f'<h1 class="main-header">AI Corporate Intelligence: {selected_company}</h1>', unsafe_allow_html=True)

if market_data.empty and news_data.empty:
     st.warning(f"No data found for {selected_company}. Please run the collection and processing scripts first using 'python main.py collect' and 'python main.py process'.")

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
tab_market = selected_tab[0]
tab_chat = selected_tab[1] 
tab_news = selected_tab[2]
tab_deep = selected_tab[3]

# Navigation section
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
        
        # Market overview cards
        col1, col2, col3 = st.columns(3)
        with col1:
            latest = market_data.iloc[-1]
            st.metric("Current Price", f"${latest['Close']:.2f}", f"{latest['Close'] - market_data.iloc[-2]['Close']:+.2f}")
        with col2:
            st.metric("Day Range", f"${latest['Low']:.2f} - ${latest['High']:.2f}")
        with col3:
            st.metric("Volume", f"{latest['Volume']:,}")
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=market_data['Date'], 
            open=market_data['Open'], 
            high=market_data['High'], 
            low=market_data['Low'], 
            close=market_data['Close'], 
            name="Price"
        )])
        fig.update_layout(
            title=f"{selected_company} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No market data available. Please run the data collection script.")

with tab_chat:
    # Clean chat header
    st.header("🤖 AI Financial Analyst")
    st.markdown("*Your intelligent companion for financial analysis and SEC filing insights*")
    
    # Enhanced pro tips
    st.markdown("""
    <div style="padding: 20px; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); border-radius: 10px; margin: 20px 0; border: 2px solid #3498db; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 25px;">
        <div style="text-align: center; padding: 15px; background: rgba(52, 152, 219, 0.2); border-radius: 8px; border: 1px solid #3498db;">
            <div style="font-size: 24px; margin-bottom: 8px;">📊</div>
            <strong style="color: #ecf0f1; font-size: 16px;">Financial Metrics</strong><br>
            <small style="color: #bdc3c7;">Ask about specific numbers</small>
        </div>
        <div style="text-align: center; padding: 15px; background: rgba(46, 204, 113, 0.2); border-radius: 8px; border: 1px solid #2ecc71;">
            <div style="font-size: 24px; margin-bottom: 8px;">📄</div>
            <strong style="color: #ecf0f1; font-size: 16px;">SEC Filings</strong><br>
            <small style="color: #bdc3c7;">Get direct EDGAR links</small>
        </div>
        <div style="text-align: center; padding: 15px; background: rgba(155, 89, 182, 0.2); border-radius: 8px; border: 1px solid #9b59b6;">
            <div style="font-size: 24px; margin-bottom: 8px;">📈</div>
            <strong style="color: #ecf0f1; font-size: 16px;">Smart Analysis</strong><br>
            <small style="color: #bdc3c7;">Direct source paragraphs</small>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    session_key = f"messages_{selected_company}"
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    
    # Display chat messages with better styling
    if st.session_state[session_key]:
        st.markdown("---")
        st.markdown("**💬 Conversation History**")
        for message in st.session_state[session_key]:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])
    
    # Chat input with unique key per company
    chat_key = f"chat_input_{selected_company}"
    prompt = st.chat_input(f"Ask about {selected_company}'s financials, SEC filings, or market performance...", key=chat_key)
    
    if prompt:
        st.session_state[session_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing financial data and SEC filings..."):
                try:
                    response, sources = query_rag_system(selected_company, prompt)
                except Exception as e:
                    response, sources = (f"Sorry, I couldn't complete the analysis: {e}", [])
                
                st.markdown(response)
                
                # Enhanced source display
                if sources:
                    display_enhanced_sources(sources, prompt)
        
        # Persist assistant response
        st.session_state[session_key].append({"role": "assistant", "content": response})

with tab_news:
    st.header("📰 Recent Company News")
    st.markdown("*Latest news and sentiment analysis for market insights*")
    
    if not news_data.empty:
        display_df = news_data.copy()
        if 'publishedAt' in display_df:
            display_df['publishedAt'] = pd.to_datetime(display_df['publishedAt'], errors='coerce')
        
        # News summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(display_df))
        with col2:
            latest_news = display_df['publishedAt'].max() if 'publishedAt' in display_df else None
            if latest_news:
                st.metric("Latest News", latest_news.strftime('%Y-%m-%d'))
        with col3:
            st.metric("News Sources", display_df['source.name'].nunique() if 'source.name' in display_df else 0)
        
        # News articles with better styling
        for i, (_, article) in enumerate(display_df.head(5).iterrows()):
            title = article.get('title', 'Untitled')
            url = article.get('url', '')
            source = article.get('source.name', article.get('source', 'Unknown'))
            published = article.get('publishedAt')
            published_str = pd.to_datetime(published, errors='coerce').strftime('%Y-%m-%d') if pd.notna(published) else ''
            description = article.get('description', '')
            
            with st.container():
                st.markdown(f"""
                <div style="padding: 15px; border: 1px solid #e9ecef; border-radius: 8px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="margin: 0 0 10px 0;"><a href="{url}" target="_blank">{title}</a></h4>
                <p style="color: #6c757d; margin: 5px 0; font-size: 0.9em;">
                📰 <strong>{source}</strong> | 📅 {published_str}
                </p>
                """, unsafe_allow_html=True)
                
                if description:
                    st.markdown(f"<p style='margin: 10px 0 0 0;'>{description}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📰 No news data available. Please run the data collection script.")

with tab_deep:
    st.header("💡 AI-Powered Deep Dive Analysis")
    st.markdown("*Comprehensive AI-generated insights and executive summary*")
    
    if not market_data.empty:
        # Analysis overview cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI Health Score", f"{health_score:.1f}/100", 
                     "🟢 Strong" if health_score >= 70 else "🟡 Moderate" if health_score >= 50 else "🔴 Weak")
        with col2:
            st.metric("AI Forecast", prediction, f"{confidence:.1%} confidence")
        with col3:
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("News Sentiment", sentiment_label, f"{avg_sentiment:.2f}")
        
        # AI Executive Summary
        st.markdown("---")
        st.markdown("### 🤖 AI Executive Summary")
        
        with st.spinner("🤖 AI is analyzing company data and generating insights..."):
            summary_prompt = f"""
            Generate a comprehensive executive summary for {selected_company} based on:
            - Health Score: {health_score:.1f}/100
            - Forecast: {prediction} ({confidence:.1%} confidence)
            - News Sentiment: {avg_sentiment:.2f}
            
            Please provide:
            1. Key financial highlights
            2. Market position analysis
            3. Risk factors and concerns
            4. Strategic recommendations
            5. Future outlook
            
            Format the response in clear sections with bullet points where appropriate.
            """
            try:
                from src.utils.config import GROQ_LLM_MODEL
                response = qual_brain.groq_client.chat.completions.create(
                    model=GROQ_LLM_MODEL,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3
                )
                
                # Display the AI response in a styled container
                ai_summary = response.choices[0].message.content
                st.markdown(f"""
                <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 20px 0;">
                <h4 style="margin: 0 0 15px 0; text-align: center;">🎯 AI Analysis Results</h4>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; line-height: 1.6;">
                {ai_summary}
                </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ Could not generate AI summary: {e}")
                st.info("💡 Please ensure your GROQ API key is properly configured.")
        
        # Additional Analysis Tools
        st.markdown("---")
        st.markdown("### 🛠️ Advanced Analysis Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            # New code for streamlit_app.py
            if st.button("📊 Generate Financial Ratios Analysis", key="financial_ratios"):
                with st.spinner("Calculating financial ratios..."):
                    if financials_data.empty:
                        st.warning("❌ No financial data available to calculate ratios.")
                    else:
                        try:
                            ratios_df = calculate_all_ratios(financials_data)

                            # Find the latest row that has at least one valid, non-null ratio
                            latest_valid_row = ratios_df.dropna(
                                how='all', 
                                subset=ratios_df.columns.drop('Date')
                            ).head(1)

                            if not latest_valid_row.empty:
                                st.success("✅ Financial Ratios Calculated Successfully!")
                                
                                # Display the latest *available* ratios in a clean table
                                latest_date = latest_valid_row['Date'].iloc[0].strftime('%Y-%m-%d')
                                st.markdown(f"#### Latest Available Financial Ratios (as of {latest_date})")
                                
                                display_df = latest_valid_row.set_index('Date').T
                                st.dataframe(display_df.style.format("{:.2f}", na_rep="N/A"))

                                # Display a chart to show trends over time
                                st.markdown("#### Ratio Trends Over Time")
                                st.line_chart(ratios_df.set_index('Date')[['DebtToEquity', 'NetProfitMargin', 'ReturnOnEquity']])

                            else:
                                st.error("❌ Could not calculate any valid ratios from the available data.")

                        except Exception as e:
                            st.error(f"An error occurred during ratio calculation: {e}")
        with col2:
            if st.button("📈 Generate Peer Comparison", key="peer_comparison"):
                with st.spinner("Analyzing peer companies..."):
                    try:
                        # This would ideally compare with other companies in TARGET_COMPANIES
                        st.markdown("#### Peer Company Comparison")
                        
                        # Show comparison with other available companies
                        available_companies = [comp for comp in TARGET_COMPANIES if comp != selected_company]
                        
                        if available_companies:
                            st.markdown(f"**Comparing {selected_company} with:**")
                            for comp in available_companies[:5]:  # Show up to 5 peers
                                st.markdown(f"• {comp}")
                            
                            st.info("💡 Detailed peer analysis would compare financial metrics, market performance, and growth rates.")
                        else:
                            st.warning("No peer companies available for comparison.")
                            
                    except Exception as e:
                        st.error(f"Error generating peer comparison: {e}")
        
        # Risk Assessment Section
        st.markdown("---")
        st.markdown("### ⚠️ Risk Assessment Dashboard")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            # Market Risk
            market_risk = "High" if confidence < 0.6 else "Medium" if confidence < 0.8 else "Low"
            risk_color = "#dc3545" if market_risk == "High" else "#ffc107" if market_risk == "Medium" else "#28a745"
            st.markdown(f"""
            <div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;">
            <h4 style="color: {risk_color}; margin: 0;">📊 Market Risk</h4>
            <p style="margin: 5px 0 0 0; font-weight: bold;">{market_risk}</p>
            <small>Based on prediction confidence</small>
            </div>
            """, unsafe_allow_html=True)
        
        with risk_col2:
            # Financial Health Risk
            health_risk = "Low" if health_score >= 70 else "Medium" if health_score >= 50 else "High"
            risk_color = "#28a745" if health_risk == "Low" else "#ffc107" if health_risk == "Medium" else "#dc3545"
            st.markdown(f"""
            <div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;">
            <h4 style="color: {risk_color}; margin: 0;">💰 Financial Risk</h4>
            <p style="margin: 5px 0 0 0; font-weight: bold;">{health_risk}</p>
            <small>Based on health score</small>
            </div>
            """, unsafe_allow_html=True)
        
        with risk_col3:
            # Sentiment Risk
            sentiment_risk = "Low" if avg_sentiment > 0.1 else "High" if avg_sentiment < -0.1 else "Medium"
            risk_color = "#28a745" if sentiment_risk == "Low" else "#ffc107" if sentiment_risk == "Medium" else "#dc3545"
            st.markdown(f"""
            <div style="padding: 15px; background-color: {risk_color}20; border-left: 4px solid {risk_color}; border-radius: 5px;">
            <h4 style="color: {risk_color}; margin: 0;">📰 Sentiment Risk</h4>
            <p style="margin: 5px 0 0 0; font-weight: bold;">{sentiment_risk}</p>
            <small>Based on news sentiment</small>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.info("📊 Not enough data available to generate a deep dive analysis. Please run the data collection script first.")
        
        # Show data collection guidance
        st.markdown("""
        ### 🚀 Getting Started
        
        To enable full analysis capabilities:
        
        1. **Collect Market Data**: Run `python main.py collect` to gather market data
        2. **Process SEC Filings**: Run `python main.py process` to analyze SEC documents  
        3. **Schedule Updates**: Set up automated data collection for real-time insights
        
        Once data is available, you'll see:
        - 📈 Interactive market charts
        - 🤖 AI-powered financial analysis
        - 📄 Direct SEC filing access
        - 💡 Comprehensive risk assessment
        """)

# Footer with additional information
st.markdown("---")