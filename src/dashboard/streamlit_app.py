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
import hashlib
import time
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from pathlib import PurePath, Path
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH, GROQ_LLM_MODEL, HF_REPO_ID
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
def load_sec_dataset():
    """Load the SEC filings dataset from HuggingFace Hub."""
    try:
        dataset = load_dataset(HF_REPO_ID, split="train")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load SEC dataset: {e}")
        return None

@st.cache_data(ttl=3600)
def get_filing_content_from_hub(ticker, filename):
    """Downloads a raw filing file from HF Hub and returns its content."""
    try:
        # Handle different possible path formats
        if filename.startswith('data/'):
            # Already has full path
            repo_file_path = filename
        elif '/' in filename and 'sec-edgar-filings' in filename:
            # Already formatted path
            repo_file_path = filename
        else:
            # Just filename, construct the path
            repo_file_path = f"data/raw/sec-edgar-filings/{ticker}/{filename}"

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
        
@st.cache_data(ttl=3600)
def extract_ticker_and_filename_from_path(doc_path):
    """
    Extract ticker and filename from various document path formats.
    Returns tuple (ticker, filename) or (None, None) if parsing fails.
    """
    try:
        # Normalize path separators
        normalized_path = str(PurePath(doc_path))
        
        # Common patterns to match
        patterns = [
            # Pattern 1: ...sec-edgar-filings/TICKER/some/path/file.html
            r'sec-edgar-filings[/\\]([A-Z]+)[/\\](.+)$',
            # Pattern 2: TICKER/file.html (simple format)
            r'^([A-Z]+)[/\\]([^/\\]+\.html)$',
            # Pattern 3: Any path ending with TICKER_something.html
            r'([A-Z]+)_[^/\\]*\.html$',
            # Pattern 4: Path containing ticker in the filename
            r'[/\\]([A-Z]+)[_-][^/\\]*\.html$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized_path)
            if match:
                if len(match.groups()) == 2:
                    ticker, filename = match.groups()
                    return ticker, filename
                elif len(match.groups()) == 1:
                    # Extract ticker from filename
                    filename = os.path.basename(normalized_path)
                    ticker = match.group(1)
                    return ticker, filename
        
        # Fallback: try to extract from the path components
        path_parts = normalized_path.replace('\\', '/').split('/')
        
        # Look for sec-edgar-filings in the path
        if 'sec-edgar-filings' in path_parts:
            try:
                sec_index = path_parts.index('sec-edgar-filings')
                if sec_index + 1 < len(path_parts):
                    ticker = path_parts[sec_index + 1]
                    filename = '/'.join(path_parts[sec_index + 2:])
                    return ticker, filename
            except (ValueError, IndexError):
                pass
        
        # Last resort: try to find ticker in filename
        filename = os.path.basename(normalized_path)
        ticker_match = re.search(r'([A-Z]{2,5})', filename)
        if ticker_match:
            return ticker_match.group(1), filename
            
        return None, None
        
    except Exception as e:
        logging.error(f"Error parsing path {doc_path}: {e}")
        return None, None

def extract_filing_info_from_text(text_content):
    """Extract filing information from SEC document text."""
    try:
        filing_info = {}
        
        accession_match = re.search(r'ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})', text_content)
        if accession_match:
            filing_info['accession'] = accession_match.group(1)
        
        filing_type_match = re.search(r'CONFORMED SUBMISSION TYPE:\s*([^\n]+)', text_content)
        if filing_type_match:
            filing_info['filing_type'] = filing_type_match.group(1).strip()
        
        company_match = re.search(r'COMPANY CONFORMED NAME:\s*([^\n]+)', text_content)
        if company_match:
            filing_info['company_name'] = company_match.group(1).strip()
        
        cik_match = re.search(r'CENTRAL INDEX KEY:\s*(\d+)', text_content)
        if cik_match:
            filing_info['cik'] = cik_match.group(1)
        
        filed_date_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', text_content)
        if filed_date_match:
            filing_info['filed_date'] = filed_date_match.group(1)
        
        return filing_info
    except Exception as e:
        logging.error(f"Error extracting filing info: {e}")
        return {}

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
def find_relevant_filings(dataset, selected_company, query_keywords=None):
    """Find filings relevant to the selected company and query."""
    try:
        company_mapping = {
            'AAPL': 'APPLE INC',
            'MSFT': 'MICROSOFT',
            'GOOGL': 'ALPHABET INC',
            'AMZN': 'AMAZON',
            'TSLA': 'TESLA',
            'META': 'META PLATFORMS',
            'NVDA': 'NVIDIA',
            'BRK.B': 'BERKSHIRE HATHAWAY',
            'JNJ': 'JOHNSON & JOHNSON',
            'V': 'VISA'
        }
        
        company_name_pattern = company_mapping.get(selected_company, selected_company)
        relevant_filings = []
        
        for i, item in enumerate(dataset):
            text_content = item['text']
            if company_name_pattern.upper() in text_content.upper():
                filing_info = extract_filing_info_from_text(text_content)
                filing_info['index'] = i
                filing_info['content'] = text_content
                relevant_filings.append(filing_info)
                
                if len(relevant_filings) >= 5:  # Limit to prevent issues
                    break
        
        return relevant_filings
    except Exception as e:
        logging.error(f"Error finding relevant filings: {e}")
        return []

def get_sec_link_from_filing_info(filing_info, filename=""):
    """Generate SEC EDGAR link from filing information."""
    try:
        if 'accession' in filing_info and 'cik' in filing_info:
            accession = filing_info['accession']
            cik = filing_info['cik'].zfill(10)
            accession_no_dashes = accession.replace('-', '')
            
            if filename:
                return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{filename}"
            else:
                return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession}-index.html"
    except Exception as e:
        logging.error(f"Error generating SEC link: {e}")
    
    return None

def extract_document_sections(text_content):
    """Extract different sections from SEC filing text."""
    sections = {}
    
    try:
        documents = re.split(r'<DOCUMENT>', text_content)
        
        for doc in documents:
            if '<TYPE>' in doc:
                doc_type_match = re.search(r'<TYPE>([^<\n]+)', doc)
                if doc_type_match:
                    doc_type = doc_type_match.group(1).strip()
                    
                    text_match = re.search(r'<TEXT>(.*?)(?:</TEXT>|$)', doc, re.DOTALL)
                    if text_match:
                        sections[doc_type] = text_match.group(1).strip()
        
        return sections
    except Exception as e:
        logging.error(f"Error extracting document sections: {e}")
        return {}

def extract_relevant_paragraphs(content, query_keywords, max_paragraphs=3):
    """Extract paragraphs most relevant to the query using BeautifulSoup for cleaning."""
    if not query_keywords:
        return []

    try:
        # Use BeautifulSoup to parse and get clean text
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get all text and split into potential paragraphs
        all_text = soup.get_text(separator='\n', strip=True)
        # Filter for longer paragraphs that are more likely to contain prose
        paragraphs = [p.strip() for p in all_text.split('\n') if len(p.strip()) > 150]

        scored_paragraphs = []
        for para in paragraphs:
            # Simple scoring: count keyword occurrences
            score = sum(1 for keyword in query_keywords if keyword.lower() in para.lower())
            # Penalize paragraphs that look like code/links/tables
            if para.count('http') > 2 or para.count('|') > 5 or len(para.split()) < 15:
                score -= 2

            if score > 0:
                scored_paragraphs.append((score, para))

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        return [para for score, para in scored_paragraphs[:max_paragraphs]]
        
    except Exception as e:
        logging.error(f"Error extracting paragraphs: {e}")
        return []

def generate_unique_key(base_key, additional_info=""):
    """Generate a unique key for Streamlit components."""
    timestamp = str(int(time.time() * 1000))  # Current timestamp in milliseconds
    combined = f"{base_key}_{additional_info}_{timestamp}"
    # Create a hash to ensure uniqueness and reasonable length
    return hashlib.md5(combined.encode()).hexdigest()[:12]

def display_enhanced_sources_fixed(sources, prompt=""):
    """
    Fixed version that avoids Streamlit key collisions.
    """
    if not sources:
        return
        
    st.markdown("---")
    st.markdown("""
    <div style="padding: 15px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); border-radius: 8px; margin: 20px 0;">
    <h3 style="color: white; margin: 0; text-align: center;">📚 SEC Filings & Analysis</h3>
    <p style="color: white; margin: 10px 0 0 0; text-align: center; opacity: 0.9;">
    Direct access to SEC EDGAR filings with AI-powered analysis
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Load the SEC dataset
    with st.spinner("Loading SEC filings dataset..."):
        sec_dataset = load_sec_dataset()
    
    if not sec_dataset:
        st.error("❌ Failed to load SEC filings dataset from HuggingFace Hub")
        return

    # Find relevant filings for the selected company
    query_keywords = prompt.lower().split() if prompt else []
    relevant_filings = find_relevant_filings(sec_dataset, selected_company, query_keywords)
    
    if not relevant_filings:
        st.warning(f"No SEC filings found for {selected_company} in the dataset.")
        return

    st.success(f"✅ Found {len(relevant_filings)} relevant SEC filings for {selected_company}")

    # Display each relevant filing with unique keys
    for i, filing in enumerate(relevant_filings[:3]):  # Limit to 3 most recent
        filing_type = filing.get('filing_type', 'Unknown')
        accession = filing.get('accession', f'Unknown_{i}')  # Ensure accession is unique
        company_name = filing.get('company_name', 'Unknown')
        filed_date = filing.get('filed_date', 'Unknown')
        
        # Create a unique identifier for this filing
        filing_id = f"{i}_{accession}_{filed_date}".replace('-', '_').replace(' ', '_')
        
        # Format the date
        if filed_date != 'Unknown' and len(filed_date) == 8:
            try:
                formatted_date = f"{filed_date[:4]}-{filed_date[4:6]}-{filed_date[6:8]}"
            except:
                formatted_date = filed_date
        else:
            formatted_date = filed_date

        # Generate unique keys for this filing
        view_state_key = f'active_view_filing_{filing_id}'
        st.session_state.setdefault(view_state_key, None)

        with st.expander(f"📄 {filing_type} - {accession} ({formatted_date})", expanded=True):
            
            # Display filing metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filing Type", filing_type)
            with col2:
                st.metric("Filed Date", formatted_date)
            with col3:
                st.metric("Company", company_name)
            
            # SEC EDGAR link
            sec_link = get_sec_link_from_filing_info(filing)
            if sec_link:
                st.markdown(f'''
                <div style="padding: 15px; background-color: #e3f2fd; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3;">
                    <h4 style="margin: 0 0 10px 0; color: #1976d2;">🔗 Direct SEC EDGAR Link</h4>
                    <a href="{sec_link}" target="_blank" style="color: #1976d2; text-decoration: none; font-weight: bold;">
                        📊 View Official Filing on SEC.gov
                    </a>
                </div>
                ''', unsafe_allow_html=True)

            # Display source content from RAG
            if sources:
                st.markdown("### 📝 Relevant Content from RAG System")
                for j, ref in enumerate(sources, 1):
                    snippet = ref.page_content
                    st.markdown(f"""
                    <div style="border-left: 4px solid #007bff; padding-left: 15px; margin: 15px 0; background-color: rgba(0, 123, 255, 0.05); border-radius: 5px;">
                        <h5 style="color: #00aaff; margin-top: 5px; margin-bottom: 10px;">📍 Reference {j}</h5>
                        <p style="margin: 0; line-height: 1.6; color: #e0e0e0; font-size: 16px;">{snippet}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📁 Advanced Analysis & Access Options")
            
            col1, col2, col3 = st.columns(3)
            
            # Generate unique keys for buttons
            summary_key = generate_unique_key("summary", filing_id)
            sections_key = generate_unique_key("sections", filing_id)
            content_key = generate_unique_key("content", filing_id)
            
            with col1:
                if st.button(f"🤖 Generate AI Summary", key=summary_key):
                    st.session_state[view_state_key] = 'summary' if st.session_state[view_state_key] != 'summary' else None
            with col2:
                if st.button(f"📖 Extract Key Sections", key=sections_key):
                    st.session_state[view_state_key] = 'sections' if st.session_state[view_state_key] != 'sections' else None
            with col3:
                if st.button(f"📄 View Full Filing", key=content_key):
                    st.session_state[view_state_key] = 'content' if st.session_state[view_state_key] != 'content' else None
            
            active_view = st.session_state.get(view_state_key)
            
            if active_view:
                content = filing.get('content', '')
                
                if active_view == 'summary':
                    with st.spinner("🤖 Generating AI summary..."):
                        try:
                            # Extract clean text for summary
                            soup = BeautifulSoup(content, 'html.parser')
                            clean_content = soup.get_text()[:15000]  # Limit for API
                            
                            summary_prompt = f"""Analyze this {filing_type} SEC filing for {company_name} and provide a comprehensive executive summary.

Focus on:
- Key financial highlights and performance metrics
- Major business developments and strategic initiatives  
- Risk factors and challenges mentioned
- Management outlook and forward guidance
- Any significant changes from previous periods

Keep the summary under 400 words and use bullet points for clarity.

FILING CONTENT:
{clean_content}"""
                            
                            response = qual_brain.groq_client.chat.completions.create(
                                model=GROQ_LLM_MODEL,
                                messages=[{"role": "user", "content": summary_prompt}],
                                temperature=0.2,
                                max_tokens=600
                            )
                            
                            summary_content = response.choices[0].message.content
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1E293B 0%, #334155 100%); 
                                       padding: 25px; border-radius: 12px; color: white; 
                                       border-left: 5px solid #3B82F6; margin: 20px 0;">
                                <h4 style="margin: 0 0 15px 0; color: #60A5FA;">🤖 AI Executive Summary</h4>
                                <div style="line-height: 1.6;">{summary_content}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Failed to generate AI summary: {e}")
                            st.info("This might be due to API limits. Please try again later.")
                
                elif active_view == 'sections':
                    with st.spinner("📄 Extracting document sections..."):
                        try:
                            sections = extract_document_sections(content)
                            
                            if sections:
                                st.markdown("**📋 Document Sections Found:**")
                                for section_type, section_content in sections.items():
                                    section_key = generate_unique_key(f"section_{section_type}", filing_id)
                                    with st.expander(f"📄 {section_type}", expanded=False):
                                        # Clean and display section content
                                        soup = BeautifulSoup(section_content, 'html.parser')
                                        clean_section = soup.get_text()
                                        
                                        if len(clean_section) > 5000:
                                            st.info("📄 Section is very long. Showing first 5,000 characters.")
                                            clean_section = clean_section[:5000] + "\n\n... [Content truncated]"
                                        
                                        st.text_area(f"{section_type} Content", clean_section, height=300, key=section_key)
                            else:
                                st.warning("No distinct document sections found in this filing.")
                                
                        except Exception as e:
                            st.error(f"Failed to extract sections: {e}")
                
                elif active_view == 'content':
                    with st.spinner("📄 Preparing full filing content..."):
                        try:
                            # Clean up the content
                            soup = BeautifulSoup(content, 'html.parser')
                            clean_content = soup.get_text(separator='\n')
                            
                            if len(clean_content) > 50000:
                                st.warning("📄 Filing is very large. Showing first 50,000 characters.")
                                clean_content = clean_content[:50000] + "\n\n... [Content truncated for display]"
                            
                            st.markdown("#### 📄 Full SEC Filing Content")
                            
                            # Generate unique key for text area
                            text_area_key = generate_unique_key("content_full", filing_id)
                            
                            st.text_area(
                                "Filing Content", 
                                clean_content, 
                                height=500, 
                                key=text_area_key,
                                help="Complete text content of the SEC filing"
                            )
                            
                        except Exception as e:
                            st.error(f"Failed to display content: {e}")

# Alternative approach for the original display_enhanced_sources function
def display_enhanced_sources(sources, prompt=""):
    """
    Fixed wrapper function that handles both old and new source formats.
    """
    # If we have sources from the RAG system, display them in a simpler way first
    if sources:
        st.markdown("---")
        st.markdown("""
        <div style="padding: 15px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); border-radius: 8px; margin: 20px 0;">
        <h3 style="color: white; margin: 0; text-align: center;">📚 Sources & Analysis</h3>
        </div>
        """, unsafe_allow_html=True)

        # Group sources by document
        doc_sources = {}
        for s in sources:
            doc_key = s.metadata.get("source", "Unknown")
            if doc_key not in doc_sources:
                doc_sources[doc_key] = []
            doc_sources[doc_key].append(s)

        # Display each document source
        for doc_index, (doc_path, doc_refs) in enumerate(doc_sources.items()):
            if doc_path == "Unknown":
                continue
                
            filename = os.path.basename(doc_path) if doc_path else f"Document_{doc_index}"
            
            # Create unique identifier for this document
            doc_hash = hashlib.md5(f"{doc_path}_{doc_index}".encode()).hexdigest()[:8]
            
            with st.expander(f"📄 {filename}", expanded=True):
                # Display content snippets
                st.markdown("### 📝 Relevant Content")
                for i, ref in enumerate(doc_refs, 1):
                    snippet = ref.page_content
                    st.markdown(f"""
                    <div style="border-left: 4px solid #007bff; padding-left: 15px; margin: 15px 0; 
                               background-color: rgba(0, 123, 255, 0.05); border-radius: 5px;">
                        <h5 style="color: #00aaff; margin-top: 5px; margin-bottom: 10px;">📍 Reference {i}</h5>
                        <p style="margin: 0; line-height: 1.6; color: #e0e0e0; font-size: 16px;">{snippet}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simple analysis buttons with unique keys
                st.markdown("### 🔍 Quick Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    summary_key = f"quick_summary_{doc_hash}"
                    if st.button("🤖 Analyze Content", key=summary_key):
                        with st.spinner("Analyzing content..."):
                            try:
                                combined_content = " ".join([ref.page_content for ref in doc_refs])
                                analysis_prompt = f"""Analyze this content from a SEC filing and provide key insights:

{combined_content[:5000]}

Provide:
- Main financial points
- Key risks mentioned  
- Important business updates
- Strategic outlook

Keep it concise and bullet-pointed."""

                                response = qual_brain.groq_client.chat.completions.create(
                                    model=GROQ_LLM_MODEL,
                                    messages=[{"role": "user", "content": analysis_prompt}],
                                    temperature=0.2,
                                    max_tokens=400
                                )
                                
                                analysis_content = response.choices[0].message.content
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #1E293B 0%, #334155 100%); 
                                           padding: 20px; border-radius: 10px; color: white; margin: 15px 0;">
                                    <h4 style="margin: 0 0 15px 0; color: #60A5FA;">🔍 Content Analysis</h4>
                                    <div style="line-height: 1.6;">{analysis_content}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")
                
                with col2:
                    if len(combined_content := " ".join([ref.page_content for ref in doc_refs])) > 0:
                        st.info(f"Content length: {len(combined_content)} characters")
                        
    # Try to also load and display SEC filings if available
    try:
        display_enhanced_sources_fixed(sources, prompt)
    except Exception as e:
        st.warning(f"Advanced SEC analysis not available: {e}")
        logging.error(f"SEC analysis error: {e}")

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





