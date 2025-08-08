import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TARGET_COMPANIES = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
# Your name and email are required for the SEC EDGAR API
SEC_API_USER_AGENT = os.environ.get("SEC_API_USER_AGENT", "Vibhor Aggarwal vibhor.aggarwal1601@gmail.com")

# --- API Keys ---
# It's highly recommended to set these in a .env file
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- LLM & RAG ---
# Using a Llama3 model available on Groq
GROQ_LLM_MODEL = "llama3-8b-8192" 
# Using a local, open-source model for embeddings
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_STORE_PATH = "./data/vector_store"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# --- Data Paths ---
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"
MARKET_DATA_PATH = "./data/market_data"
NEWS_DATA_PATH = "./data/news"
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "./models/saved")


# --- Alerting ---
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # Use an App Password for Gmail
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

# --- Scheduler ---
# Defines the frequency in hours for different tasks
SCHEDULER_MARKET_DATA_HOURS = 4
SCHEDULER_NEWS_DATA_HOURS = 6
SCHEDULER_AI_ANALYSIS_HOURS = 8
SCHEDULER_SEC_FILINGS_DAYS = 1