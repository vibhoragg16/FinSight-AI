# FinSight AI: AI-Powered Corporate Intelligence Platform

**FinSight AI** is a sophisticated, end-to-end platform designed to automate the complex workflow of a financial analyst. It autonomously gathers vast amounts of corporate data—including SEC filings, global news, and market data—processes it, and applies a suite of AI models to deliver actionable insights. The results are presented through a polished, interactive dashboard and a proactive email alerting system.

This platform is built on a modular architecture, ensuring scalability, maintainability, and easy extension of its capabilities.

<!-- It's recommended to replace this with a real screenshot of your app -->

## Key Features

### 📊 Interactive Dashboard
- **Multi-Tab Interface**: A clean, organized Streamlit dashboard with dedicated tabs for Market Analysis, AI Analyst Chat, News Analysis, and a Deep Dive section.
- **At-a-Glance Metrics**: The main view presents the most critical AI-generated insights:
  - **🧠 Fundamental Health Score**: A proprietary score from 0-100 based on a PCA model trained on key financial ratios.
  - **🎯 Stock Forecast**: A bullish or bearish prediction on future price movement with a confidence score, powered by an XGBoost model.
  - **📰 News Sentiment**: Real-time sentiment analysis of the latest news headlines.
- **Dynamic Charts**: Interactive candlestick charts for market performance analysis.
- **Risk Assessment**: A dashboard that categorizes market, financial, and sentiment risks into Low, Medium, or High.
- **Modern News Feed**: A visually appealing card-based layout for news articles, complete with images and descriptions.

### 🤖 Advanced AI & Document Interaction
The AI Analyst Chat tab provides a powerful suite of tools to interact with source documents like SEC filings:

- **AI-Powered Document Summary**: On-demand, AI-generated executive summaries of lengthy SEC filings. It focuses on critical information like financial performance, key business segments, risk factors, and future outlook, presenting it in a clean, readable format.
- **RAG-Powered Analyst Chat**: A sophisticated Retrieval-Augmented Generation (RAG) system allows you to "chat" with your documents. Ask specific questions and get answers sourced directly from the text.
- **Intelligent Paragraph Extraction**: Automatically identifies and extracts the most relevant paragraphs from a document based on your query, using a `BeautifulSoup`-powered cleaning engine for maximum readability.
- **Interactive Document Viewer**: A custom-styled, scrollable "code block" viewer for inspecting the complete, cleaned raw text of any source document.
- **Robust Financial Ratio Analysis**: Calculates a comprehensive suite of liquidity, leverage, profitability, and efficiency ratios. It now transparently reports on any missing source data points, giving you clear insight into the quality of the underlying financial statements.

### ⚙️ Automated Backend & Data Pipelines
- **Full-Lifecycle CLI**: A powerful `main.py` script to manage the entire platform: collect data, process documents, train models, and launch the dashboard.
- **Automated Model Training**: A dedicated `train` command to retrain all AI models (Health Scorer and Stock Predictor) on the latest available data, ensuring the models stay current.
- **Multi-Source Data Collection**: Automated scripts to download and process:
  - **SEC Filings**: Fetches 10-K and 10-Q documents directly from EDGAR.
  - **Global News**: Collects real-time news from various sources via NewsAPI.
  - **Market Data**: Gathers historical stock prices and volumes.
- **Proactive Alerting System**: A rule-based engine that monitors for critical events (e.g., health score drops, price volatility) and sends immediate email alerts for high-severity events and daily summaries for others.

## Tech Stack & Architecture

FinSight AI is built with a modern, modular Python stack:

- **Backend & CLI**: Python, Argparse
- **Web Dashboard**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **AI & Machine Learning**:
  - **Prediction**: Scikit-learn, XGBoost
  - **LLM/RAG**: LangChain, Groq (for Llama3), Sentence-Transformers
  - **Text Processing**: BeautifulSoup
- **File Storage**: Joblib for model serialization

The architecture separates concerns into distinct modules for data collection, processing, AI analysis, and presentation, making the system easy to maintain and extend.

## 🚀 Getting Started: Setup & Usage

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Installation
It is highly recommended to use a Python virtual environment.

```bash
# Clone the repository
git clone <your-repo-url>
cd finsight-ai

# Create and activate a virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Configuration
The platform uses a `.env` file to manage sensitive keys and configurations.

1.  **Rename the example file:**
    ```bash
    rename .env.example .env
    ```
2.  **Edit the `.env` file** and add your credentials:
    - `SEC_API_USER_AGENT`: **Required by the SEC.** Must be your name and email (e.g., "Jane Doe jane.doe@email.com").
    - `GROQ_API_KEY`: Required for the LLM-powered features (Summarization, Chat).
    - `NEWS_API_KEY`: Required for news data collection.
    - `EMAIL_*`: Required for the alerting system. For Gmail, you will need to create an "App Password".

### 4. Application Workflow
Follow these steps in order to run the platform.

#### **Step 1: Collect Initial Data**
Populate the system with data for the target companies defined in `src/utils/config.py`.

```bash
# Collect all data types (SEC filings, news, and market data)
python main.py collect
```

#### **Step 2: Process Documents**
Parse the downloaded SEC filings and store them in the vector database for the RAG system.

```bash
python main.py process
```

#### **Step 3: Train the AI Models**
This is a crucial step. Train the health scorer and stock predictor models on the data you just collected and processed.

```bash
python main.py train
```

#### **Step 4: Launch the Dashboard**
Start the interactive Streamlit application to view the results.

```bash
python main.py dashboard
```

Your FinSight AI dashboard will now be running and accessible in your web browser!
