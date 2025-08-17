# FinSight AI: AI-Powered Corporate Intelligence Platform

**FinSight AI** is a sophisticated, end-to-end platform designed to automate the complex workflow of a financial analyst. It autonomously gathers vast amounts of corporate data—including SEC filings, global news, and market data—processes it, and applies a suite of AI models to deliver actionable insights. The results are presented through a polished, interactive dashboard and a proactive email alerting system.

This platform is built on a modular, cloud-integrated architecture, ensuring scalability, maintainability, and easy extension of its capabilities.

---

## 🏛️ System Architecture & Technology Stack

FinSight AI is designed with a clear separation between data, AI models, and the user interface. Data assets are stored and versioned on Hugging Face Hub, while the application logic is managed in a Python-based backend and served via Streamlit.

### High-Level Data Flow

```
[Data Sources]        [Backend Processing]        [AI/ML Models]          [Frontend]
  - SEC EDGAR  -----> |                        |                         |
  - NewsAPI    -----> |  1. Data Collection  |                         |
  - yfinance   -----> | (main.py collect)    |                         |
                      |________________________|                         |
                                |                                      |
                                V                                      |
                      |                        |                         |
                      |  2. Document          |                         |
                      |  Processing &         |                         |
                      |  Vectorization        |                         |
                      | (main.py process)     |                         |
                      |________________________|                         |
                                |                                      |
                                V (Text Chunks)                        |
                      |                        |                         |
                      |  3. RAG Vector Store  |                         |
                      |  (ChromaDB on HF Hub) |                         |
                      |________________________|                         |
                                |                                      |
                                V (Financial Ratios)                   |
+----------------------------------------------------------------------+
|                                                                      |
|  4. Model Training (main.py train)                                   |
|     - PCA Health Scorer                                              |
|     - XGBoost Stock Predictor                                        |
|     (Models saved as .pkl on HF Hub)                                 |
|                                                                      |
+----------------------------------------------------------------------+
       |                  |                      |                      |
       V (User Query)     V (Context)            V (Predictions)        V (Data)
|                        |                      |                      |
|  5. Streamlit Dashboard (main.py dashboard)                          |
|     - Displays metrics, charts, and news.                            |
|     - Sends user queries to the RAG system.                          |
|     - RAG retrieves context, combines with query, and gets           |
|       response from Groq LLM.                                        |
|______________________________________________________________________|
```

### Technology Stack

-   **Backend & CLI**: Python, Argparse
-   **Web Dashboard**: Streamlit
-   **Data Manipulation**: Pandas, NumPy
-   **AI & Machine Learning**:
    -   **Prediction**: Scikit-learn (for PCA), XGBoost (for forecasting)
    -   **LLM/RAG**: LangChain (for orchestrating the RAG pipeline), Groq (for fast Llama 3 inference)
    -   **Embeddings**: Sentence-Transformers (for converting text to vectors)
    -   **Vector Database**: ChromaDB (for storing and retrieving text vectors)
-   **Data Collection**: `yfinance` (market data), `newsapi-python` (news), `sec-edgar-downloader` (SEC filings)
-   **File Storage & Versioning**: Hugging Face Hub (for datasets and models), Git LFS
-   **Utilities**: `python-dotenv` (environment variables), `schedule` (for automated tasks)

---

## ✨ Key Features in Depth

### 📊 Interactive Dashboard (`streamlit_app.py`)

The heart of the platform, providing a user-friendly interface to complex financial data.

-   **Multi-Tab Interface**: A clean, organized layout with four distinct sections:
    1.  **Market Analysis**: For viewing historical price and volume data.
    2.  **AI Analyst Chat**: The core RAG-powered feature for interacting with documents.
    3.  **News Analysis**: A feed of the latest news related to the selected company.
    4.  **Deep Dive**: A summary of all AI-generated insights and advanced analysis tools.

-   **At-a-Glance Metrics**: The main view presents the most critical AI-generated insights:
    -   **🧠 Fundamental Health Score**: A proprietary score from 0-100, calculated by a Principal Component Analysis (PCA) model trained on key financial ratios. This score provides a quick assessment of the company's financial stability.
    -   **🎯 Stock Forecast**: A bullish or bearish prediction on future price movement, complete with a confidence score. This is powered by an XGBoost model trained on a combination of technical indicators and financial ratios.
    -   **📰 News Sentiment**: Real-time sentiment analysis of the latest news headlines, providing a gauge of market perception.

### 🤖 Advanced AI & Document Interaction (`src/rag/`)

The "AI Analyst Chat" tab allows you to have a conversation with the company's financial documents.

-   **Retrieval-Augmented Generation (RAG)**: Instead of just asking an LLM a question, the system first retrieves relevant text snippets from the company's SEC filings (stored in a ChromaDB vector database). These snippets are then provided to the LLM as context, allowing it to generate highly accurate, source-based answers.
-   **On-Demand AI Summary**: Users can click a button to generate a concise, professional executive summary of any source document. The app fetches the full document from Hugging Face Hub, sends it to the Groq LLM, and displays the summary.
-   **Full Content Viewer & Paragraph Extraction**: You can view the complete, cleaned text of any source document or have the AI extract only the paragraphs most relevant to your query, all fetched on-demand from the cloud.

### ⚙️ Automated Backend & Data Pipelines (`main.py`)

The platform is managed through a powerful Command-Line Interface (CLI).

-   **Full-Lifecycle CLI**: `main.py` acts as the central control panel for the entire platform, allowing you to run data collection, processing, model training, and launch the dashboard with simple commands.
-   **Automated Model Training**: The `python main.py train` command automatically gathers all processed data, retrains both the PCA Health Scorer and the XGBoost Stock Predictor, and saves the updated models for use by the dashboard.
-   **Multi-Source Data Collection**: The `python main.py collect` command triggers scripts to download and process data from multiple sources:
    -   **SEC Filings**: Fetches 10-K and 10-Q documents directly from the SEC's EDGAR database.
    -   **Global News**: Collects real-time news from various sources via NewsAPI.
    -   **Market Data**: Gathers historical stock prices, volumes, and technical indicators using `yfinance` and the `ta` library.

---

## 📂 Project Structure Explained

The project is organized into a modular structure to ensure maintainability and a clear separation of concerns.

```
├── data/                  # (Locally) Stores all raw, processed, and model data. This is versioned on Hugging Face Hub.
│   ├── market_data/
│   ├── news/
│   ├── processed/
│   ├── raw/
│   │   └── sec-edgar-filings/
│   └── vector_store/
├── models/                # (Locally) Stores the trained machine learning models.
│   └── saved/
├── src/                   # Main source code for the application.
│   ├── ai_core/           # Core AI logic and coordination.
│   │   ├── feature_engineering.py # Prepares data for ML models.
│   │   ├── qualitative_brain.py   # Handles sentiment analysis and summarization via Groq.
│   │   └── quantitative_brain.py  # Coordinates numerical analysis (health score, predictions).
│   ├── alerting/          # Manages the email alerting system.
│   │   └── alert_system.py
│   ├── data_collection/   # Scripts for fetching data from external APIs.
│   │   ├── market_data.py
│   │   ├── news.py
│   │   └── sec_filings.py
│   ├── dashboard/         # Contains all code for the Streamlit web app.
│   │   ├── components/
│   │   │   └── style.css
│   │   └── streamlit_app.py
│   ├── document_processing/ # Scripts for parsing and cleaning raw data.
│   │   ├── parser.py      # Extracts text and creates the vector store.
│   │   └── table_extractor.py # Pulls tables from PDF documents.
│   ├── models/            # Defines the ML model classes.
│   │   ├── pca_health_scorer.py
│   │   └── xgboost_predictor.py
│   ├── rag/               # Handles the Retrieval-Augmented Generation system.
│   │   ├── query_processor.py # Expands queries and orchestrates the RAG chain.
│   │   └── retrieval_engine.py  # Manages interaction with the ChromaDB vector store.
│   └── utils/             # Utility functions and configuration.
│       ├── config.py      # Central configuration for API keys, paths, etc.
│       └── financial_ratios.py # Logic for calculating financial ratios.
├── .env.example           # Template for environment variables.
├── main.py                # The main Command-Line Interface (CLI) for running the platform.
└── requirements.txt       # A list of all Python package dependencies.
```

---

## 🚀 Getting Started: Setup & Usage

### 1. Prerequisites
-   Python 3.9+
-   Git and Git LFS

### 2. Installation
It is highly recommended to use a Python virtual environment.

```bash
# Clone the repository
git clone <your-repo-url>
cd finsight-ai

# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Configuration
The platform uses a `.env` file to manage sensitive keys.

1.  **Rename the example file:**
    ```bash
    # On Windows
    rename .env.example .env
    # On macOS/Linux
    mv .env.example .env
    ```
2.  **Edit the `.env` file** and add your API keys:
    -   `SEC_API_USER_AGENT`: **Required by the SEC.** Must be your name and email (e.g., "Jane Doe jane.doe@email.com").
    -   `GROQ_API_KEY`: Required for the LLM-powered features.
    -   `NEWS_API_KEY`: Required for news data collection.

### 4. Application Workflow
Follow these steps in order to run the platform.

#### **Step 1: Collect Initial Data**
Populate the system with data for the target companies. This will download all necessary files into your local `data/` directory.

```bash
python main.py collect
```

#### **Step 2: Process Documents**
Parse the downloaded SEC filings, extract their text, and build the vector database required for the RAG system.

```bash
python main.py process
```

#### **Step 3: Train the AI Models**
Train the health scorer and stock predictor models on the data you just collected and processed.

```bash
python main.py train
```

#### **Step 4: Launch the Dashboard**
Start the interactive Streamlit application to view the results.

```bash
python main.py dashboard
```

Your FinSight AI dashboard will now be running and accessible in your web browser!
