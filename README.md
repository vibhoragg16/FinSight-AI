# FinSight-AI: AI-Powered Corporate Intelligence Platform

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/vibhoragg16/finsight-ai)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/vibhoragg16/finsight-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**FinSight-AI** is a sophisticated, end-to-end platform designed to automate the complex workflow of a financial analyst. It autonomously gathers vast amounts of corporate data—including SEC filings, global news, and market data—processes it, and applies a suite of AI models to deliver actionable insights. The results are presented through a polished, interactive dashboard and a proactive email alerting system.

This platform is built on a modular, cloud-integrated architecture, ensuring scalability, maintainability, and easy extension of its capabilities.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Project In-Depth](#project-in-depth)
  - [Data & Processing Pipeline](#data--processing-pipeline)
  - [Machine Learning Models](#machine-learning-models)
  - [RAG Pipeline for AI Analyst Chat](#rag-pipeline-for-ai-analyst-chat)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project

FinSight-AI is engineered to tackle the overwhelming challenge of information overload in the financial industry. Analysts and investors are often buried under a deluge of SEC filings, news articles, and market data, making it difficult to extract timely, actionable insights. This platform automates the entire intelligence-gathering and analysis pipeline, from data acquisition to insight delivery.

**Key benefits include:**

-   **Automated Data Pipelines:** Drastically reduces manual effort by automatically fetching and processing data from multiple sources.
-   **AI-Powered Insights:** Leverages machine learning and large language models (LLMs) to uncover trends, risks, and opportunities that might otherwise be missed.
-   **Interactive & Intuitive UI:** Presents complex financial information in a clear, digestible format through a Streamlit-powered dashboard.
-   **Proactive Alerting:** Keeps users informed of critical events and market changes through a sophisticated, rule-based email alert system.

This project is for financial analysts, retail investors, and anyone looking to gain a competitive edge through data-driven decision-making.

---

## Project In-Depth

### Data & Processing Pipeline

The platform's intelligence is built upon a robust data pipeline that gathers, processes, and prepares data for AI analysis.

1.  **Data Collection:** The pipeline begins by collecting three core types of data using the `main.py collect` command:
    * **SEC Filings:** Downloads 10-K (annual) and 10-Q (quarterly) reports from the SEC EDGAR database using the `sec-edgar-downloader` library.
    * **Market Data:** Fetches historical stock prices, trading volumes, and calculates a suite of technical indicators (RSI, MACD, etc.) using `yfinance` and the `ta` library.
    * **News Data:** Gathers the latest global news articles for each target company via the `NewsAPI`.

2.  **Document Processing:** Once collected, the raw, unstructured data (especially SEC filings) is processed with the `main.py process` command:
    * **Text Extraction:** The `parser.py` script uses `PyMuPDFLoader` and `UnstructuredHTMLLoader` to extract clean text from PDF and HTML documents.
    * **Table Extraction:** It identifies and extracts financial tables from PDF filings using `camelot-py`, saving them as structured CSV files for quantitative analysis.
    * **Chunking & Vectorization:** The extracted text is split into smaller, semantically coherent chunks. Each chunk is then converted into a numerical vector (embedding) using a `SentenceTransformer` model (`all-MiniLM-L6-v2`) and stored in a `ChromaDB` vector store, which powers the RAG system.

### Machine Learning Models

FinSight-AI employs two specialized machine learning models to generate its core quantitative insights. These are trained using the `main.py train` command.

1.  **🧠 PCA Health Scorer (`pca_health_scorer.py`):**
    * **Purpose:** To distill a wide array of complex financial ratios into a single, intuitive "Fundamental Health Score" from 0-100.
    * **Technique:** This model uses **Principal Component Analysis (PCA)**, an unsupervised learning technique, to identify the primary drivers of financial health from ratios like Debt-to-Equity, Return on Assets, and more. It provides a holistic view of a company's financial stability.
    * **Output:** A single score representing the company's overall financial health relative to the dataset it was trained on.

2.  **🎯 XGBoost Stock Predictor (`xgboost_predictor.py`):**
    * **Purpose:** To forecast short-term stock price movement (Bullish or Bearish) and provide a confidence level for its prediction.
    * **Technique:** It uses an **XGBoost (Extreme Gradient Boosting)** classifier, a powerful supervised learning algorithm.
    * **Features:** The model is trained on a rich feature set prepared by `build_features.py`, which includes:
        * **Technical Indicators:** SMA, RSI, MACD, Bollinger Bands, etc.
        * **Fundamental Ratios:** Current Ratio, Debt-to-Equity, etc.
        * **Sentiment Score:** The latest news sentiment is included as a feature.
    * **Output:** A prediction ("Bullish" or "Bearish") and a confidence score (e.g., 85%).

### RAG Pipeline for AI Analyst Chat

The "AI Analyst Chat" feature is powered by a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline, ensuring that the LLM's answers are grounded in factual, company-specific data.

1.  **Query Expansion (`query_processor.py`):** When a user submits a query (e.g., "What are the main risks for this company?"), it is first sent to the **Groq Llama 3 LLM** to generate several alternative phrasings. This enhances the search by capturing different semantic variations of the original question.

2.  **Retrieval (`retrieval_engine.py`):** All query versions are converted into vector embeddings. The `RetrievalEngine` then searches the company-specific **ChromaDB vector store** to find the text chunks from SEC filings that are most semantically similar to the user's queries.

3.  **Augmentation & Generation (`query_processor.py`):** The retrieved text chunks (the "context") are bundled together with the original user query into a new, detailed prompt. This augmented prompt is then sent to the **Groq Llama 3 LLM**. By providing this direct evidence, the LLM can generate a highly accurate and contextually relevant answer, often citing the source documents directly.

---

## Features

-   **📊 Interactive Dashboard (`streamlit_app.py`):**
    -   **Multi-Tab Interface:** Clean layout with dedicated sections for Market Analysis, AI Analyst Chat, News Analysis, and a Deep Dive summary.
    -   **At-a-Glance Metrics:** Critical AI-generated insights, including a Fundamental Health Score, Stock Forecast with confidence levels, and real-time News Sentiment.

-   **🤖 Advanced AI & Document Interaction (`src/rag/`):**
    -   **Retrieval-Augmented Generation (RAG):** Chat with SEC filings to get accurate, source-based answers, powered by LangChain and Groq's Llama 3.
    -   **On-Demand AI Summary:** Generate concise executive summaries of source documents with a single click.

-   **⚙️ Automated Backend & Data Pipelines (`main.py`):**
    -   **Full-Lifecycle CLI:** A central command-line interface to manage data collection, processing, model training, and dashboard launch.
    -   **Automated Model Training:** A simple command (`python main.py train`) to retrain all models on the latest data.
    -   **Multi-Source Data Collection:** Gathers data from SEC EDGAR, NewsAPI, and yfinance.

-   **⏰ Scheduling & Alerting (`src/scheduler/`, `src/alerting/`):**
    -   **Task Scheduler:** Automates the entire data pipeline with configurable frequencies for different tasks.
    -   **Proactive Alerting:** A rule-based system that monitors for critical events and sends immediate email alerts.

-   **🧠 AI Core & Document Parsing (`src/ai_core/`, `src/document_processing/`):**
    -   **Qualitative & Quantitative Brains:** Separates AI logic for text-based analysis (sentiment, summarization) and numerical analysis (health score, predictions).
    -   **Advanced Document Parsing:** Extracts text and tables from PDF and HTML filings, chunks them, and creates a vector store using ChromaDB.

---


## System Architecture

FinSight-AI is designed with a clear separation between data, AI models, and the user interface. Data assets are stored and versioned on Hugging Face Hub, while the application logic is managed in a Python-based backend and served via Streamlit.

### High-Level Data Flow

```
[Data Sources]        [Backend Processing]        [AI/ML Models]          [Frontend]
  - SEC EDGAR  -----> |                        |                       |
  - NewsAPI    -----> |  1. Data Collection    |                       |
  - yfinance   -----> | (main.py collect)      |                       |
                      |________________________|                       |
                                |                                      |
                                V                                      |
                      |                        |                       |
                      |  2. Document           |                       |
                      |  Processing &          |                       |
                      |  Vectorization         |                       |
                      | (main.py process)      |                       |
                      |________________________|                       |
                                |                                      |
                                V (Text Chunks)                        |
                      |                        |                       |
                      |  3. RAG Vector Store   |                       |
                      |  (ChromaDB on HF Hub)  |                       |
                      |________________________|                       |
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

## Technology Stack

-   **Backend & CLI:** Python, Argparse
-   **Web Dashboard:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **AI & Machine Learning:**
    -   **Prediction:** Scikit-learn (PCA), XGBoost (forecasting)
    -   **LLM/RAG:** LangChain, Groq (Llama 3 inference)
    -   **Embeddings:** Sentence-Transformers
    -   **Vector Database:** ChromaDB
-   **Data Collection:** `yfinance`, `newsapi-python`, `sec-edgar-downloader`
-   **File Storage & Versioning:** Hugging Face Hub, Git LFS
-   **Utilities:** `python-dotenv`, `schedule`

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
