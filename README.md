# AI-Powered Corporate Intelligence Platform (Complete)

This is an advanced, automated system designed to replicate the workflow of a financial analyst. It ingests a wide array of corporate data, analyzes it using quantitative and qualitative AI models, and presents insights through an interactive dashboard and proactive alerts.

This version features a highly modular structure inspired by best practices, separating concerns for better scalability and maintainability.

## Key Features

-   **Command-Line Interface**: Manage the platform (run dashboard, scheduler, data collection) via `main.py`.
-   **Automated Scheduler**: A background service (`scheduler`) that automates the entire data pipeline and analysis cycle with granular, multi-frequency tasks.
-   **Modular Data Pipeline**: Separate modules for collecting SEC filings, global news, and historical market data with technical indicators.
-   **Intelligent Document Processing**: Parses and chunks PDF documents, storing them in a vector database for retrieval.
-   **Modular AI Core**:
    -   **Quantitative Brain**: A high-level coordinator that uses specialized modules for feature engineering, health scoring (`PCAHealthScorer`), and stock prediction (`XGBoostPredictor`).
    -   **Qualitative Brain**: Uses a RAG system and LLMs to analyze sentiment and generate narrative summaries from source documents.
-   **Advanced RAG System**: Features a dedicated `RetrievalEngine` and query expansion for sophisticated document retrieval.
-   **Sophisticated Alerting**: A configurable system to detect critical events and send both immediate notifications and daily summaries.
-   **Interactive Dashboard**: A Streamlit application to visualize data, forecasts, and interact with the AI analyst via a chatbot interface.


## Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    -   Rename `.env.example` to `.env`.
    -   Open the `.env` file and add your API keys and email configuration.
        -   `SEC_API_USER_AGENT`: Must be set to your name and email (e.g., "John Doe john.doe@email.com").
        -   `OPENAI_API_KEY`: Required for the RAG and qualitative analysis.
        -   `NEWS_API_KEY`: Required for news collection.
        -   `EMAIL_*`: Required for the alerting system. For Gmail, you'll need to create an "App Password".

## How to Use

The platform is managed via the `main.py` script.

1.  **Collect Initial Data**
    First, you need to populate the system with data.
    ```bash
    python main.py collect
    ```

2.  **Process Documents**
    Next, process the downloaded filings into the vector store for the RAG system.
    ```bash
    python main.py process
    ```

3.  **Launch the Dashboard**
    To view and interact with the data.
    ```bash
    python main.py dashboard
    ```

4.  **Run the Automated Scheduler**
    To keep the platform running and updating automatically in the background.
    ```bash
    # Start the scheduler to run tasks at different intervals
    python main.py scheduler
    