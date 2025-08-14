import schedule
import time
import threading
import logging
import pandas as pd
import os
from datetime import datetime
from src.utils.config import TARGET_COMPANIES, MARKET_DATA_PATH, NEWS_DATA_PATH, SCHEDULER_MARKET_DATA_HOURS, SCHEDULER_NEWS_DATA_HOURS, SCHEDULER_AI_ANALYSIS_HOURS, SCHEDULER_SEC_FILINGS_DAYS
from src.data_collection import market_data, news, sec_filings
from src.document_processing.parser import process_all_documents
from src.alerting.alert_system import AlertSystem
from src.ai_core.qualitative_brain import QualitativeBrain
from src.ai_core.quantitative_brain import QuantitativeBrain
from src.training.train import run_training_pipeline

class TaskScheduler:
    """
    Automates the entire data pipeline with granular, multi-frequency tasks.
    It now works entirely with local data managed by Git LFS.
    """
    def __init__(self):
        self.running = False
        self.thread = None
        self.alert_system = AlertSystem()
        self.qual_brain = QualitativeBrain()
        self.quant_brain = QuantitativeBrain()
        self.latest_analysis = {}

    def run_full_cycle(self):
        """
        Runs one full cycle of data collection, processing, and analysis.
        """
        try:
            logging.info("CYCLE START: Running a full data and analysis cycle...")

            logging.info("CYCLE STEP 1: Collecting new data...")
            sec_filings.download_sec_filings(TARGET_COMPANIES, limit=2)
            news.fetch_company_news(TARGET_COMPANIES)
            market_data.fetch_market_data(TARGET_COMPANIES)
            
            logging.info("CYCLE STEP 2: Processing documents...")
            process_all_documents()

            logging.info("CYCLE STEP 3: Running analysis and checking alerts...")
            self.run_ai_analysis()
            self.check_alerts()
            
            logging.info("CYCLE COMPLETE. New data has been saved locally.")

        except Exception as e:
            logging.error(f"An error occurred during the scheduler cycle: {e}")


    # --- Individual Tasks ---
    def update_market_data(self):
        logging.info("SCHEDULER: Updating market data...")
        market_data.fetch_market_data(TARGET_COMPANIES, period="1y")

    def update_news_data(self):
        logging.info("SCHEDULER: Updating news data...")
        news.fetch_company_news(TARGET_COMPANIES)

    def update_sec_filings(self):
        logging.info("SCHEDULER: Updating SEC filings...")
        sec_filings.download_sec_filings(TARGET_COMPANIES, limit=2)
        process_all_documents()

    def run_ai_analysis(self):
        logging.info("SCHEDULER: Running AI analysis...")
        for company in TARGET_COMPANIES:
            market_file = os.path.join(MARKET_DATA_PATH, f"{company}_market_data.csv")
            news_file = os.path.join(NEWS_DATA_PATH, f"{company}_news.json")
            
            market_df = pd.read_csv(market_file) if os.path.exists(market_file) else pd.DataFrame()
            news_df = pd.read_json(news_file) if os.path.exists(news_file) else pd.DataFrame()
            
            financials_df = pd.DataFrame()

            news_sentiment = 0
            if not news_df.empty and 'articles' in news_df:
                articles = pd.json_normalize(news_df['articles'])
                if not articles.empty:
                    news_sentiment = articles['title'].apply(lambda x: self.qual_brain.analyze_text_sentiment(str(x))).mean()

            self.latest_analysis[company] = self.quant_brain.get_analysis(market_df, financials_df, news_sentiment)
        logging.info("SCHEDULER: AI analysis complete.")

    def check_alerts(self):
        logging.info("SCHEDULER: Checking for alerts...")
        if not self.latest_analysis:
            return

        for company, analysis_results in self.latest_analysis.items():
            market_df = pd.read_csv(os.path.join(MARKET_DATA_PATH, f"{company}_market_data.csv"))
            self.alert_system.check_all_conditions(
                company_ticker=company,
                analysis_results=analysis_results,
                market_data=market_df
            )

    def send_daily_summary(self):
        logging.info("SCHEDULER: Sending daily alert summary...")
        self.alert_system.send_daily_summary()

    def run_model_retraining(self):
        """Triggers the automated retraining and promotion pipeline."""
        logging.info("SCHEDULER: Kicking off monthly model retraining job...")
        try:
            run_training_pipeline()
        except Exception as e:
            logging.error(f"Automated model retraining failed: {e}")

    def _run_schedule(self):
        """The actual scheduling loop to be run in a thread."""
        logging.info("Configuring granular schedule...")
        
        schedule.every(SCHEDULER_MARKET_DATA_HOURS).hours.do(self.update_market_data)
        schedule.every(SCHEDULER_NEWS_DATA_HOURS).hours.do(self.update_news_data)
        schedule.every(SCHEDULER_AI_ANALYSIS_HOURS).hours.do(self.run_ai_analysis)
        schedule.every(SCHEDULER_AI_ANALYSIS_HOURS).hours.at(":30").do(self.check_alerts)
        schedule.every(SCHEDULER_SEC_FILINGS_DAYS).days.at("04:00").do(self.update_sec_filings)
        schedule.every().day.at("08:00").do(self.send_daily_summary)
        schedule.every().month.at("01:00").do(self.run_model_retraining)

        logging.info("Scheduler configured. Waiting for jobs...")
        while self.running:
            schedule.run_pending()
            time.sleep(60)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_schedule, daemon=True)
            self.thread.start()
            logging.info("Task scheduler started.")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join()
            logging.info("Task scheduler stopped.")
