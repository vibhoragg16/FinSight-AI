import argparse
import sys
import subprocess
import logging
import time
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("platform.log"), logging.StreamHandler()]
)

def run_dashboard():
    """Launches the Streamlit dashboard."""
    logging.info("Launching Streamlit dashboard...")
    dashboard_path = Path("src/dashboard/streamlit_app.py")
    if not dashboard_path.exists():
        logging.error(f"Dashboard file not found at {dashboard_path}")
        return
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        logging.info("Dashboard stopped.")
    except Exception as e:
        logging.error(f"Failed to launch dashboard: {e}")

def run_scheduler(immediate: bool = False):
    """Runs the background task scheduler."""
    logging.info("Starting background scheduler...")
    try:
        from src.scheduler.task_scheduler import TaskScheduler
        scheduler = TaskScheduler()
        if immediate:
            logging.info("Running an immediate analysis cycle before starting schedule.")
            scheduler.run_full_cycle()
        scheduler.start()
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            scheduler.stop()
            logging.info("Scheduler stopped.")
    except ImportError as e:
        logging.error(f"Could not import scheduler modules: {e}")
    except Exception as e:
        logging.error(f"An error occurred in the scheduler: {e}")


def run_collection(types: list):
    """Runs data collection scripts."""
    logging.info(f"Starting data collection for: {types if types else 'all'}")
    types = types if types else ['sec', 'news', 'market']
    
    if 'sec' in types:
        from src.data_collection.sec_filings import download_sec_filings
        from src.utils.config import TARGET_COMPANIES
        download_sec_filings(TARGET_COMPANIES)
    if 'news' in types:
        from src.data_collection.news import fetch_company_news
        from src.utils.config import TARGET_COMPANIES
        fetch_company_news(TARGET_COMPANIES)
    if 'market' in types:
        from src.data_collection.market_data import fetch_market_data
        from src.utils.config import TARGET_COMPANIES
        fetch_market_data(TARGET_COMPANIES)
    
    logging.info("Data collection finished.")

def run_processing():
    """Runs document processing and vectorization."""
    logging.info("Starting document processing...")
    from src.document_processing.parser import process_all_documents
    process_all_documents()
    logging.info("Document processing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-Powered Corporate Intelligence Platform.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py dashboard       # Launch the Streamlit web interface
  python main.py scheduler       # Start the automated background tasks
  python main.py scheduler --now # Run one full cycle immediately, then start schedule
  python main.py collect --type news --type market # Collect only news and market data
  python main.py process         # Process downloaded documents into the vector store
"""
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    dashboard_parser = subparsers.add_parser('dashboard', help='Launch the Streamlit dashboard.')
    scheduler_parser = subparsers.add_parser('scheduler', help='Run the background task scheduler.')
    scheduler_parser.add_argument('--now', action='store_true', help='Run one full analysis cycle immediately.')
    
    collect_parser = subparsers.add_parser('collect', help='Run data collection tasks.')
    collect_parser.add_argument('--type', action='append', choices=['sec', 'news', 'market'], help='Specify data types to collect.')

    process_parser = subparsers.add_parser('process', help='Process downloaded documents.')

    args = parser.parse_args()

    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'scheduler':
        run_scheduler(immediate=args.now)
    elif args.command == 'collect':
        run_collection(args.type)
    elif args.command == 'process':
        run_processing()
    else:
        parser.print_help()
