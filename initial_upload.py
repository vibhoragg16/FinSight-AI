# initial_upload.py
import logging
from datetime import datetime
from src.utils.hf_utils import push_data_to_hf

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_initial_data():
    """
    A simple script to perform the first-time upload of the local data
    directory to the Hugging Face Hub.
    """
    print("Starting the initial data upload to Hugging Face Hub...")
    try:
        commit_message = f"Initial data commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        push_data_to_hf(commit_message=commit_message)
        print("\nInitial data upload was successful!")
        print("You can now delete this 'initial_upload.py' script.")
    except Exception as e:
        print(f"\nAn error occurred during the upload: {e}")
        print("Please ensure your HF_TOKEN and HF_DATASET_REPO_ID are correct in your .env file.")

if __name__ == "__main__":
    upload_initial_data()