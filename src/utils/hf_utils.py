# src/utils/hf_utils.py
import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, Repository
from src.utils.config import HF_TOKEN, HF_DATASET_REPO_ID, DATA_PATH

def setup_hf_repo():
    """Logs into Hugging Face and clones the data repository if it doesn't exist."""
    if not HF_TOKEN or not HF_DATASET_REPO_ID:
        raise ValueError("Hugging Face token and repo ID must be set in the environment.")

    # Save the token for CLI usage
    HfFolder.save_token(HF_TOKEN)

    # Check if the data directory exists and is a git repo
    data_path = Path(DATA_PATH)
    if data_path.exists() and (data_path / ".git").exists():
        logging.info("Data repository already exists locally.")
        return Repository(local_dir=DATA_PATH)
    else:
        logging.info(f"Cloning data repository '{HF_DATASET_REPO_ID}' to '{DATA_PATH}'...")
        repo_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID}"
        repo = Repository(local_dir=DATA_PATH, clone_from=repo_url, use_auth_token=True)
        return repo

def pull_data_from_hf():
    """Pulls the latest data from the Hugging Face repository."""
    repo = setup_hf_repo()
    logging.info("Pulling latest data from Hugging Face Hub...")
    repo.git_pull()
    logging.info("Data pull complete.")

def push_data_to_hf(commit_message: str):
    """Commits and pushes the local data directory to the Hugging Face repository."""
    repo = setup_hf_repo()
    logging.info(f"Pushing data to Hugging Face Hub with message: '{commit_message}'")
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    repo.git_push()
    logging.info("Data push complete.")