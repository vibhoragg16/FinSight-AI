# File: src/rag/retrieval_engine.py

import os
import logging
import streamlit as st
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, HF_REPO_ID

@st.cache_resource
def download_vector_store_folder(repo_id, local_dir_base, hf_folder_path):
    """
    Downloads a specific folder from a Hugging Face Hub dataset repo.
    """
    target_path = os.path.join(local_dir_base, os.path.basename(hf_folder_path))
    if os.path.exists(target_path):
        logging.info(f"Vector store for {os.path.basename(hf_folder_path)} already exists locally.")
        return

    logging.info(f"Downloading vector store from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{hf_folder_path}/**",
            local_dir=local_dir_base,
            local_dir_use_symlinks=False # Important for Streamlit Cloud
        )
        logging.info("Vector store folder downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download vector store folder: {e}")
        st.error("Could not initialize the RAG system's vector store.")

class RetrievalEngine:
    def __init__(self, company_ticker):
        self.company_ticker = company_ticker
        
        # This ensures the specific company's vector store is downloaded
        download_vector_store_folder(
            repo_id=HF_REPO_ID,
            local_dir_base=VECTOR_STORE_PATH,
            hf_folder_path=f"data/vector_store/{self.company_ticker}"
        )
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        persist_directory = os.path.join(VECTOR_STORE_PATH, self.company_ticker)
        if not os.path.exists(persist_directory):
            logging.warning(f"Vector store not found for {self.company_ticker}")
            return None
        try:
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception as e:
            logging.error(f"Failed to load vector store for {self.company_ticker}: {e}")
            return None

    def retrieve_documents(self, query, k=5):
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Error during document retrieval for {self.company_ticker}: {e}")
            return []
