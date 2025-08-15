# File: src/rag/retrieval_engine.py

import os
import logging
import streamlit as st
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, HF_REPO_ID

@st.cache_resource
def download_and_load_vector_store(repo_id, local_dir_base, company_ticker):
    """
    Downloads a specific company's vector store folder from Hugging Face Hub
    and loads it into a Chroma object.
    """
    persist_directory = os.path.join(local_dir_base, company_ticker)
    hf_folder_path = f"data/vector_store/{company_ticker}"

    # Step 1: Download the folder if it doesn't exist locally
    if not os.path.exists(persist_directory):
        logging.info(f"Vector store for {company_ticker} not found locally. Downloading...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=f"{hf_folder_path}/**",
                local_dir=local_dir_base,
                local_dir_use_symlinks=False
            )
            logging.info(f"Vector store for {company_ticker} downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download vector store folder for {company_ticker}: {e}")
            st.error(f"Could not download the database for {company_ticker}.")
            return None
    
    # Step 2: Load the vector store from the (now guaranteed) local path
    try:
        # Use the correct, modern embeddings class
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logging.info(f"Successfully loaded vector store for {company_ticker}.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to load vector store from {persist_directory}: {e}")
        return None

class RetrievalEngine:
    """
    Handles the retrieval of documents from the vector store using local embeddings.
    """
    def __init__(self, company_ticker):
        self.company_ticker = company_ticker
        self.vector_store = download_and_load_vector_store(
            repo_id=HF_REPO_ID,
            local_dir_base=VECTOR_STORE_PATH,
            company_ticker=self.company_ticker
        )

    def retrieve_documents(self, query, k=5):
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if self.vector_store is None:
            logging.warning(f"Vector store for {self.company_ticker} is not available.")
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Error during document retrieval for {self.company_ticker}: {e}")
            return []
