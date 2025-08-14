# File: src/rag/retrieval_engine.py

import os
import logging
import streamlit as st
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, HF_REPO_ID

@st.cache_resource
def download_and_load_vector_store(repo_id, local_dir, hf_folder_path):
    """
    Downloads the entire vector store folder from Hugging Face Hub if it doesn't exist locally.
    """
    if not os.path.exists(local_dir):
        logging.info(f"Vector store not found locally. Downloading from Hugging Face Hub repo: {repo_id}...")
        try:
            # Use snapshot_download to download the entire folder
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset", # Make sure your repo is a "dataset" type on HF
                allow_patterns=f"{hf_folder_path}/**", # Pattern to download only the vector_store folder
                local_dir=os.path.dirname(local_dir) # Download to the parent of the target dir
            )
            logging.info("Vector store downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download vector store from Hugging Face Hub: {e}")
            st.error("Could not initialize the RAG system's vector store. Please check the logs.")
            return None
    
    # Load the vector store from the (now local) path
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=local_dir, embedding_function=embeddings)
        logging.info(f"Successfully loaded vector store for {os.path.basename(local_dir)}.")
        return vector_store
    except Exception as e:
        logging.error(f"Failed to load vector store from {local_dir}: {e}")
        return None


class RetrievalEngine:
    """
    Handles the retrieval of documents from the vector store using local embeddings.
    """
    def __init__(self, company_ticker):
        self.company_ticker = company_ticker
        
        # Define the local path for the company's vector store
        persist_directory = os.path.join(VECTOR_STORE_PATH, self.company_ticker)
        
        # Define the path of the folder on Hugging Face Hub
        hf_folder_path = f"data/vector_store/{self.company_ticker}"
        
        # This function handles both downloading and loading
        self.vector_store = download_and_load_vector_store(
            repo_id=HF_REPO_ID,
            local_dir=persist_directory,
            hf_folder_path=hf_folder_path
        )

    def retrieve_documents(self, query, k=5):
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if self.vector_store is None:
            logging.warning(f"Vector store for {self.company_ticker} is not available. Cannot retrieve documents.")
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Error during document retrieval for {self.company_ticker}: {e}")
            return []
