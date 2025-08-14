import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, HF_REPO_ID
import streamlit as st
from huggingface_hub import hf_hub_download
import shutil

@st.cache_resource
def load_vector_store_from_hub(repo_id, local_path, hf_folder="vector_store"):
    """Downloads the vector store from Hugging Face Hub."""
    if os.path.exists(local_path):
        logging.info("Vector store already exists locally.")
        return
    
    try:
        logging.info(f"Downloading vector store from {repo_id}...")
        # hf_hub_download doesn't support downloading a whole folder directly in one go
        # A common pattern is to zip the folder, upload it, and unzip it here.
        # Assuming you uploaded a 'vector_store.zip' file to your repo:
        zip_path = hf_hub_download(repo_id=repo_id, filename="vector_store.zip")
        shutil.unpack_archive(zip_path, local_path)
        logging.info("Vector store downloaded and unzipped successfully.")
    except Exception as e:
        logging.error(f"Failed to download vector store: {e}")
        st.error("Could not initialize the RAG system's vector store.")


class RetrievalEngine:
    """
    Handles the retrieval of documents from the vector store using local embeddings.
    """
    def __init__(self, company_ticker):
        self.company_ticker = company_ticker
        load_vector_store_from_hub(repo_id=HF_REPO_ID, local_path=VECTOR_STORE_PATH)

    def _load_vector_store(self):
        """Loads the vector store for the specified company."""
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
        """
        Retrieves the most relevant document chunks for a given query.
        """
        if self.vector_store is None:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Error during document retrieval for {self.company_ticker}: {e}")
            return []

