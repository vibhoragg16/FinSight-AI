import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME

class RetrievalEngine:
    """
    Handles the retrieval of documents from the vector store using local embeddings.
    """
    def __init__(self, company_ticker):
        self.company_ticker = company_ticker
        self.vector_store = self._load_vector_store()

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
