import os
import logging
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from src.utils.config import RAW_DATA_PATH, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TARGET_COMPANIES, EMBEDDING_MODEL_NAME, PROCESSED_DATA_PATH
from src.document_processing.table_extractor import extract_tables_from_pdf, clean_financial_table

def process_documents_for_company(company_ticker):
    """
    Processes all downloaded filings for a single company.
    - Extracts text from PDF/HTML for the RAG system.
    - Extracts tables from PDFs for quantitative analysis.
    """
    logging.info(f"Starting document processing for {company_ticker}...")
    
    company_path = os.path.join(RAW_DATA_PATH, "sec-edgar-filings", company_ticker)
    documents_for_rag = []

    if not os.path.exists(company_path):
        logging.warning(f"No filings directory found for {company_ticker} at {company_path}")
        return

    # Walk through the directory and load all supported documents
    for root, _, files in os.walk(company_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # --- Text Extraction for RAG ---
            loaded_docs = []
            try:
                if file.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path)
                    loaded_docs = loader.load()
                # CORRECTED FILENAME: Look for 'primary-document.html'
                elif file == "primary-document.html":
                    loader = UnstructuredHTMLLoader(file_path)
                    loaded_docs = loader.load()

                if loaded_docs:
                    for doc in loaded_docs:
                        doc.metadata['source'] = file_path
                    documents_for_rag.extend(loaded_docs)

            except Exception as e:
                logging.error(f"Failed to load text from {file_path}: {e}")

            # --- Table Extraction for Quantitative Analysis ---
            if file.endswith(".pdf"):
                logging.info(f"Extracting tables from PDF: {file_path}")
                extracted_tables = extract_tables_from_pdf(file_path, flavor='stream')
                if not extracted_tables:
                    extracted_tables = extract_tables_from_pdf(file_path, flavor='lattice')
                
                if extracted_tables:
                    output_dir = os.path.join(PROCESSED_DATA_PATH, company_ticker, "tables")
                    os.makedirs(output_dir, exist_ok=True)
                    for i, table_df in enumerate(extracted_tables):
                        try:
                            cleaned_df = clean_financial_table(table_df)
                            filing_name = os.path.splitext(os.path.basename(file_path))[0]
                            csv_path = os.path.join(output_dir, f"{filing_name}_table_{i+1}.csv")
                            cleaned_df.to_csv(csv_path, index=False)
                        except Exception as e:
                            logging.warning(f"Could not clean or save table {i+1} from {file_path}: {e}")
                    logging.info(f"Saved {len(extracted_tables)} tables from {file_path}")


    # --- Vector Store Creation for RAG ---
    if not documents_for_rag:
        logging.warning(f"No processable text documents found for {company_ticker}.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(documents_for_rag)

        logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        persist_directory = os.path.join(VECTOR_STORE_PATH, company_ticker)
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info(f"Successfully vectorized text from {len(documents_for_rag)} documents for {company_ticker}.")

def process_all_documents():
    """Iterates through all target companies and processes their documents."""
    logging.info("Starting document processing for all target companies.")
    for company in TARGET_COMPANIES:
        process_documents_for_company(company)
    logging.info("Document processing for all companies complete.")