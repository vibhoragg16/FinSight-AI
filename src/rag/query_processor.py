import logging
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag.retrieval_engine import RetrievalEngine
from src.utils.config import GROQ_API_KEY, GROQ_LLM_MODEL, EMBEDDING_MODEL_NAME

def _expand_query(query: str) -> list[str]:
    """Uses Groq's LLM to expand the user's query into several alternatives."""
    if not GROQ_API_KEY:
        return [query]
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    Given the financial query "{query}", generate 3 alternative ways to phrase it to improve search results in a vector database of SEC filings.
    Return only the new queries, each on a new line.
    """
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150
        )
        expanded = response.choices[0].message.content.strip().split('\n')
        cleaned_queries = [q.split('. ', 1)[-1] for q in expanded if q]
        return [query] + cleaned_queries
    except Exception as e:
        logging.error(f"Query expansion with Groq failed: {e}")
        return [query]

def query_rag_system(company_ticker, query):
    """
    Queries the RAG system, now with query expansion and using Groq.
    """
    if not GROQ_API_KEY:
        return "Groq API key is not configured.", []

    retriever = RetrievalEngine(company_ticker)
    if retriever.vector_store is None:
        return f"No data found for {company_ticker}. Please run the collection and processing scripts.", []

    expanded_queries = _expand_query(query)
    all_docs = []
    for q in expanded_queries:
        all_docs.extend(retriever.retrieve_documents(q, k=2))
    
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    
    if not unique_docs:
        return "I could not find any relevant information in the documents.", []

    llm = ChatGroq(temperature=0.1, model_name=GROQ_LLM_MODEL, groq_api_key=GROQ_API_KEY)
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Use the updated Chroma import
        temp_retriever = Chroma.from_documents(list(unique_docs), embeddings).as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=temp_retriever,
            return_source_documents=True,
        )
        result = qa_chain({"query": query})
        return result["result"], result["source_documents"]
    except Exception as e:
        logging.error(f"Error querying RAG system for {company_ticker}: {e}")
        return f"An error occurred: {e}", []

