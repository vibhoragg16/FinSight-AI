from groq import Groq
from transformers import pipeline
from src.utils.config import GROQ_API_KEY, GROQ_LLM_MODEL

class QualitativeBrain:
    """
    Handles advanced qualitative analysis including sentiment and summarization using Groq.
    """
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze_text_sentiment(self, text):
        """Analyzes the sentiment of a given text, returning a score from -1 to 1."""
        try:
            results = self.sentiment_analyzer(text[:512])
            score = results[0]['score']
            return score if results[0]['label'] == 'POSITIVE' else -score
        except Exception:
            return 0

    def generate_qualitative_summary(self, company_ticker, rag_results):
        """
        Uses Groq's LLM to generate a qualitative summary based on RAG context.
        """
        context = "\n---\n".join([f"Source: {res.metadata.get('source', 'N/A')}\n{res.page_content}" for res in rag_results])
        
        prompt = f"""
        Based on the following excerpts from financial documents for {company_ticker},
        provide a qualitative analysis summary. Cover these key areas:
        1.  **Overall Sentiment:** What is the general tone (optimistic, cautious, pessimistic)?
        2.  **Key Themes & Narratives:** What are the main topics discussed?
        3.  **Potential Risks:** Identify any mentioned risks or uncertainties.
        4.  **Strategic Initiatives:** Note any discussion of future plans.

        Do not invent information. If a topic is not covered, state that.

        Context:
        {context}

        Analysis:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary with Groq: {e}"
