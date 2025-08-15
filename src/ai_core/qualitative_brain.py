# src/ai_core/qualitative_brain.py

import os
import logging
from groq import Groq
from textblob import TextBlob
from src.utils.config import GROQ_API_KEY, GROQ_LLM_MODEL

class QualitativeBrain:
    """
    Handles qualitative analysis tasks like sentiment analysis and text summarization.
    """
    def __init__(self):
        """Initialize the QualitativeBrain with Groq client."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables or .env file.")
        
        # Initialize Groq client with minimal arguments to avoid compatibility issues
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            logging.info("QualitativeBrain initialized successfully with Groq client.")
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            # Fallback initialization attempt
            try:
                self.groq_client = Groq(
                    api_key=GROQ_API_KEY,
                    # Remove any problematic parameters
                )
                logging.info("QualitativeBrain initialized with fallback method.")
            except Exception as e2:
                logging.error(f"Fallback initialization also failed: {e2}")
                raise ValueError(f"Could not initialize Groq client: {e2}")
                
    def analyze_text_sentiment(self, text):
        """
        Analyzes the sentiment of a given text using TextBlob as a fallback
        or Groq for more sophisticated analysis.
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        # Use TextBlob for quick sentiment analysis as primary method
        try:
            blob = TextBlob(text)
            return float(blob.sentiment.polarity)
        except Exception as e:
            logging.error(f"TextBlob sentiment analysis failed: {e}")
            return 0.0

    def generate_summary(self, text, max_length=150):
        """
        Generates a summary of the given text using Groq.
        """
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize."
        
        try:
            prompt = f"Please provide a concise summary of the following text in no more than {max_length} words:\n\n{text[:2000]}"  # Limit input length
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return "Could not generate summary."

    def analyze_financial_text(self, text, context="financial"):
        """
        Performs specialized financial text analysis using Groq.
        """
        if not text:
            return {"sentiment": 0.0, "key_points": [], "summary": "No text provided."}
        
        try:
            prompt = f"""
            Analyze the following {context} text and provide:
            1. Sentiment score (-1 to 1)
            2. Key financial insights (3-5 bullet points)
            3. Brief summary (2-3 sentences)
            
            Text: {text[:1500]}
            
            Format your response as:
            SENTIMENT: [score]
            KEY_POINTS:
            - [point 1]
            - [point 2]
            ...
            SUMMARY: [summary]
            """
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the response
            lines = result_text.split('\n')
            sentiment = 0.0
            key_points = []
            summary = ""
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('SENTIMENT:'):
                    try:
                        sentiment = float(line.split(':', 1)[1].strip())
                    except:
                        sentiment = 0.0
                elif line.startswith('KEY_POINTS:'):
                    current_section = 'key_points'
                elif line.startswith('SUMMARY:'):
                    current_section = 'summary'
                    summary = line.split(':', 1)[1].strip()
                elif current_section == 'key_points' and line.startswith('- '):
                    key_points.append(line[2:])
                elif current_section == 'summary' and line:
                    summary += " " + line
            
            return {
                "sentiment": sentiment,
                "key_points": key_points,
                "summary": summary.strip()
            }
            
        except Exception as e:
            logging.error(f"Financial text analysis failed: {e}")
            # Fallback to simple sentiment analysis
            return {
                "sentiment": self.analyze_text_sentiment(text),
                "key_points": ["Analysis unavailable due to API error"],
                "summary": "Could not perform detailed analysis."
            }

