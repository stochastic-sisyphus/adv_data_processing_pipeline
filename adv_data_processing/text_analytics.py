import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Text analytics features will be limited.")

def perform_sentiment_analysis(text: str) -> Dict[str, float]:
    """Perform sentiment analysis on the given text."""
    if not TEXTBLOB_AVAILABLE:
        logger.warning("TextBlob not available. Returning neutral sentiment.")
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    try:
        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"polarity": 0.0, "subjectivity": 0.0}

def summarize_text(text: str, sentences: int = 3) -> str:
    """Generate a summary of the given text."""
    if not TEXTBLOB_AVAILABLE:
        logger.warning("TextBlob not available. Returning original text.")
        return f"{text[:100]}..." if len(text) > 100 else text
    
    try:
        blob = TextBlob(text)
        # Get the most relevant sentences based on word frequency
        sentences = blob.sentences[:sentences]
        return " ".join(str(sentence) for sentence in sentences)
    except Exception as e:
        logger.error(f"Error in text summarization: {str(e)}")
        return f"{text[:100]}..." if len(text) > 100 else text

