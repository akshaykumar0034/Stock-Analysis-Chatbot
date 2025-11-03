from transformers import pipeline
import warnings
from typing import List, Dict

# Suppress a known warning from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

# Initialize the pipeline once. 
# Using a model specifically trained on financial news (finbert)
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"Error loading sentiment model: {e}. Using default pipeline.")
    sentiment_pipeline = pipeline("sentiment-analysis")


def analyze_sentiment(news_list: List[str]) -> Dict:
    """Analyzes the sentiment of a list of news headlines."""
    if not news_list:
        return {"overall_sentiment": "Neutral", "details": "No news found."}
    
    try:
        sentiments = sentiment_pipeline(news_list)
        
        # Aggregate sentiments
        positive = 0
        negative = 0
        neutral = 0
        
        for s in sentiments:
            if s['label'] == 'positive':
                positive += 1
            elif s['label'] == 'negative':
                negative += 1
            else:
                neutral += 1
        
        if positive > negative and positive > neutral:
            overall = "Positive"
        elif negative > positive and negative > neutral:
            overall = "Negative"
        else:
            overall = "Neutral"
            
        return {
            "overall_sentiment": overall,
            "positive_articles": positive,
            "negative_articles": negative,
            "neutral_articles": neutral,
            "total_articles": len(news_list)
        }
    except Exception as e:
        return {"overall_sentiment": "Error", "details": str(e)}