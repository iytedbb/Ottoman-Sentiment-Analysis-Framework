"""
Classical Sentiment Analysis Module
====================================

Sentiment analysis for historical Turkish texts (1900-1950).

Performance: 92.63% accuracy, F1: 0.9262

Classes:
    - Negative (0)
    - Neutral (1)
    - Positive (2)
"""

from .architecture import EnhancedSentimentBERT, WeightedFocalLoss, R_Drop, AdvancedDataCollator
from .config import SENTIMENT_CONFIG
from .train import train_sentiment_model
from .inference import SentimentPredictor

__all__ = [
    "EnhancedSentimentBERT",
    "WeightedFocalLoss",
    "R_Drop",
    "AdvancedDataCollator",
    "SENTIMENT_CONFIG",
    "train_sentiment_model",
    "SentimentPredictor"
]
