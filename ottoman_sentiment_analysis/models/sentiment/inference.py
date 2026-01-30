"""
Sentiment Analysis Model Inference
===================================

Inference pipeline for classical sentiment analysis.
"""

import torch
from transformers import AutoTokenizer

from .architecture import EnhancedSentimentBERT
from .config import SENTIMENT_LABELS, ID_TO_SENTIMENT


class SentimentPredictor:
    """
    Sentiment analysis inference class.
    
    Example:
        >>> predictor = SentimentPredictor("path/to/model")
        >>> result = predictor.predict("Bu kitap çok güzeldi, çok beğendim.")
        >>> print(result)
        {'sentiment': 'positive', 'label': 2, 'confidence': 0.95}
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize sentiment predictor.
        
        Args:
            model_path (str): Path to saved model directory
            device (str, optional): Device ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = EnhancedSentimentBERT(
            model_name=model_path,
            num_labels=3
        )
        
        # Load state dict
        state_dict = torch.load(
            f"{model_path}/pytorch_model.bin",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.id2sentiment = ID_TO_SENTIMENT
    
    def predict(self, text, return_probabilities=False):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction result with sentiment, label, and confidence
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        result = {
            'sentiment': self.id2sentiment[predicted_class],
            'label': predicted_class,
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.id2sentiment[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict sentiments for multiple texts.
        
        Args:
            texts (list): List of input texts
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(text, return_probabilities) for text in texts]


def load_sentiment_model(model_path):
    """
    Convenience function to load sentiment model.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        SentimentPredictor: Loaded predictor instance
    """
    return SentimentPredictor(model_path)


if __name__ == "__main__":
    # Example usage
    predictor = SentimentPredictor("path/to/sentiment_model")
    
    test_texts = [
        "Bu kitap mükemmeldi, çok beğendim.",
        "Orta seviyede bir eser.",
        "Çok kötü, hiç beğenmedim."
    ]
    
    for text in test_texts:
        result = predictor.predict(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
