"""
CISA/CISA Model Inference
==========================

Inference pipeline for Cross-Individual Sentiment Analysis.
"""

import torch
from transformers import AutoTokenizer

from .architecture import PositionAwareDualEncoderCISA
from .config import CISA_LABELS, ID_TO_CISA_LABEL


class CISAPredictor:
    """
    CISA model inference class.
    
    Analyzes author's sentiment toward specific individuals in text.
    
    Example:
        >>> predictor = CISAPredictor("path/to/model")
        >>> text = "Ali Bey'in vefatı bizleri hüzne boğmuştu"
        >>> entity = "Ali Bey"
        >>> result = predictor.predict(text, entity)
        >>> print(result)
        {'sentiment': 'positive', 'label': 2, 'confidence': 0.89}
        # Note: Despite sad context, sentiment toward Ali Bey is positive (respect)
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize CISA predictor.
        
        Args:
            model_path (str): Path to saved model directory
            device (str, optional): Device ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = PositionAwareDualEncoderCISA(
            model_name=model_path,
            num_sentiment_labels=3
        )
        
        # Load state dict
        state_dict = torch.load(
            f"{model_path}/pytorch_model.bin",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = ID_TO_CISA_LABEL
    
    def predict(self, text, entity, entity_start=None, entity_end=None, return_probabilities=False):
        """
        Predict CISA sentiment for entity in text.
        
        Args:
            text (str): Full text
            entity (str): Entity mention
            entity_start (int, optional): Entity start position in text
            entity_end (int, optional): Entity end position in text
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction result with sentiment, label, and confidence
        """
        # Find entity position if not provided
        if entity_start is None or entity_end is None:
            entity_start = text.find(entity)
            if entity_start == -1:
                raise ValueError(f"Entity '{entity}' not found in text")
            entity_end = entity_start + len(entity)
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # Tokenize entity
        entity_encoding = self.tokenizer(
            entity,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors="pt"
        )
        
        # Create position mask
        offset_mapping = text_encoding.pop('offset_mapping')[0]
        position_mask = []
        entity_positions = [0, 0]
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start >= entity_start and end <= entity_end:
                position_mask.append(True)
                if entity_positions[0] == 0:
                    entity_positions[0] = idx
                entity_positions[1] = idx
            else:
                position_mask.append(False)
        
        # Prepare inputs
        inputs = {
            'text_input_ids': text_encoding['input_ids'].to(self.device),
            'text_attention_mask': text_encoding['attention_mask'].to(self.device),
            'entity_input_ids': entity_encoding['input_ids'].to(self.device),
            'entity_attention_mask': entity_encoding['attention_mask'].to(self.device),
            'entity_positions': [entity_positions],
            'position_mask': [position_mask]
        }
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentiment_logits = outputs['sentiment_logits']
            probabilities = torch.softmax(sentiment_logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        result = {
            'sentiment': self.id2label[predicted_class],
            'label': predicted_class,
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.id2label[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        
        return result
    
    def predict_entities_in_text(self, text, entities, return_probabilities=False):
        """
        Predict CISA sentiments for multiple entities in the same text.
        
        Args:
            text (str): Full text
            entities (list): List of entity dictionaries with 'text', 'start', 'end'
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        for entity_info in entities:
            entity_text = entity_info['text']
            entity_start = entity_info.get('start')
            entity_end = entity_info.get('end')
            
            result = self.predict(
                text, entity_text, entity_start, entity_end, return_probabilities
            )
            result['entity'] = entity_text
            results.append(result)
        
        return results


def load_cisa_model(model_path):
    """
    Convenience function to load CISA model.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        CISAPredictor: Loaded predictor instance
    """
    return CISAPredictor(model_path)


if __name__ == "__main__":
    # Example usage
    predictor = CISAPredictor("path/to/cisa_model")
    
    # Classic CISA example
    text = "Ali Bey'in vefatı bizleri hüzne boğmuştu. O büyük bir devlet adamıydı."
    entity = "Ali Bey"
    
    result = predictor.predict(text, entity, return_probabilities=True)
    print(f"Text: {text}")
    print(f"Entity: {entity}")
    print(f"CISA Result: {result}")
    print(f"\nNote: Despite sad context ('hüzne boğmuştu'), sentiment toward Ali Bey is positive (respect)")
