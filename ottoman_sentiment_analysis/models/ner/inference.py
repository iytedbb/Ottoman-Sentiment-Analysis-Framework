"""
NER Model Inference
===================

Inference pipeline for Named Entity Recognition.

Main class:
    - NERPredictor: Load trained model and predict entities
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from .config import LABEL_MAP, ID_TO_LABEL
from ...utils.data_processing import normalize_text


class NERPredictor:
    """
    NER model inference class.
    
    Loads a trained NER model and provides entity extraction from text.
    
    Example:
        >>> predictor = NERPredictor("path/to/model")
        >>> entities = predictor.predict("Mustafa Kemal Paşa İstanbul'a geldi.")
        >>> print(entities)
        [{'text': 'Mustafa Kemal Paşa', 'label': 'PERSON', 'start': 0, 'end': 18},
         {'text': 'İstanbul', 'label': 'LOC', 'start': 19, 'end': 27}]
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize NER predictor.
        
        Args:
            model_path (str): Path to saved model directory
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = ID_TO_LABEL
    
    def predict(self, text, normalize=True):
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            normalize (bool): Whether to normalize text (remove diacritics)
            
        Returns:
            list: List of entity dictionaries with 'text', 'label', 'start', 'end'
        """
        if normalize:
            text = normalize_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
        
        # Convert predictions to entities
        entities = []
        current_entity = None
        
        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            # Skip special tokens
            if start == end:
                continue
            
            pred_label = self.id2label[pred_id.item()]
            
            if pred_label == "O":
                # Save current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                # Start new entity or extend current
                if current_entity and current_entity['label'] == pred_label:
                    # Extend current entity
                    current_entity['end'] = end.item()
                else:
                    # Save previous and start new
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': text[start:end],
                        'label': pred_label,
                        'start': start.item(),
                        'end': end.item()
                    }
        
        # Add last entity
        if current_entity:
            entities.append(current_entity)
        
        # Extract full entity text
        for entity in entities:
            entity['text'] = text[entity['start']:entity['end']]
        
        return entities
    
    def predict_batch(self, texts, normalize=True):
        """
        Extract entities from multiple texts.
        
        Args:
            texts (list): List of input texts
            normalize (bool): Whether to normalize texts
            
        Returns:
            list: List of entity lists (one per input text)
        """
        return [self.predict(text, normalize) for text in texts]


def load_ner_model(model_path):
    """
    Convenience function to load NER model.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        NERPredictor: Loaded predictor instance
    """
    return NERPredictor(model_path)


if __name__ == "__main__":
    # Example usage
    predictor = NERPredictor("path/to/ner_model")
    
    test_text = "Namık Kemal ve Ziya Paşa İstanbul'da buluştu."
    entities = predictor.predict(test_text)
    
    print(f"Text: {test_text}")
    print(f"Entities: {entities}")
