"""
CISA/CISA Model Inference
==========================

Inference pipeline for Cross-Individual Sentiment Analysis.
"""

import torch
import torch.nn.functional as F
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
    
    def __init__(self, model_path, device=None, max_length=256):
        """
        Initialize CISA predictor.
        
        Args:
            model_path (str): Path to saved model directory or HuggingFace model ID
            device (str, optional): Device ('cuda' or 'cpu')
            max_length (int): Maximum sequence length
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model - use fixed base model name (same as training)
        self.model = PositionAwareDualEncoderCISA(
            model_name='dbmdz/bert-base-turkish-cased',  # Base Turkish BERT
            num_sentiment_labels=3,
            dropout_rate=0.1,
            use_r_drop=False,
            stochastic_depth_rate=0.1
        )
        
        # Download and load weights from HuggingFace Hub if needed
        try:
            from huggingface_hub import hf_hub_download
            import os
            
            # Check if it's a HuggingFace model ID or local path
            if not os.path.exists(model_path):
                # Try pytorch_model.bin first (primary format on Hub)
                try:
                    model_file = hf_hub_download(
                        repo_id=model_path,
                        filename="pytorch_model.bin"
                    )
                    state_dict = torch.load(model_file, map_location=self.device)
                except:
                    # Fallback to safetensors
                    from safetensors.torch import load_file
                    model_file = hf_hub_download(
                        repo_id=model_path,
                        filename="model.safetensors"
                    )
                    state_dict = load_file(model_file)
            else:
                # Load from local path - try pytorch_model.bin first
                model_file = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location=self.device)
                else:
                    # Fallback to safetensors
                    from safetensors.torch import load_file
                    model_file = os.path.join(model_path, "model.safetensors")
                    state_dict = load_file(model_file)
            
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = ID_TO_CISA_LABEL
    
    def predict(self, text, entity, entity_start=None, entity_end=None, return_probabilities=False):
        """
        Predict CISA sentiment for entity in text.
        
        Args:
            text (str): Full text
            entity (str): Entity mention text
            entity_start (int, optional): Entity start position in text
            entity_end (int, optional): Entity end position in text
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction result with sentiment, label, confidence, and relation
        """
        # Find entity position if not provided
        if entity_start is None or entity_end is None:
            # Try case-insensitive search
            entity_start = text.lower().find(entity.lower())
            if entity_start == -1:
                raise ValueError(f"Entity '{entity}' not found in text: '{text[:100]}...'")
            entity_end = entity_start + len(entity)
        
        # Extract context after entity (up to 1800 characters)
        context_start = entity_end
        context_end = min(len(text), entity_end + 1800)
        context = text[context_start:context_end]
        
        # Create entity input with context (same format as training)
        entity_input = f"[CLS] {entity} [SEP] {context} [SEP]"
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
            return_offsets_mapping=True
        )
        
        # Tokenize entity with context
        entity_encoding = self.tokenizer(
            entity_input,
            padding='max_length',
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
            return_token_type_ids=False
        )
        
        # Find entity token positions
        offset_mapping = text_encoding['offset_mapping'][0]
        start_token_idx = None
        end_token_idx = None
        
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start <= entity_start < token_end:
                start_token_idx = token_idx
            if token_start < entity_end <= token_end:
                end_token_idx = token_idx
                break
        
        if start_token_idx is None:
            start_token_idx = 0
        if end_token_idx is None:
            end_token_idx = min(len(offset_mapping) - 1, start_token_idx + 1)
        
        entity_positions = [[start_token_idx, end_token_idx]]
        
        # Create position mask (0/1 integers, same as training)
        pos_mask = [0] * self.max_length
        for idx in range(start_token_idx, min(end_token_idx + 1, self.max_length)):
            pos_mask[idx] = 1
        position_mask = [pos_mask]
        
        # Remove offset_mapping before sending to model
        text_encoding.pop('offset_mapping')
        
        # Prepare inputs
        inputs = {
            'text_input_ids': text_encoding['input_ids'].to(self.device),
            'text_attention_mask': text_encoding['attention_mask'].to(self.device),
            'entity_input_ids': entity_encoding['input_ids'].to(self.device),
            'entity_attention_mask': entity_encoding['attention_mask'].to(self.device),
            'entity_positions': entity_positions,
            'position_mask': position_mask
        }
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentiment_logits = outputs['sentiment_logits']
            relation_logits = outputs['relation_logits']
            
            sentiment_pred = torch.argmax(sentiment_logits, dim=-1).cpu().item()
            relation_pred = torch.argmax(relation_logits, dim=-1).cpu().item()
            
            # Confidence scores
            sentiment_probs = F.softmax(sentiment_logits, dim=-1).cpu().numpy()[0]
            relation_probs = F.softmax(relation_logits, dim=-1).cpu().numpy()[0]
        
        result = {
            'sentiment': self.id2label[sentiment_pred],
            'label': sentiment_pred,
            'confidence': float(sentiment_probs[sentiment_pred]),
            'relation': relation_pred,
            'relation_confidence': float(relation_probs[relation_pred])
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.id2label[i]: float(sentiment_probs[i])
                for i in range(len(sentiment_probs))
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
    predictor = CISAPredictor("dbbiyte/CISA-BERTurk-sentiment")
    
    # Classic CISA example
    text = "Ali Bey'in vefatı bizleri hüzne boğmuştu. O büyük bir devlet adamıydı."
    entity = "Ali Bey"
    
    result = predictor.predict(text, entity, return_probabilities=True)
    print(f"Text: {text}")
    print(f"Entity: {entity}")
    print(f"CISA Result: {result}")
    print(f"\nNote: Despite sad context ('hüzne boğmuştu'), sentiment toward Ali Bey is positive (respect)")
