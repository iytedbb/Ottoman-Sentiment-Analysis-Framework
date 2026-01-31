"""
CISA/CISA Model Training
=========================

Training pipeline for Cross-Individual Sentiment Analysis.
"""

import os
import json
import logging
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import Dataset

from .architecture import PositionAwareDualEncoderCISA
from .config import CISA_CONFIG
from ...utils.data_processing import load_json_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed=42):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class PositionAwareDataCollator:
    """Custom data collator for CISA with R-Drop support."""
    
    def __init__(self, tokenizer, r_drop=False, max_length=256):
        self.tokenizer = tokenizer
        self.r_drop = r_drop
        self.max_length = max_length

    def __call__(self, features):
        batch = {
            "text_input_ids": torch.tensor([f["text_input_ids"] for f in features]),
            "text_attention_mask": torch.tensor([f["text_attention_mask"] for f in features]),
            "entity_input_ids": torch.tensor([f["entity_input_ids"] for f in features]),
            "entity_attention_mask": torch.tensor([f["entity_attention_mask"] for f in features]),
            "sentiment_label": torch.tensor([f["sentiment_label"] for f in features]),
            "relation_label": torch.tensor([f["relation_label"] for f in features]),
            "entity_positions": [f["entity_positions"] for f in features],
            "position_mask": [f["position_mask"] for f in features]
        }
        
        if self.r_drop:
            batch["text_input_ids2"] = batch["text_input_ids"].clone()
            batch["text_attention_mask2"] = batch["text_attention_mask"].clone()
            batch["entity_input_ids2"] = batch["entity_input_ids"].clone()
            batch["entity_attention_mask2"] = batch["entity_attention_mask"].clone()
        
        return batch


def prepare_cisa_dataset(data, tokenizer, max_length=256):
    """Prepare CISA dataset with dual encoding."""
    processed_examples = []
    
    for item in data:
        text = item['text']
        entities = item['entities']
        
        for entity in entities:
            entity_text = entity['target']
            entity_start = entity['start']
            entity_end = entity['end']
            sentiment = entity['sentiment']
            author_related = entity.get('author_related', False)
            
            # Tokenize full text
            text_encoding = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True
            )
            
            # Tokenize entity
            entity_encoding = tokenizer(
                entity_text,
                padding='max_length',
                truncation=True,
                max_length=64
            )
            
            # Create position mask
            offset_mapping = text_encoding.pop('offset_mapping')
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
            
            # Sentiment label mapping
            sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
            sentiment_label = sentiment_map.get(sentiment.lower(), 1)
            
            relation_label = 1 if author_related else 0
            
            processed_examples.append({
                'text_input_ids': text_encoding['input_ids'],
                'text_attention_mask': text_encoding['attention_mask'],
                'entity_input_ids': entity_encoding['input_ids'],
                'entity_attention_mask': entity_encoding['attention_mask'],
                'sentiment_label': sentiment_label,
                'relation_label': relation_label,
                'entity_positions': entity_positions,
                'position_mask': position_mask
            })
    
    return processed_examples


def compute_cisa_metrics(pred):
    """Calculate CISA evaluation metrics for both sentiment and relation tasks."""
    # Unpack predictions (tuple of sentiment_logits, relation_logits)
    sentiment_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    relation_logits = pred.predictions[1] if isinstance(pred.predictions, tuple) else None
    
    # Unpack labels (tuple of sentiment_labels, relation_labels)
    sentiment_labels = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids
    relation_labels = pred.label_ids[1] if isinstance(pred.label_ids, tuple) else None
    
    # Sentiment predictions and metrics
    sentiment_preds = np.argmax(sentiment_logits, axis=1)
    sentiment_acc = accuracy_score(sentiment_labels, sentiment_preds)
    s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
        sentiment_labels, sentiment_preds, average='weighted'
    )
    
    sentiment_cm = confusion_matrix(sentiment_labels, sentiment_preds)
    
    metrics = {
        'sentiment_accuracy': float(sentiment_acc),
        'sentiment_precision': float(s_prec),
        'sentiment_recall': float(s_rec),
        'sentiment_f1': float(s_f1),
        'sentiment_confusion_matrix': sentiment_cm.tolist()
    }
    
    # Relation predictions and metrics (if available)
    if relation_logits is not None and relation_labels is not None:
        relation_preds = np.argmax(relation_logits, axis=1)
        relation_acc = accuracy_score(relation_labels, relation_preds)
        r_prec, r_rec, r_f1, _ = precision_recall_fscore_support(
            relation_labels, relation_preds, average='binary'
        )
        
        relation_cm = confusion_matrix(relation_labels, relation_preds)
        
        metrics.update({
            'relation_accuracy': float(relation_acc),
            'relation_precision': float(r_prec),
            'relation_recall': float(r_rec),
            'relation_f1': float(r_f1),
            'relation_confusion_matrix': relation_cm.tolist(),
            'combined_f1': float((s_f1 + r_f1) / 2)
        })
    
    return metrics


def train_cisa_model(json_file_path, model_name=None, output_dir="./cisa_model"):
    """
    Complete CISA/CISA training pipeline.
    
    Args:
        json_file_path (str): Path to CISA training data
        model_name (str): Pretrained model name
        output_dir (str): Output directory
        
    Returns:
        tuple: (trainer, tokenizer)
    """
    set_seed(CISA_CONFIG["seed"])
    
    if model_name is None:
        model_name = CISA_CONFIG["model_name"]
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cisa_experiment_{timestamp}"
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Load data
        logging.info(f"Loading CISA data from {json_file_path}...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not raw_data:
            logging.error("Empty dataset!")
            return None, None
        
        logging.info(f"Total samples: {len(raw_data)}")
        
        # Split data: train/val/test (70%/10%/20%)
        train_val_data, test_data = train_test_split(
            raw_data, test_size=0.2, random_state=42
        )
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
        )
        
        logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save test set
        with open(os.path.join(experiment_dir, 'test_data.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Load tokenizer
        logging.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare datasets
        train_examples = prepare_cisa_dataset(train_data, tokenizer, CISA_CONFIG["max_length"])
        val_examples = prepare_cisa_dataset(val_data, tokenizer, CISA_CONFIG["max_length"])
        test_examples = prepare_cisa_dataset(test_data, tokenizer, CISA_CONFIG["max_length"])
        
        logging.info(f"Processed examples - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        # Initialize model
        logging.info("Initializing DECA-CISA model...")
        model = PositionAwareDualEncoderCISA(
            model_name=model_name,
            num_sentiment_labels=CISA_CONFIG["num_labels"],
            dropout_rate=CISA_CONFIG["dropout_rate"],
            use_r_drop=CISA_CONFIG["use_r_drop"],
            stochastic_depth_rate=CISA_CONFIG["stochastic_depth_rate"]
        )
        
        # Data collator
        data_collator = PositionAwareDataCollator(
            tokenizer=tokenizer,
            r_drop=CISA_CONFIG["use_r_drop"],
            max_length=CISA_CONFIG["max_length"]
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            num_train_epochs=CISA_CONFIG["num_train_epochs"],
            per_device_train_batch_size=CISA_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=CISA_CONFIG["per_device_eval_batch_size"],
            learning_rate=CISA_CONFIG["learning_rate"],
            weight_decay=CISA_CONFIG["weight_decay"],
            warmup_ratio=CISA_CONFIG["warmup_ratio"],
            eval_strategy=CISA_CONFIG["evaluation_strategy"],
            save_strategy=CISA_CONFIG["save_strategy"],
            load_best_model_at_end=CISA_CONFIG["load_best_model_at_end"],
            metric_for_best_model=CISA_CONFIG["metric_for_best_model"],
            greater_is_better=CISA_CONFIG["greater_is_better"],
            fp16=CISA_CONFIG["fp16"] and torch.cuda.is_available(),
            gradient_accumulation_steps=CISA_CONFIG["gradient_accumulation_steps"],
            logging_steps=CISA_CONFIG["logging_steps"],
            save_total_limit=CISA_CONFIG["save_total_limit"],
            report_to=["tensorboard"],
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_examples,
            eval_dataset=val_examples,  # Use validation set, not test!
            data_collator=data_collator,
            compute_metrics=compute_cisa_metrics,  # Add metrics computation
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=10,
                    early_stopping_threshold=0.0001
                )
            ]
        )
        
        # Train
        logging.info("Starting CISA training...")
        trainer.train()
        
        # Save model
        logging.info(f"Saving model to {experiment_dir}")
        model.save_pretrained(experiment_dir)
        tokenizer.save_pretrained(experiment_dir)
        
        # Save config
        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(CISA_CONFIG, f, indent=2)
        
        logging.info("Training complete!")
        return trainer, tokenizer
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    train_cisa_model("path/to/cisa_data.json")
