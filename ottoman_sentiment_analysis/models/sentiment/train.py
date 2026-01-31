"""
Sentiment Analysis Model Training
==================================

Training pipeline for classical sentiment analysis.
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

from .architecture import EnhancedSentimentBERT, WeightedFocalLoss, AdvancedDataCollator
from .config import SENTIMENT_CONFIG
from ...utils.data_processing import load_json_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class DetailedCallback(TrainerCallback):
    """Callback for detailed training progress tracking."""
    
    def __init__(self):
        self.training_loss = []
        self.eval_metrics = []
        self.best_f1 = 0.0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.training_loss.append({'step': state.global_step, 'loss': logs["loss"]})
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.eval_metrics.append({'step': state.global_step, 'metrics': metrics})
            current_f1 = metrics.get('eval_f1', 0)
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                logging.info(f"New best F1 score: {current_f1:.4f}")


def compute_metrics(pred):
    """Calculate evaluation metrics."""
    logits = pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm.tolist()
    }


def prepare_dataset(data, tokenizer, max_length=256):
    """Prepare dataset for training."""
    dataset_dict = {
        'text': [d['text'] for d in data],
        'labels': [d['label'] for d in data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            return_token_type_ids=False
        )
        tokenized['labels'] = examples['labels']
        return tokenized
    
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    encoded_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    return encoded_dataset


def train_sentiment_model(json_file_path, model_name=None, output_dir="./sentiment_model"):
    """
    Complete sentiment analysis training pipeline.
    
    Args:
        json_file_path (str): Path to training data JSON
        model_name (str): Pretrained model name
        output_dir (str): Output directory
        
    Returns:
        tuple: (trainer, tokenizer)
    """
    set_seed(SENTIMENT_CONFIG["seed"])
    
    if model_name is None:
        model_name = SENTIMENT_CONFIG["model_name"]
    
    try:
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"sentiment_experiment_{timestamp}"
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Load data
        logging.info(f"Loading data from {json_file_path}...")
        data = load_json_dataset(json_file_path)
        
        if not data:
            logging.error("Empty dataset!")
            return None, None
        
        logging.info(f"Total samples: {len(data)}")
        
        # Split data (70% train, 10% val, 20% test)
        labels = [d['label'] for d in data]
        train_val_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_val_labels = [d['label'] for d in train_val_data]
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.125, random_state=42, stratify=train_val_labels
        )
        
        logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save test set
        with open(os.path.join(experiment_dir, 'test_data.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Load tokenizer
        logging.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare datasets
        train_dataset = prepare_dataset(train_data, tokenizer, SENTIMENT_CONFIG["max_length"])
        val_dataset = prepare_dataset(val_data, tokenizer, SENTIMENT_CONFIG["max_length"])
        
        # Initialize model
        logging.info("Initializing model...")
        model = EnhancedSentimentBERT(
            model_name=model_name,
            num_labels=SENTIMENT_CONFIG["num_labels"],
            dropout_rate=SENTIMENT_CONFIG["dropout_rate"],
            use_r_drop=SENTIMENT_CONFIG["use_r_drop"],
            stochastic_depth_rate=SENTIMENT_CONFIG["stochastic_depth_rate"]
        )
        
        # Data collator with R-Drop support
        data_collator = AdvancedDataCollator(
            tokenizer=tokenizer,
            r_drop=SENTIMENT_CONFIG["use_r_drop"],
            max_length=SENTIMENT_CONFIG["max_length"]
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            num_train_epochs=SENTIMENT_CONFIG["num_train_epochs"],
            per_device_train_batch_size=SENTIMENT_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=SENTIMENT_CONFIG["per_device_eval_batch_size"],
            learning_rate=SENTIMENT_CONFIG["learning_rate"],
            weight_decay=SENTIMENT_CONFIG["weight_decay"],
            warmup_ratio=SENTIMENT_CONFIG["warmup_ratio"],
            gradient_accumulation_steps=SENTIMENT_CONFIG["gradient_accumulation_steps"],
            eval_strategy=SENTIMENT_CONFIG["evaluation_strategy"],
            eval_steps=SENTIMENT_CONFIG.get("eval_steps", 100),
            save_strategy=SENTIMENT_CONFIG["save_strategy"],
            save_steps=SENTIMENT_CONFIG.get("save_steps", 200),
            load_best_model_at_end=SENTIMENT_CONFIG["load_best_model_at_end"],
            metric_for_best_model=SENTIMENT_CONFIG["metric_for_best_model"],
            greater_is_better=SENTIMENT_CONFIG["greater_is_better"],
            fp16=SENTIMENT_CONFIG["fp16"] and torch.cuda.is_available(),
            logging_steps=SENTIMENT_CONFIG["logging_steps"],
            save_total_limit=SENTIMENT_CONFIG["save_total_limit"],
            report_to=["tensorboard"],
        )
        
        # Callbacks
        detailed_callback = DetailedCallback()
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=6,
            early_stopping_threshold=0.0001
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[detailed_callback, early_stopping_callback]
        )
        
        # Train
        logging.info("Starting training...")
        trainer.train()
        
        # Save model
        logging.info(f"Saving model to {experiment_dir}")
        model.save_pretrained(experiment_dir)
        tokenizer.save_pretrained(experiment_dir)
        
        # Evaluate
        logging.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_results}")
        
        # Save config
        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(SENTIMENT_CONFIG, f, indent=2)
        
        logging.info("Training complete!")
        return trainer, tokenizer
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    train_sentiment_model("path/to/sentiment_data.json")
