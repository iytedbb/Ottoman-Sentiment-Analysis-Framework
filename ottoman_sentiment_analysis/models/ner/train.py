"""
NER Model Training
==================

Training pipeline for Named Entity Recognition model.

Main Functions:
    - train_ner_model: Complete training pipeline
    - calculate_class_weights: Compute class weights for imbalanced data
    - compute_metrics: Evaluation metrics calculation
"""

import os
import json
import logging
import hashlib
import random
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    EarlyStoppingCallback
)

from .architecture import NERDataset, CustomTrainer
from .config import NER_CONFIG, NORMALIZATION_PATTERNS
from ...utils.data_processing import normalize_text, load_json_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_class_weights(data):
    """
    Calculate class weights using Effective Number of Samples.
    
    Addresses class imbalance by computing weights based on the
    effective number of samples per class.
    
    Args:
        data (list): Training data with 'text' and 'entities' fields
        
    Returns:
        dict: Mapping from label names to weight values
    """
    label_counts = {
        "O": 0,
        "PERSON": 0,
        "LOC": 0,
        "ORG": 0
    }
    
    # Count tokens for each label
    for item in data:
        tokens = item['text'].split()
        token_labels = ["O"] * len(tokens)
        
        # Mark entity tokens
        for start, end, label in item['entities']:
            start_idx = len(item['text'][:start].split())
            end_idx = len(item['text'][:end].split())
            
            for i in range(start_idx, end_idx):
                if i < len(token_labels):
                    token_labels[i] = label
        
        for label in token_labels:
            label_counts[label] += 1
    
    logging.info(f"Token-level label distribution: {label_counts}")
    
    # Effective Number of Samples formula
    beta = NER_CONFIG["class_weight_beta"]
    effective_samples = {}
    for label, count in label_counts.items():
        if count > 0:
            effective_samples[label] = (1 - beta**count) / (1 - beta)
        else:
            effective_samples[label] = 1.0
    
    # Calculate weights
    total_effective = sum(effective_samples.values())
    class_weights = {
        label: total_effective / (eff * len(label_counts))
        for label, eff in effective_samples.items()
    }
    
    # Normalize (min weight = 1.0)
    min_weight = min(class_weights.values())
    class_weights = {k: v/min_weight for k, v in class_weights.items()}
    
    logging.info(f"Calculated class weights: {class_weights}")
    
    return class_weights


def clean_and_validate_entities(data):
    """
    Validate and clean entity annotations.
    
    Removes invalid entities and fixes boundary issues.
    
    Args:
        data (list): Raw dataset
        
    Returns:
        list: Cleaned dataset
    """
    cleaned_data = []
    skipped_count = 0
    fixed_count = 0
    
    for item in data:
        valid_entities = []
        text_length = len(item['text'])
        
        for start, end, label in item['entities']:
            # Check boundaries
            if start < 0 or end > text_length or start >= end:
                if start < 0:
                    start = 0
                    fixed_count += 1
                if end > text_length:
                    end = text_length
                    fixed_count += 1
                if start >= end:
                    skipped_count += 1
                    continue
            
            # Check label validity
            if label not in ["PERSON", "LOC", "ORG"]:
                skipped_count += 1
                continue
            
            valid_entities.append((start, end, label))
        
        # Skip examples with no entities if dataset is large
        if len(valid_entities) == 0 and len(data) > 1000:
            skipped_count += 1
            continue
        
        item_copy = item.copy()
        item_copy['entities'] = valid_entities
        cleaned_data.append(item_copy)
    
    logging.info(f"Cleaning: {skipped_count} skipped, {fixed_count} fixed")
    logging.info(f"Samples after cleaning: {len(cleaned_data)}")
    
    return cleaned_data


def check_data_separation(train_data, test_data):
    """
    Verify train and test sets are completely separate.
    
    Uses content hashing to detect duplicates.
    
    Returns:
        bool: True if sets are separate, False if overlap detected
    """
    def get_content_hash(item):
        content = normalize_text(item['text'].lower().strip())
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    train_hashes = {get_content_hash(item) for item in train_data}
    test_hashes = {get_content_hash(item) for item in test_data}
    
    intersection = train_hashes.intersection(test_hashes)
    if intersection:
        logging.warning(f"WARNING: {len(intersection)} overlapping samples!")
        return False
    else:
        logging.info("Train and test sets are completely separate")
        return True


def print_data_statistics(train_data, test_data):
    """Print detailed dataset statistics."""
    for dataset_name, dataset in [("Train", train_data), ("Test", test_data)]:
        entity_counts = {
            "PERSON": sum(1 for item in dataset if any(ent[2] == "PERSON" for ent in item['entities'])),
            "LOC": sum(1 for item in dataset if any(ent[2] == "LOC" for ent in item['entities'])),
            "ORG": sum(1 for item in dataset if any(ent[2] == "ORG" for ent in item['entities']))
        }
        
        no_entity_count = sum(1 for item in dataset if len(item['entities']) == 0)
        
        logging.info(f"\n{dataset_name} set statistics:")
        logging.info(f"Total samples: {len(dataset)}")
        logging.info(f"Samples without entities: {no_entity_count} ({no_entity_count/len(dataset)*100:.2f}%)")
        
        for entity_type, count in entity_counts.items():
            percentage = count/len(dataset)*100 if len(dataset) > 0 else 0
            logging.info(f"{entity_type} samples: {count} ({percentage:.2f}%)")
        
        # Entity count distribution
        all_entities = []
        for item in dataset:
            all_entities.extend([ent[2] for ent in item['entities']])
        
        entity_counter = Counter(all_entities)
        logging.info(f"Total entity distribution: {dict(entity_counter)}")


def compute_metrics(pred):
    """
    Calculate evaluation metrics for NER.
    
    Computes precision, recall, F1 per entity type and overall metrics.
    
    Args:
        pred: Predictions object from Trainer
        
    Returns:
        dict: Metrics dictionary
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Create mask to ignore padding (-100)
    mask = labels != -100
    
    true_predictions = predictions[mask]
    true_labels = labels[mask]
    
    label_names = ["O", "PERSON", "LOC", "ORG"]
    all_metrics = {}
    
    try:
        # Confusion matrix
        cm = confusion_matrix(true_labels, true_predictions)
        all_metrics['confusion_matrix'] = cm.tolist()
        
        # Detailed classification report
        class_report = classification_report(
            true_labels,
            true_predictions,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        
        # Per-class metrics
        for label in label_names:
            prefix = f"{label}_metrics"
            all_metrics[prefix] = {
                'precision': float(class_report[label]['precision']),
                'recall': float(class_report[label]['recall']),
                'f1': float(class_report[label]['f1-score']),
                'support': int(class_report[label]['support'])
            }
        
        # Entity macro F1 (excluding "O")
        entity_f1s = [all_metrics[f"{label}_metrics"]['f1'] for label in label_names if label != "O"]
        all_metrics['entity_macro_f1'] = float(np.mean(entity_f1s))
        
        # Overall metrics
        all_metrics['macro_f1'] = float(class_report['macro avg']['f1-score'])
        all_metrics['weighted_f1'] = float(class_report['weighted avg']['f1-score'])
        
        # Entity-level metrics (average of PERSON, LOC, ORG)
        entity_metrics = {
            'entity_precision': float(np.mean([all_metrics[f"{label}_metrics"]['precision'] 
                                               for label in label_names if label != "O"])),
            'entity_recall': float(np.mean([all_metrics[f"{label}_metrics"]['recall'] 
                                           for label in label_names if label != "O"])),
            'entity_f1': float(np.mean([all_metrics[f"{label}_metrics"]['f1'] 
                                       for label in label_names if label != "O"]))
        }
        all_metrics.update(entity_metrics)
        
        # Log results
        logging.info("\nEvaluation Results:")
        logging.info(f"Entity Macro F1: {all_metrics['entity_macro_f1']:.4f}")
        logging.info(f"Weighted F1: {all_metrics['weighted_f1']:.4f}")
        
        for label in label_names:
            metrics = all_metrics[f"{label}_metrics"]
            logging.info(f"\n{label} Metrics:")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall: {metrics['recall']:.4f}")
            logging.info(f"  F1 Score: {metrics['f1']:.4f}")
        
    except Exception as e:
        logging.error(f"Error computing metrics: {str(e)}")
        correct = (true_predictions == true_labels).sum()
        total = len(true_labels)
        all_metrics['accuracy'] = float(correct / total if total > 0 else 0)
    
    return all_metrics


def train_ner_model(json_file_path, model_name=None, output_dir="./ner_model"):
    """
    Complete NER model training pipeline.
    
    Args:
        json_file_path (str): Path to training data JSON
        model_name (str): Pretrained model name (default from config)
        output_dir (str): Directory to save trained model
        
    Returns:
        tuple: (trainer, tokenizer) or (None, None) on failure
    """
    set_seed(NER_CONFIG["seed"])
    
    # Use config defaults if not specified
    if model_name is None:
        model_name = NER_CONFIG["model_name"]
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"ner_experiment_{timestamp}"
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Load data
        logging.info(f"Loading data from {json_file_path}...")
        data = load_json_dataset(json_file_path)
        
        if not data:
            logging.error(f"Empty dataset: {json_file_path}")
            return None, None
        
        logging.info(f"Total samples (original): {len(data)}")
        
        # Clean and validate
        data = clean_and_validate_entities(data)
        
        # Remove duplicates
        content_hashes = {}
        for item in data:
            text = normalize_text(item['text'].lower().strip())
            content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if content_hash not in content_hashes:
                content_hashes[content_hash] = item
            else:
                # Keep item with more entities
                if len(item['entities']) > len(content_hashes[content_hash]['entities']):
                    content_hashes[content_hash] = item
        
        unique_data = list(content_hashes.values())
        logging.info(f"Unique samples: {len(unique_data)}")
        
        # Split data with stratification
        try:
            def get_entity_types(item):
                entity_types = set(ent[2] for ent in item['entities'])
                return "_".join(sorted(entity_types)) or "NONE"
            
            stratify_labels = [get_entity_types(item) for item in unique_data]
            train_data, test_data = train_test_split(
                unique_data,
                test_size=NER_CONFIG["test_size"],
                random_state=NER_CONFIG["seed"],
                stratify=stratify_labels
            )
            logging.info("Stratified split successful")
        except Exception as e:
            logging.warning(f"Stratified split failed: {e}, using simple split")
            train_data, test_data = train_test_split(
                unique_data,
                test_size=NER_CONFIG["test_size"],
                random_state=NER_CONFIG["seed"]
            )
        
        # Verify separation
        check_data_separation(train_data, test_data)
        
        # Print statistics
        print_data_statistics(train_data, test_data)
        
        # Calculate class weights
        class_weights = calculate_class_weights(train_data)
        class_weights_tensor = torch.tensor([
            class_weights["O"],
            class_weights["PERSON"],
            class_weights["LOC"],
            class_weights["ORG"]
        ])
        
        # Load model and tokenizer
        logging.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=NER_CONFIG["num_labels"],
            label2id=NER_CONFIG["label2id"],
            id2label=NER_CONFIG["id2label"]
        )
        
        # Create datasets
        train_dataset = NERDataset(train_data, tokenizer, NER_CONFIG["max_length"])
        test_dataset = NERDataset(test_data, tokenizer, NER_CONFIG["max_length"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            num_train_epochs=NER_CONFIG["num_train_epochs"],
            per_device_train_batch_size=NER_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=NER_CONFIG["per_device_eval_batch_size"],
            learning_rate=NER_CONFIG["learning_rate"],
            weight_decay=NER_CONFIG["weight_decay"],
            warmup_ratio=NER_CONFIG["warmup_ratio"],
            eval_strategy=NER_CONFIG["evaluation_strategy"],
            save_strategy=NER_CONFIG["save_strategy"],
            load_best_model_at_end=NER_CONFIG["load_best_model_at_end"],
            metric_for_best_model=NER_CONFIG["metric_for_best_model"],
            greater_is_better=NER_CONFIG["greater_is_better"],
            fp16=NER_CONFIG["fp16"] and torch.cuda.is_available(),
            logging_dir=NER_CONFIG["logging_dir"],
            logging_steps=NER_CONFIG["logging_steps"],
            save_total_limit=NER_CONFIG["save_total_limit"],
            gradient_accumulation_steps=NER_CONFIG["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            report_to=["tensorboard"],
        )
        
        # Create trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor if torch.cuda.is_available() 
                          else class_weights_tensor.cpu(),
            focal_loss_gamma=NER_CONFIG["focal_loss_gamma"] if NER_CONFIG["use_focal_loss"] else 0,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=NER_CONFIG["early_stopping_patience"],
                    early_stopping_threshold=NER_CONFIG["early_stopping_threshold"]
                )
            ]
        )
        
        # Train
        logging.info("Starting training...")
        trainer.train()
        
        # Save final model
        logging.info(f"Saving model to {experiment_dir}")
        trainer.save_model(experiment_dir)
        tokenizer.save_pretrained(experiment_dir)
        
        # Evaluate
        logging.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_results}")
        
        # Save config
        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(NER_CONFIG, f, indent=2)
        
        logging.info("Training complete!")
        return trainer, tokenizer
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Example usage
    train_ner_model("path/to/ner_data.json")
