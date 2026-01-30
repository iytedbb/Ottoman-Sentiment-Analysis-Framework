"""
NER Model Configuration
=======================

Configuration parameters for Named Entity Recognition model.
"""

# Label mapping for NER entities
LABEL_MAP = {
    "O": 0,
    "PERSON": 1,
    "LOC": 2,
    "ORG": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Model configuration
NER_CONFIG = {
    "model_name": "dbmdz/bert-base-turkish-cased",
    "max_length": 256,
    "num_labels": 4,
    "label2id": LABEL_MAP,
    "id2label": ID_TO_LABEL,
    
    # Training parameters
    "learning_rate": 3e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,  # Mixed precision training
    "gradient_accumulation_steps": 2,
    
    # Focal Loss parameters
    "use_focal_loss": True,
    "focal_loss_gamma": 2.0,
    "class_weight_beta": 0.999,  # For Effective Number of Samples
    
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    
    # Evaluation
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "entity_macro_f1",
    "greater_is_better": True,
    
    # Logging
    "logging_dir": "./logs",
    "logging_steps": 50,
    "save_total_limit": 2,
    
    # Reproducibility
    "seed": 42,
    
    # Data split
    "test_size": 0.2,
    "stratify": True,
}

# Text normalization patterns
NORMALIZATION_PATTERNS = {
    'â': 'a',
    'î': 'i',
    'û': 'u',
}
