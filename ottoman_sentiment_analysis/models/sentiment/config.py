"""
Sentiment Analysis Model Configuration
=======================================

Configuration parameters for classical sentiment analysis model.
"""

# Sentiment labels
SENTIMENT_LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_LABELS.items()}

# Model configuration
SENTIMENT_CONFIG = {
    "model_name": "dbmdz/bert-base-turkish-cased",
    "num_labels": 3,
    "max_length": 256,
    "dropout_rate": 0.1,
    
    # Training parameters
    "learning_rate": 3e-5,
    "num_train_epochs": 8,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    
    # Advanced features
    "use_r_drop": True,
    "r_drop_alpha": 0.3,
    "stochastic_depth_rate": 0.1,
    "use_adversarial_training": True,
    "adversarial_epsilon": 0.01,
    
    # Focal loss parameters
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,
    
    # Label smoothing
    "label_smoothing": 0.1,
    
    # Evaluation
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    
    # Logging
    "logging_steps": 50,
    "save_total_limit": 2,
    
    # Reproducibility
    "seed": 42,
    
    # Data split
    "test_size": 0.2,
    "validation_size": 0.1,
}

# Class weights (can be recalculated from data)
DEFAULT_CLASS_WEIGHTS = [1.0, 1.0, 1.0]  # Will be updated during training
