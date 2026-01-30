"""
CISA/CISA Model Configuration
==============================

Configuration for Entity-Based Sentiment Analysis (Cross-Individual).
"""

# CISA sentiment labels
CISA_LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID_TO_CISA_LABEL = {v: k for k, v in CISA_LABELS.items()}

# Model configuration
CISA_CONFIG = {
    "model_name": "dbmdz/bert-base-turkish-cased",
    "num_labels": 3,
    "max_length": 256,
    "dropout_rate": 0.1,
    
    # Dual encoder architecture
    "use_dual_encoder": True,
    "entity_encoder_layers": 2,
    
    # Turkish linguistic features
    "use_linguistic_features": True,
    "linguistic_feature_dim": 64,
    
    # Enhanced attention
    "attention_heads": 12,
    "attention_dropout": 0.1,
    
    # Training parameters
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "gradient_accumulation_steps": 4,
    
    # R-Drop
    "use_r_drop": True,
    "r_drop_alpha": 0.3,
    
    # Stochastic depth
    "stochastic_depth_rate": 0.1,
    
    # Adaptive focal loss
    "use_adaptive_focal_loss": True,
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,
    "difficulty_weight": True,
    
    # Data augmentation
    "use_augmentation": True,
    "augmentation_probability": 0.3,
    
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

# Default class weights
DEFAULT_CISA_CLASS_WEIGHTS = [1.0, 1.0, 1.0]
