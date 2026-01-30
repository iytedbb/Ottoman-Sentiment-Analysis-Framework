"""
NER Model Architecture
======================

Neural network architectures and components for Named Entity Recognition.

Components:
    - NERDataset: Custom dataset class for token classification
    - FocalLoss: Focal loss for handling class imbalance
    - CustomTrainer: Enhanced Hugging Face trainer with layer-wise learning rates
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer
from .config import LABEL_MAP


class NERDataset(Dataset):
    """
    Custom dataset for NER task with efficient token-entity alignment.
    
    Handles Turkish text tokenization and entity labeling for BERT-based models.
    Uses offset mapping for accurate character-to-token alignment.
    """
    
    def __init__(self, data, tokenizer, max_length=256):
        """
        Args:
            data (list): List of dictionaries with 'text' and 'entities' keys
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = LABEL_MAP

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item['entities']
        
        # Tokenize with offset mapping for entity alignment
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # Extract offset mapping and squeeze tensors
        offset_mapping = encoding.pop("offset_mapping").squeeze(0)
        
        for k in encoding:
            encoding[k] = encoding[k].squeeze(0)
        
        # Initialize labels with -100 (ignore index for special tokens)
        labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        # Label valid tokens as "O" (non-entity) by default
        attention_mask = encoding['attention_mask']
        for idx, (is_valid, (token_start, token_end)) in enumerate(zip(attention_mask, offset_mapping)):
            if is_valid and token_start != token_end:  # Real token (not padding/special)
                labels[idx] = 0  # O (non-entity)
        
        # Mark entity tokens
        for start, end, label in entities:
            for idx, (is_valid, (token_start, token_end)) in enumerate(zip(attention_mask, offset_mapping)):
                if is_valid and token_start != token_end:  # Real token
                    # Check if token overlaps with entity span
                    if (token_start >= start and token_end <= end) or \
                       (token_start < end and token_end > start):
                        labels[idx] = self.label2id[label]
        
        encoding['labels'] = labels
        
        return encoding


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance in NER.
    
    Focuses training on hard-to-classify examples by down-weighting
    easy negatives. Particularly useful when "O" (non-entity) class
    is much more frequent than entity classes.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha (Tensor): Class weights tensor of shape (num_classes,)
        gamma (float): Focusing parameter. Higher values focus more on hard examples
        ignore_index (int): Label value to ignore (e.g., -100 for padding)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predictions of shape (batch_size, num_classes)
            targets (Tensor): Ground truth labels of shape (batch_size,)
            
        Returns:
            Tensor: Computed focal loss
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=self.alpha,
            ignore_index=self.ignore_index
        )
        
        # Compute focal term: (1 - pt)^gamma
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Mask out ignore_index
        mask = targets != self.ignore_index
        focal_loss = focal_loss * mask.float()
        
        if self.reduction == 'mean':
            return focal_loss.sum() / mask.sum().clamp(min=1.0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomTrainer(Trainer):
    """
    Enhanced Trainer with advanced optimization techniques.
    
    Features:
        - Focal Loss integration
        - Layer-wise learning rates (lower for BERT, higher for classifier)
        - OneCycle learning rate scheduling
        - Class weight support for imbalanced data
    """
    
    def __init__(self, *args, class_weights=None, focal_loss_gamma=2.0, **kwargs):
        """
        Args:
            class_weights (Tensor): Weights for each class
            focal_loss_gamma (float): Focal loss gamma parameter
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss_gamma = focal_loss_gamma
        self.loss_fn = FocalLoss(
            alpha=self.class_weights,
            gamma=self.focal_loss_gamma,
            ignore_index=-100
        ) if focal_loss_gamma > 0 else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation with Focal Loss support.
        
        Args:
            model: The model being trained
            inputs (dict): Input tensors including labels
            return_outputs (bool): Whether to return model outputs
            **kwargs: Additional arguments (e.g., num_items_in_batch)
            
        Returns:
            Tensor or tuple: Loss value, optionally with outputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.loss_fn is not None:
            # Use Focal Loss
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, model.config.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(0).type_as(labels)
            )
            
            loss = self.loss_fn(active_logits, active_labels)
        else:
            # Use standard CrossEntropyLoss with class weights
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=-100
            )
            
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, model.config.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            
            loss = loss_fct(active_logits, active_labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer(self):
        """
        Create AdamW optimizer with layer-wise learning rates.
        
        BERT layers use half the learning rate of the classifier layer
        to prevent catastrophic forgetting of pretrained knowledge.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            
            # Group parameters with different learning rates
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay) and "classifier" not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate / 2.0,  # Lower LR for BERT
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay) and "classifier" in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,  # Full LR for classifier
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate / 2.0,  # Lower LR for biases
                },
            ]
            
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Create OneCycle learning rate scheduler.
        
        OneCycle policy: linearly increases LR from initial to max (warmup 10%),
        then decreases with cosine annealing to very low value.
        
        Args:
            num_training_steps (int): Total training steps
            optimizer: Optimizer instance (uses self.optimizer if None)
            
        Returns:
            torch.optim.lr_scheduler: OneCycleLR scheduler
        """
        if optimizer is None:
            optimizer = self.optimizer
            
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.args.learning_rate,
            pct_start=0.1,  # 10% warmup
            total_steps=num_training_steps,
            anneal_strategy='cos',  # Cosine annealing
            cycle_momentum=True,
            div_factor=10.0,  # initial_lr = max_lr/10
            final_div_factor=100.0,  # min_lr = initial_lr/100
        )
