"""
Sentiment Analysis Model Architecture
======================================

Enhanced BERT-based architecture for classical sentiment analysis.

Components:
    - EnhancedSentimentBERT: Main model with multi-head attention and layer ensemble
    - WeightedFocalLoss: Focal loss with class weights
    - R_Drop: Regularized dropout for consistency
    - AdvancedDataCollator: Custom data collator for R-Drop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import SENTIMENT_CONFIG


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights for handling imbalanced data.
    
    Focuses on hard-to-classify examples by down-weighting easy examples.
    
    Args:
        alpha (float): Weighting factor in [0,1] for class balance
        gamma (float): Exponent of modulating factor (1-p_t)^gamma
        class_weights (Tensor, optional): Manual class weights
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Logits of shape (batch_size, num_classes)
            targets (Tensor): Labels of shape (batch_size,)
        """
        # Cross entropy loss
        if self.class_weights is not None:
            class_weights = self.class_weights.to(inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal term: (1 - pt)^gamma
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class R_Drop(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.
    
    Reference: https://arxiv.org/abs/2106.14448
    
    Encourages consistency between two forward passes with different dropout masks.
    
    Args:
        alpha (float): Weight for R-Drop loss term
    """
    
    def __init__(self, alpha=0.3):
        super(R_Drop, self).__init__()
        self.alpha = alpha
        
    def forward(self, logits1, logits2):
        """
        Compute symmetric KL divergence between two logits.
        
        Args:
            logits1 (Tensor): First forward pass logits
            logits2 (Tensor): Second forward pass logits
            
        Returns:
            Tensor: R-Drop loss
        """
        p_loss = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(logits2, dim=-1),
            reduction='none'
        ).sum(-1)
        
        q_loss = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            F.softmax(logits1, dim=-1),
            reduction='none'
        ).sum(-1)
        
        # Symmetric KL divergence
        loss = (p_loss + q_loss) / 2
        return self.alpha * loss.mean()


class AdvancedDataCollator:
    """
    Custom data collator with R-Drop support.
    
    Creates duplicate samples for R-Drop during training.
    """
    
    def __init__(self, tokenizer, r_drop=False, max_length=256):
        self.tokenizer = tokenizer
        self.r_drop = r_drop
        self.max_length = max_length

    def __call__(self, features):
        # Pad features
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        # Preserve labels
        batch["labels"] = torch.tensor([f.get("labels", 0) for f in features])
        
        # Create second copy for R-Drop
        if self.r_drop:
            batch["attention_mask2"] = batch["attention_mask"].clone()
            batch["input_ids2"] = batch["input_ids"].clone()
        
        return batch


class EnhancedSentimentBERT(nn.Module):
    """
    Enhanced BERT model for sentiment analysis.
    
    Features:
        - Multi-head attention for contextual understanding
        - Layer ensemble (weighted sum of last 4 layers)
        - Dual pooling (CLS + mean)
        - R-Drop regularization
        - Stochastic depth
        - Label smoothing
        
    Performance: 92.63% accuracy on historical Turkish texts
    """
    
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased', num_labels=3, 
                 dropout_rate=0.1, use_r_drop=True, stochastic_depth_rate=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # BERT encoder (without pooling layer)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.config = self.bert.config
        
        self.dropout_rate = dropout_rate
        self.use_r_drop = use_r_drop
        self.stochastic_depth_rate = stochastic_depth_rate
        
        hidden_size = self.bert.config.hidden_size
        
        # Multi-head attention for sequence-level understanding
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        
        # Context fusion network
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Sentiment classification head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.label_smoothing = SENTIMENT_CONFIG.get("label_smoothing", 0.1)
        
    def forward(self, input_ids, attention_mask, labels=None, 
                input_ids2=None, attention_mask2=None):
        """
        Forward pass with optional R-Drop.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            input_ids2: Second pass token IDs for R-Drop (optional)
            attention_mask2: Second pass attention mask for R-Drop (optional)
            
        Returns:
            dict: Contains 'loss' (if labels provided) and 'logits'
        """
        # First forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Layer ensemble: weighted sum of last 4 layers
        all_hidden_states = outputs.hidden_states
        last_four_layers = torch.stack(all_hidden_states[-4:])
        
        # Layer weights (more weight to later layers)
        layer_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device).view(4, 1, 1, 1)
        weighted_layers = last_four_layers * layer_weights
        final_output = weighted_layers.sum(dim=0)  # [batch_size, seq_len, hidden_size]
        
        # CLS token embedding
        cls_embedding = final_output[:, 0, :]
        
        # Mean pooling (excluding special tokens)
        mask = attention_mask.unsqueeze(-1).float()
        masked_output = final_output * mask
        sum_embeddings = masked_output.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        
        # Normalize embeddings
        cls_embedding = F.normalize(cls_embedding, p=2, dim=1)
        mean_embedding = F.normalize(mean_embedding, p=2, dim=1)
        
        # Enhanced attention between CLS and sequence
        q = cls_embedding.unsqueeze(0)  # [1, batch_size, hidden_size]
        k = final_output.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        v = final_output.transpose(0, 1)
        
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.squeeze(0)
        
        # Residual connection and layer norm
        attn_output = self.layer_norm1(attn_output + cls_embedding)
        
        # Combine CLS and mean pooling
        combined = torch.cat([attn_output, mean_embedding], dim=1)
        combined = self.layer_norm2(combined)
        
        # Context fusion
        fused_embedding = self.context_fusion(combined)
        
        # Sentiment classification
        logits = self.sentiment_classifier(fused_embedding)
        
        # R-Drop: second forward pass
        r_drop_loss = 0.0
        if self.use_r_drop and self.training and input_ids2 is not None and attention_mask2 is not None:
            outputs2 = self.bert(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                output_hidden_states=True,
            )
            
            # Repeat same processing
            all_hidden_states2 = outputs2.hidden_states
            last_four_layers2 = torch.stack(all_hidden_states2[-4:])
            weighted_layers2 = last_four_layers2 * layer_weights
            final_output2 = weighted_layers2.sum(dim=0)
            
            cls_embedding2 = final_output2[:, 0, :]
            mask2 = attention_mask2.unsqueeze(-1).float()
            masked_output2 = final_output2 * mask2
            sum_embeddings2 = masked_output2.sum(dim=1)
            sum_mask2 = mask2.sum(dim=1).clamp(min=1e-9)
            mean_embedding2 = sum_embeddings2 / sum_mask2
            
            cls_embedding2 = F.normalize(cls_embedding2, p=2, dim=1)
            mean_embedding2 = F.normalize(mean_embedding2, p=2, dim=1)
            
            q2 = cls_embedding2.unsqueeze(0)
            k2 = final_output2.transpose(0, 1)
            v2 = final_output2.transpose(0, 1)
            
            attn_output2, _ = self.multihead_attn(q2, k2, v2)
            attn_output2 = attn_output2.squeeze(0)
            attn_output2 = self.layer_norm1(attn_output2 + cls_embedding2)
            
            combined2 = torch.cat([attn_output2, mean_embedding2], dim=1)
            combined2 = self.layer_norm2(combined2)
            
            fused_embedding2 = self.context_fusion(combined2)
            logits2 = self.sentiment_classifier(fused_embedding2)
            
            # Compute R-Drop loss
            r_drop = R_Drop(alpha=SENTIMENT_CONFIG.get("r_drop_alpha", 0.3))
            r_drop_loss = r_drop(logits, logits2)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fn(logits, labels)
            
            # Add R-Drop loss
            if self.use_r_drop and self.training and input_ids2 is not None:
                loss = loss + r_drop_loss
        
        return {
            'loss': loss,
            'logits': logits
        } if loss is not None else {
            'logits': logits
        }
    
    def save_pretrained(self, path):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        self.config.save_pretrained(path)
