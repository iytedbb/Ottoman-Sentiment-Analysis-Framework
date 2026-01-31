"""
CISA/CISA Model Architecture
=============================

Dual-Encoder Context-Aware Entity-Based Sentiment Analysis (DECA-CISA).

This is the core architecture for Cross-Individual Sentiment Analysis (CISA),
analyzing author's sentiment toward specific individuals in historical Turkish texts.

Components:
    - PositionAwareDualEncoderCISA: Main dual-encoder model
    - TurkishLinguisticFeatures: Turkish-specific linguistic patterns
    - EnhancedEntityContextAttention: Advanced entity-context attention
    - ContextualSentimentEncoder: Contextual sentiment encoding
    - AdaptiveFocalLoss: Adaptive focal loss for class imbalance
    - R_Drop: Regularized dropout

Performance: 87.08% accuracy, F1: 87.05% on CISA task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import CISA_CONFIG


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with difficulty weighting for historical Turkish CISA.
    
    Handles class imbalance and focuses on hard-to-classify examples,
    particularly important for nuanced sentiment in historical texts.
    
    Args:
        alpha (Tensor or float): Class weights
        gamma (float): Focusing parameter
        size_average (bool): Whether to average the loss
        difficulty_weight (bool): Apply difficulty-aware weighting
    """
    
    def __init__(self, alpha=None, gamma=2.0, size_average=True, difficulty_weight=True):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.difficulty_weight = difficulty_weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
        else:
            alpha_t = 1.0
            
        # Focal term
        focal_weight = (1 - pt) ** self.gamma
        
        # Difficulty-aware weighting for historical Turkish
        if self.difficulty_weight:
            difficulty_factor = 1 + torch.exp(-pt * 2)
            focal_weight = focal_weight * difficulty_factor
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class R_Drop(nn.Module):
    """
    R-Drop: Regularized Dropout for consistency.
    
    Reference: https://arxiv.org/abs/2106.14448
    """
    
    def __init__(self, alpha=0.3):
        super(R_Drop, self).__init__()
        self.alpha = alpha
        
    def forward(self, logits1, logits2):
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
        
        loss = (p_loss + q_loss) / 2
        return self.alpha * loss.mean()


class TurkishLinguisticFeatures(nn.Module):
    """
    Turkish and Ottoman Turkish linguistic feature extractor.
    
    Captures:
        - Adjective-noun patterns
        - Respect/affection expressions
        - Formality level detection
        - Morphological features
        
    These features are crucial for analyzing historical Turkish memoirs
    where formal language and respect expressions indicate sentiment.
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Turkish adjective-noun attention
        self.adjective_noun_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Ottoman/historical Turkish word projection
        self.historical_word_projection = nn.Linear(hidden_size, hidden_size)
        
        # Respect/affection pattern detector
        self.respect_pattern_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 64)
        )
        
        # Formality level detector
        self.formality_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 32)
        )
        
        # Morphological analyzer
        self.morphological_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 48)
        )
        
        # Linguistic feature fusion
        self.linguistic_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 48, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def detect_turkish_patterns(self, text_repr):
        """Detect Turkish-specific linguistic patterns."""
        respect_features = self.respect_pattern_detector(text_repr)
        formality_features = self.formality_detector(text_repr)
        morphological_features = self.morphological_analyzer(text_repr)
        
        combined_features = torch.cat([
            respect_features,
            formality_features,
            morphological_features
        ], dim=-1)
        
        return self.linguistic_fusion(combined_features)
        
    def forward(self, text_repr, entity_repr):
        """Extract Turkish linguistic features."""
        # Self-attention for adjective-noun patterns
        enhanced_text, _ = self.adjective_noun_attention(
            query=text_repr.unsqueeze(1),
            key=text_repr.unsqueeze(1),
            value=text_repr.unsqueeze(1)
        )
        enhanced_text = enhanced_text.squeeze(1)
        
        # Historical Turkish projection
        historical_features = self.historical_word_projection(enhanced_text)
        combined_repr = enhanced_text + historical_features
        
        # Extract Turkish patterns
        turkish_features = self.detect_turkish_patterns(combined_repr)
        
        return turkish_features


class EnhancedEntityContextAttention(nn.Module):
    """
    Enhanced Entity-Context Attention with position awareness.
    
    Computes attention between entity and surrounding text context,
    with special weighting for tokens near the entity mention.
    
    Features:
        - Multi-head attention
        - Position-aware weighting
        - Local context attention
        - Hierarchical attention fusion
    """
    
    def __init__(self, hidden_size, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Global entity-context attention
        self.entity_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Position embeddings
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # Local context attention
        self.local_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Hierarchical attention fusion
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def create_position_weight_matrix(self, entity_positions, seq_length, device):
        """Create position-based weight matrix."""
        batch_size = len(entity_positions)
        weight_matrix = torch.ones(batch_size, seq_length, device=device)
        
        for i, (start_pos, end_pos) in enumerate(entity_positions):
            # Highest weight for entity tokens
            weight_matrix[i, start_pos:end_pos+1] = 3.0
            
            # Medium weight for surrounding context (±3 tokens)
            context_start = max(0, start_pos - 3)
            context_end = min(seq_length, end_pos + 4)
            weight_matrix[i, context_start:start_pos] = 2.0
            weight_matrix[i, end_pos+1:context_end] = 2.0
            
            # Low weight for distant tokens
            weight_matrix[i, :context_start] = 0.5
            weight_matrix[i, context_end:] = 0.5
            
        return weight_matrix
    
    def forward(self, entity_repr, text_sequence, entity_positions, attention_mask):
        """
        Args:
            entity_repr: [batch_size, hidden_size]
            text_sequence: [batch_size, seq_len, hidden_size]
            entity_positions: List of [start_pos, end_pos] for each sample
            attention_mask: [batch_size, seq_len]
            
        Returns:
            tuple: (attended representation, attention weights)
        """
        batch_size, seq_len, hidden_size = text_sequence.shape
        device = text_sequence.device
        
        # Add position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        enhanced_text = text_sequence + position_emb
        enhanced_text = self.layer_norm1(enhanced_text)
        
        # Position-based weighting
        position_weights = self.create_position_weight_matrix(entity_positions, seq_len, device)
        
        # Global entity-context attention
        entity_query = entity_repr.unsqueeze(1)
        global_attended, global_attention_weights = self.entity_context_attention(
            query=entity_query,
            key=enhanced_text,
            value=enhanced_text,
            key_padding_mask=~attention_mask.bool()
        )
        global_attended = global_attended.squeeze(1)
        
        # Apply position weighting
        weighted_attention = global_attention_weights.squeeze(1) * position_weights
        weighted_attended = torch.bmm(weighted_attention.unsqueeze(1), enhanced_text).squeeze(1)
        
        # Local context attention
        local_context_features = []
        for i, (start_pos, end_pos) in enumerate(entity_positions):
            local_start = max(0, start_pos - 5)
            local_end = min(seq_len, end_pos + 6)
            
            local_context = enhanced_text[i, local_start:local_end].unsqueeze(0)
            entity_q = entity_query[i:i+1]
            
            if local_context.size(1) > 0:
                local_attended, _ = self.local_context_attention(
                    query=entity_q,
                    key=local_context,
                    value=local_context
                )
                local_context_features.append(local_attended.squeeze(1))
            else:
                local_context_features.append(entity_repr[i:i+1])
        
        local_context_repr = torch.cat(local_context_features, dim=0)
        
        # Hierarchical attention fusion
        hierarchical_input = torch.cat([global_attended, weighted_attended, local_context_repr], dim=-1)
        hierarchical_weights = self.hierarchical_attention(hierarchical_input)
        hierarchical_weights = F.softmax(hierarchical_weights.view(-1, 3), dim=-1)
        
        # Final attended representation
        final_attended = (
            hierarchical_weights[:, 0:1] * global_attended +
            hierarchical_weights[:, 1:2] * weighted_attended +
            hierarchical_weights[:, 2:3] * local_context_repr
        )
        
        final_attended = self.layer_norm2(final_attended)
        
        return final_attended, global_attention_weights


class ContextualSentimentEncoder(nn.Module):
    """
    Contextual sentiment encoder with BiLSTM and attention.
    
    Captures sequential context and sentiment-specific patterns.
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        
        # BiLSTM for context modeling
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )
        
        # Sentiment-specific pooling
        self.sentiment_pooling = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        # Context type classifier
        self.context_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )
        
    def forward(self, context_sequence, entity_position_mask):
        """
        Args:
            context_sequence: [batch_size, seq_len, hidden_size]
            entity_position_mask: [batch_size, seq_len]
            
        Returns:
            tuple: (sentiment context repr, context type logits)
        """
        # LSTM context modeling
        lstm_out, (hidden, cell) = self.context_lstm(context_sequence)
        
        # Adaptive pooling based on entity position
        if entity_position_mask is not None:
            masked_context = lstm_out * entity_position_mask.unsqueeze(-1).float()
            pooled_context = masked_context.mean(dim=1)
        else:
            pooled_context = lstm_out.mean(dim=1)
        
        # Sentiment-specific pooling
        sentiment_context, _ = self.sentiment_pooling(
            query=pooled_context.unsqueeze(1),
            key=lstm_out,
            value=lstm_out
        )
        
        # Context type classification
        context_type_logits = self.context_type_classifier(pooled_context)
        
        return sentiment_context.squeeze(1), context_type_logits


class PositionAwareDualEncoderCISA(nn.Module):
    """
    Position-Aware Dual-Encoder Entity-Based Sentiment Analysis model.
    
    Architecture: DECA-CISA (Dual-Encoder Context-Aware CISA)
    
    Uses two BERT encoders:
        1. Text encoder: Processes full text
        2. Entity encoder: Processes entity mentions
        
    Key features:
        - Dual encoder architecture
        - Turkish linguistic features
        - Enhanced entity-context attention
        - Position-aware representations
        - R-Drop regularization
        
    Example use case:
        Text: "Ali Bey'in vefatı bizleri hüzne boğmuştu"
        Entity: "Ali Bey"
        Standard Sentiment: Negative (sad text)
        CISA Output: Positive (author's respect for Ali Bey)
    """
    
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased', num_sentiment_labels=3,
                 dropout_rate=0.1, use_r_drop=True, stochastic_depth_rate=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dual BERT encoders
        self.text_encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.entity_encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.config = self.text_encoder.config
        
        self.dropout_rate = dropout_rate
        self.use_r_drop = use_r_drop
        self.clip_grad_norm = 1.0
        self.stochastic_depth_rate = stochastic_depth_rate
        
        hidden_size = self.text_encoder.config.hidden_size
        self.bert_hidden_size = hidden_size
        
        # Enhanced components
        self.enhanced_attention = EnhancedEntityContextAttention(
            hidden_size=hidden_size,
            num_heads=12,
            dropout=dropout_rate
        )
        
        self.turkish_linguistic = TurkishLinguisticFeatures(hidden_size)
        self.contextual_encoder = ContextualSentimentEncoder(hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.entity_position_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Enhanced fusion network
        fusion_input_size = hidden_size * 3 + 64  # +64 for Turkish features
        self.enhanced_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Task-specific heads
        classifier_input_size = hidden_size + 2  # +2 for context type
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_sentiment_labels)
        )
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2)
        )
        
        self.label_smoothing = 0.1
        
    def encode_with_weighted_layers(self, encoder, input_ids, attention_mask):
        """Encode with layer ensemble (weighted sum of last 4 layers)."""
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Weighted sum of last 4 layers
        all_hidden_states = outputs.hidden_states
        last_four_layers = torch.stack(all_hidden_states[-4:])
        
        layer_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device).view(4, 1, 1, 1)
        weighted_layers = last_four_layers * layer_weights
        final_output = weighted_layers.sum(dim=0)
        
        return final_output, outputs.last_hidden_state
    
    def extract_entity_aware_representation(self, text_output, entity_positions, position_mask):
        """Extract entity representation using position masks."""
        batch_size = text_output.shape[0]
        entity_representations = []
        
        for i in range(batch_size):
            start_pos, end_pos = entity_positions[i]
            pos_mask = torch.tensor(position_mask[i], device=text_output.device, dtype=torch.bool)
            
            if pos_mask.any():
                entity_tokens = text_output[i][pos_mask]
                if entity_tokens.shape[0] > 0:
                    entity_repr = entity_tokens.mean(dim=0)
                else:
                    entity_repr = text_output[i, 0]
            else:
                entity_repr = text_output[i, 0]
            
            entity_representations.append(entity_repr)
        
        return torch.stack(entity_representations)
    
    def forward(self, text_input_ids, text_attention_mask, entity_input_ids, entity_attention_mask,
                entity_positions, position_mask, sentiment_label=None, relation_label=None,
                text_input_ids2=None, text_attention_mask2=None,
                entity_input_ids2=None, entity_attention_mask2=None, **kwargs):
        """
        Forward pass.
        
        Args:
            text_input_ids: Tokenized full text
            text_attention_mask: Text attention mask
            entity_input_ids: Tokenized entity mention
            entity_attention_mask: Entity attention mask
            entity_positions: List of (start, end) positions
            position_mask: Boolean masks for entity tokens
            sentiment_label: CISA sentiment labels (optional)
            relation_label: Relation labels (optional)
            *2 arguments: For R-Drop second pass
            
        Returns:
            dict: Contains logits, loss (if labels provided), attention weights
        """
        batch_size = text_input_ids.shape[0]
        
        # Dual encoder forward
        text_output, _ = self.encode_with_weighted_layers(
            self.text_encoder, text_input_ids, text_attention_mask
        )
        
        entity_output, _ = self.encode_with_weighted_layers(
            self.entity_encoder, entity_input_ids, entity_attention_mask
        )
        
        # CLS representations
        text_cls = text_output[:, 0, :]
        entity_cls = entity_output[:, 0, :]
        
        # Position-aware entity representation
        entity_aware_repr = self.extract_entity_aware_representation(
            text_output, entity_positions, position_mask
        )
        
        # Enhanced cross-attention
        cross_attended, attention_weights = self.enhanced_attention(
            entity_cls, text_output, entity_positions, text_attention_mask
        )
        
        # Turkish linguistic features
        turkish_features = self.turkish_linguistic(text_cls, entity_cls)
        
        # Contextual encoding
        context_repr, context_type_logits = self.contextual_encoder(
            text_output,
            torch.stack([torch.tensor(pm, device=self.device) for pm in position_mask])
        )
        
        # Normalize
        text_cls = F.normalize(text_cls, p=2, dim=1)
        entity_cls = F.normalize(entity_cls, p=2, dim=1)
        cross_attended = F.normalize(cross_attended, p=2, dim=1)
        
        # Layer norm and residual
        entity_cls = self.layer_norm1(entity_cls)
        cross_attended = self.layer_norm2(cross_attended + entity_cls)
        
        # Enhanced fusion
        comprehensive_repr = torch.cat([
            text_cls, entity_cls, cross_attended, turkish_features
        ], dim=1)
        final_representation = self.enhanced_fusion(comprehensive_repr)
        
        # Classifier input
        context_type_probs = F.softmax(context_type_logits, dim=-1)
        classifier_input = torch.cat([final_representation, context_type_probs], dim=1)
        
        # Task outputs
        sentiment_logits = self.sentiment_classifier(classifier_input)
        relation_logits = self.relation_classifier(classifier_input)
        
        # R-Drop second pass
        r_drop_loss = 0.0
        if self.use_r_drop and self.training and text_input_ids2 is not None:
            text_output2, _ = self.encode_with_weighted_layers(
                self.text_encoder, text_input_ids2, text_attention_mask2
            )
            entity_output2, _ = self.encode_with_weighted_layers(
                self.entity_encoder, entity_input_ids2, entity_attention_mask2
            )
            
            text_cls2 = text_output2[:, 0, :]
            entity_cls2 = entity_output2[:, 0, :]
            
            cross_attended2, _ = self.enhanced_attention(
                entity_cls2, text_output2, entity_positions, text_attention_mask2
            )
            
            turkish_features2 = self.turkish_linguistic(text_cls2, entity_cls2)
            context_repr2, context_type_logits2 = self.contextual_encoder(
                text_output2,
                torch.stack([torch.tensor(pm, device=self.device) for pm in position_mask])
            )
            
            text_cls2 = F.normalize(text_cls2, p=2, dim=1)
            entity_cls2 = F.normalize(entity_cls2, p=2, dim=1)
            cross_attended2 = F.normalize(cross_attended2, p=2, dim=1)
            
            entity_cls2 = self.layer_norm1(entity_cls2)
            cross_attended2 = self.layer_norm2(cross_attended2 + entity_cls2)
            
            comprehensive_repr2 = torch.cat([
                text_cls2, entity_cls2, cross_attended2, turkish_features2
            ], dim=1)
            final_representation2 = self.enhanced_fusion(comprehensive_repr2)
            
            context_type_probs2 = F.softmax(context_type_logits2, dim=-1)
            classifier_input2 = torch.cat([final_representation2, context_type_probs2], dim=1)
            
            sentiment_logits2 =self.sentiment_classifier(classifier_input2)
            relation_logits2 = self.relation_classifier(classifier_input2)
            
            r_drop = R_Drop(alpha=0.3)
            sentiment_r_drop_loss = r_drop(sentiment_logits, sentiment_logits2)
            relation_r_drop_loss = r_drop(relation_logits, relation_logits2)
            r_drop_loss = sentiment_r_drop_loss + relation_r_drop_loss
        
        # Loss calculation
        loss = None
        if sentiment_label is not None and relation_label is not None:
            sentiment_loss_fn = AdaptiveFocalLoss(alpha=0.25, gamma=2.0, difficulty_weight=True)
            sentiment_loss = sentiment_loss_fn(sentiment_logits, sentiment_label)
            
            relation_loss_fn = AdaptiveFocalLoss(alpha=0.25, gamma=2.0, difficulty_weight=True)
            relation_loss = relation_loss_fn(relation_logits, relation_label)
            
            loss = sentiment_loss + relation_loss
            
            if self.use_r_drop and self.training and text_input_ids2 is not None:
                loss = loss + r_drop_loss
            
            if self.training:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        
        return {
            'loss': loss,
            'sentiment_logits': sentiment_logits,
            'relation_logits': relation_logits,
            'attention_weights': attention_weights,
            'context_type_logits': context_type_logits
        } if loss is not None else {
            'sentiment_logits': sentiment_logits,
            'relation_logits': relation_logits,
            'attention_weights': attention_weights,
            'context_type_logits': context_type_logits
        }
    
    def save_pretrained(self, path):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        self.config.save_pretrained(path)
