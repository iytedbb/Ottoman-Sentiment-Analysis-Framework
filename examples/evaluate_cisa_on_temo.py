#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    classification_report,
    roc_curve,
    auc
)
from datasets import Dataset as HFDataset
import traceback
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
def setup_logging(output_dir):
    """Logging sistemini yapılandır"""
    log_file = Path(output_dir) / 'pipeline_test.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# CUDA kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ==================== EBSA MODEL CLASSES ====================
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, size_average=True, difficulty_weight=True):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.difficulty_weight = difficulty_weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
        else:
            alpha_t = 1.0
            
        focal_weight = (1 - pt) ** self.gamma
        
        if self.difficulty_weight:
            difficulty_factor = 1 + torch.exp(-pt * 2)
            focal_weight = focal_weight * difficulty_factor
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class TurkishLinguisticFeatures(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.adjective_noun_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.historical_word_projection = nn.Linear(hidden_size, hidden_size)
        
        self.respect_pattern_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 64)
        )
        
        self.formality_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 32)
        )
        
        self.morphological_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 48)
        )
        
        self.linguistic_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 48, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def detect_turkish_patterns(self, text_repr):
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
        enhanced_text, _ = self.adjective_noun_attention(
            query=text_repr.unsqueeze(1),
            key=text_repr.unsqueeze(1),
            value=text_repr.unsqueeze(1)
        )
        enhanced_text = enhanced_text.squeeze(1)
        
        historical_features = self.historical_word_projection(enhanced_text)
        combined_repr = enhanced_text + historical_features
        turkish_features = self.detect_turkish_patterns(combined_repr)
        
        return turkish_features

class EnhancedEntityContextAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.entity_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        self.local_context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def create_position_weight_matrix(self, entity_positions, seq_length, device):
        batch_size = len(entity_positions)
        weight_matrix = torch.ones(batch_size, seq_length, device=device)
        
        for i, (start_pos, end_pos) in enumerate(entity_positions):
            weight_matrix[i, start_pos:end_pos+1] = 3.0
            
            context_start = max(0, start_pos - 3)
            context_end = min(seq_length, end_pos + 4)
            weight_matrix[i, context_start:start_pos] = 2.0
            weight_matrix[i, end_pos+1:context_end] = 2.0
            
            weight_matrix[i, :context_start] = 0.5
            weight_matrix[i, context_end:] = 0.5
            
        return weight_matrix
    
    def forward(self, entity_repr, text_sequence, entity_positions, attention_mask):
        batch_size, seq_len, hidden_size = text_sequence.shape
        device = text_sequence.device
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)
        enhanced_text = text_sequence + position_emb
        enhanced_text = self.layer_norm1(enhanced_text)
        
        position_weights = self.create_position_weight_matrix(entity_positions, seq_len, device)
        
        entity_query = entity_repr.unsqueeze(1)
        global_attended, global_attention_weights = self.entity_context_attention(
            query=entity_query,
            key=enhanced_text,
            value=enhanced_text,
            key_padding_mask=~attention_mask.bool()
        )
        global_attended = global_attended.squeeze(1)
        
        weighted_attention = global_attention_weights.squeeze(1) * position_weights
        weighted_attended = torch.bmm(
            weighted_attention.unsqueeze(1), 
            enhanced_text
        ).squeeze(1)
        
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
        
        hierarchical_input = torch.cat([global_attended, weighted_attended, local_context_repr], dim=-1)
        hierarchical_weights = self.hierarchical_attention(hierarchical_input)
        hierarchical_weights = F.softmax(hierarchical_weights.view(-1, 3), dim=-1)
        
        final_attended = (
            hierarchical_weights[:, 0:1] * global_attended +
            hierarchical_weights[:, 1:2] * weighted_attended +
            hierarchical_weights[:, 2:3] * local_context_repr
        )
        
        final_attended = self.layer_norm2(final_attended)
        
        return final_attended, global_attention_weights

class ContextualSentimentEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=2, 
            bidirectional=True, 
            dropout=0.1, 
            batch_first=True
        )
        
        self.sentiment_pooling = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        self.context_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )
        
    def forward(self, context_sequence, entity_position_mask):
        lstm_out, (hidden, cell) = self.context_lstm(context_sequence)
        
        if entity_position_mask is not None:
            masked_context = lstm_out * entity_position_mask.unsqueeze(-1).float()
            pooled_context = masked_context.mean(dim=1)
        else:
            pooled_context = lstm_out.mean(dim=1)
        
        sentiment_context, _ = self.sentiment_pooling(
            query=pooled_context.unsqueeze(1),
            key=lstm_out,
            value=lstm_out
        )
        
        context_type_logits = self.context_type_classifier(pooled_context)
        
        return sentiment_context.squeeze(1), context_type_logits

class PositionAwareDualEncoderEBSA(nn.Module):
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased', num_sentiment_labels=3, 
                 dropout_rate=0.1, use_r_drop=True, stochastic_depth_rate=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.text_encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.entity_encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.config = self.text_encoder.config
        
        self.dropout_rate = dropout_rate
        self.use_r_drop = use_r_drop
        self.clip_grad_norm = 1.0
        self.stochastic_depth_rate = stochastic_depth_rate
        
        self.bert_hidden_size = self.text_encoder.config.hidden_size
        
        self.enhanced_attention = EnhancedEntityContextAttention(
            hidden_size=self.bert_hidden_size,
            num_heads=12,
            dropout=dropout_rate
        )
        
        self.turkish_linguistic = TurkishLinguisticFeatures(self.bert_hidden_size)
        self.contextual_encoder = ContextualSentimentEncoder(self.bert_hidden_size)
        
        self.position_embedding = nn.Embedding(512, self.bert_hidden_size)
        self.entity_position_proj = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(self.bert_hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.bert_hidden_size)
        self.layer_norm3 = nn.LayerNorm(self.bert_hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        fusion_input_size = self.bert_hidden_size * 3 + 64
        self.enhanced_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, self.bert_hidden_size * 2),
            nn.LayerNorm(self.bert_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert_hidden_size * 2, self.bert_hidden_size),
            nn.LayerNorm(self.bert_hidden_size)
        )
        
        classifier_input_size = self.bert_hidden_size + 2
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, self.bert_hidden_size // 2),
            nn.LayerNorm(self.bert_hidden_size // 2),
            nn.GELU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert_hidden_size // 2, num_sentiment_labels)
        )
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, self.bert_hidden_size // 2),
            nn.LayerNorm(self.bert_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert_hidden_size // 2, 2)
        )
        
        self.label_smoothing = 0.1
        self.update_layer_drop_probs()
        
    def update_layer_drop_probs(self):
        num_layers = len(self.text_encoder.encoder.layer)
        self.layer_drop_probs = [
            self.stochastic_depth_rate * i / num_layers for i in range(num_layers)
        ]
    
    def encode_with_weighted_layers(self, encoder, input_ids, attention_mask):
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states
        last_four_layers = torch.stack(all_hidden_states[-4:])
        
        layer_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=self.device).view(4, 1, 1, 1)
        weighted_layers = last_four_layers * layer_weights
        final_output = weighted_layers.sum(dim=0)
        
        return final_output, sequence_output
    
    def extract_entity_aware_representation(self, text_output, entity_positions, position_mask):
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
        
        batch_size = text_input_ids.shape[0]
        
        text_output, text_sequence = self.encode_with_weighted_layers(
            self.text_encoder, text_input_ids, text_attention_mask
        )
        
        entity_output, entity_sequence = self.encode_with_weighted_layers(
            self.entity_encoder, entity_input_ids, entity_attention_mask
        )
        
        text_cls = text_output[:, 0, :]
        entity_cls = entity_output[:, 0, :]
        
        entity_aware_repr = self.extract_entity_aware_representation(
            text_output, entity_positions, position_mask
        )
        
        cross_attended, attention_weights = self.enhanced_attention(
            entity_cls, text_output, entity_positions, text_attention_mask
        )
        
        turkish_features = self.turkish_linguistic(text_cls, entity_cls)
        
        context_repr, context_type_logits = self.contextual_encoder(
            text_output, 
            torch.stack([torch.tensor(pm, device=self.device) for pm in position_mask])
        )
        
        text_cls = F.normalize(text_cls, p=2, dim=1)
        entity_cls = F.normalize(entity_cls, p=2, dim=1)
        cross_attended = F.normalize(cross_attended, p=2, dim=1)
        
        entity_cls = self.layer_norm1(entity_cls)
        cross_attended = self.layer_norm2(cross_attended + entity_cls)
        
        comprehensive_repr = torch.cat([
            text_cls, entity_cls, cross_attended, turkish_features
        ], dim=1)
        final_representation = self.enhanced_fusion(comprehensive_repr)
        
        context_type_probs = F.softmax(context_type_logits, dim=-1)
        classifier_input = torch.cat([final_representation, context_type_probs], dim=1)
        
        sentiment_logits = self.sentiment_classifier(classifier_input)
        relation_logits = self.relation_classifier(classifier_input)
        
        return {
            'sentiment_logits': sentiment_logits,
            'relation_logits': relation_logits,
            'attention_weights': attention_weights,
            'context_type_logits': context_type_logits
        }

# ==================== LOAD TEST DATA ====================
def load_test_data(json_path):
    """Test verisini yükle"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Toplam text örneği: {len(data)}")
        
        # Text'leri unique yapıp entity sayısını hesapla
        unique_texts = []
        total_entities = 0
        
        for item in data:
            unique_texts.append({
                'text': item['text'],
                'entities': item['entities'],
                'metadata': item.get('metadata', {})
            })
            total_entities += len(item['entities'])
        
        logging.info(f"Toplam entity sayısı: {total_entities}")
        
        return unique_texts
        
    except Exception as e:
        logging.error(f"Veri yükleme hatası: {str(e)}")
        raise

# ==================== NER FUNCTIONS ====================
def load_ner_model(model_path):
    """NER modelini yükle"""
    try:
        logging.info(f"NER Model yükleniyor: {model_path}")
        
        ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
        ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_model.to(device)
        ner_model.eval()
        
        logging.info("NER Model başarıyla yüklendi")
        return ner_model, ner_tokenizer
        
    except Exception as e:
        logging.error(f"NER Model yükleme hatası: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def predict_ner_entities(text, ner_model, ner_tokenizer, max_length=512):
    """NER modeli ile metindeki PERSON varlıkları tahmin et"""
    try:
        # Metni tokenize et
        inputs = ner_tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        
        # GPU'ya taşı
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Tahmin yap
        with torch.no_grad():
            outputs = ner_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
        
        # Tahminleri al
        predictions = predictions[0].cpu().tolist()
        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = ner_tokenizer.convert_ids_to_tokens(input_ids)
        offset_mapping = offset_mapping[0].cpu().tolist()
        
        # ID to label mapping (NER model için)
        id2label = {
            0: "O",
            1: "PERSON", 
            2: "LOC",
            3: "ORG"
        }
        
        # PERSON entity'lerini çıkart
        person_entities = []
        current_entity = {"text": "", "start": 0, "end": 0}
        in_entity = False
        
        for idx, (token, pred, token_offset) in enumerate(zip(tokens, predictions, offset_mapping)):
            # Özel tokenları atla
            if token in ["[CLS]", "[SEP]", "[PAD]"] or token_offset[0] == token_offset[1]:
                continue
            
            pred_label = id2label[pred]
            
            # PERSON entity başladıysa
            if not in_entity and pred_label == "PERSON":
                current_entity = {
                    "text": token.replace("##", ""),
                    "start": token_offset[0],
                    "end": token_offset[1]
                }
                in_entity = True
            
            # PERSON entity devam ediyorsa
            elif in_entity and pred_label == "PERSON":
                # Alt token (##) kontrolü
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    # Space ekle
                    if current_entity["text"] and not current_entity["text"].endswith(" "):
                        current_entity["text"] += " " + token
                    else:
                        current_entity["text"] += token
                
                current_entity["end"] = token_offset[1]
            
            # PERSON entity bitiyorsa
            elif in_entity:
                # Entity'yi sonlandır
                current_entity["text"] = current_entity["text"].strip()
                if current_entity["text"]:
                    person_entities.append({
                        "text": current_entity["text"],
                        "start": current_entity["start"],
                        "end": current_entity["end"]
                    })
                
                in_entity = False
                
                # Eğer yeni PERSON entity başlıyorsa
                if pred_label == "PERSON":
                    current_entity = {
                        "text": token.replace("##", ""),
                        "start": token_offset[0],
                        "end": token_offset[1]
                    }
                    in_entity = True
        
        # Son entity'yi ekle
        if in_entity:
            current_entity["text"] = current_entity["text"].strip()
            if current_entity["text"]:
                person_entities.append({
                    "text": current_entity["text"],
                    "start": current_entity["start"],
                    "end": current_entity["end"]
                })
        
        return person_entities
        
    except Exception as e:
        logging.error(f"NER tahmin hatası: {str(e)}")
        return []

# ==================== EBSA FUNCTIONS ====================
def load_ebsa_model(model_path):
    """EBSA modelini yükle"""
    try:
        logging.info(f"EBSA Model yükleniyor: {model_path}")
        
        ebsa_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        ebsa_model = PositionAwareDualEncoderEBSA(
            model_name='dbmdz/bert-base-turkish-cased',
            num_sentiment_labels=3,
            dropout_rate=0.1,
            use_r_drop=False,
            stochastic_depth_rate=0.1
        )
        
        # Check if local path or HuggingFace
        local_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(local_path):
            state_dict_path = local_path
        else:
            # HuggingFace'den indir
            from huggingface_hub import hf_hub_download
            state_dict_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
        
        state_dict = torch.load(state_dict_path, map_location=device)
        ebsa_model.load_state_dict(state_dict)
        logging.info("EBSA Model state dict başarıyla yüklendi")
        
        ebsa_model.to(device)
        ebsa_model.eval()
        
        logging.info("EBSA Model başarıyla yüklendi")
        return ebsa_model, ebsa_tokenizer
        
    except Exception as e:
        logging.error(f"EBSA Model yükleme hatası: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def predict_ebsa_sentiment(text, entity, ebsa_model, ebsa_tokenizer, max_length=256):
    """EBSA modeli ile varlık için sentiment tahmin et"""
    try:
        # Entity context oluştur
        entity_text = entity['text']
        entity_start = entity['start']
        entity_end = entity['end']
        
        # Context çıkart (entity sonrasında 200 karakter)
        context_start = entity_end
        context_end = min(len(text), entity_end + 1800)
        context = text[context_start:context_end]
        
        # EBSA input format
        entity_input = f"[CLS] {entity_text} [SEP] {context} [SEP]"
        
        # Tokenize et
        text_tokenized = ebsa_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
            return_offsets_mapping=True
        )
        
        entity_tokenized = ebsa_tokenizer(
            entity_input,
            padding='max_length',
            truncation=True,
            max_length=max_length//2,
            return_tensors="pt",
            return_token_type_ids=False
        )
        
        # Entity token pozisyonlarını bul
        offset_mapping = text_tokenized['offset_mapping'][0]
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
        
        # Position mask oluştur
        pos_mask = [0] * max_length
        for idx in range(start_token_idx, min(end_token_idx + 1, max_length)):
            pos_mask[idx] = 1
        position_mask = [pos_mask]
        
        # GPU'ya taşı
        inputs = {
            'text_input_ids': text_tokenized['input_ids'].to(device),
            'text_attention_mask': text_tokenized['attention_mask'].to(device),
            'entity_input_ids': entity_tokenized['input_ids'].to(device),
            'entity_attention_mask': entity_tokenized['attention_mask'].to(device),
            'entity_positions': entity_positions,
            'position_mask': position_mask
        }
        
        # Tahmin yap
        with torch.no_grad():
            outputs = ebsa_model(**inputs)
            sentiment_logits = outputs['sentiment_logits']
            relation_logits = outputs['relation_logits']
            
            sentiment_pred = torch.argmax(sentiment_logits, dim=-1).cpu().item()
            relation_pred = torch.argmax(relation_logits, dim=-1).cpu().item()
            
            # Confidence scores
            sentiment_probs = F.softmax(sentiment_logits, dim=-1).cpu().numpy()[0]
            relation_probs = F.softmax(relation_logits, dim=-1).cpu().numpy()[0]
        
        return {
            'sentiment': sentiment_pred,
            'relation': relation_pred,
            'sentiment_confidence': float(sentiment_probs[sentiment_pred]),
            'relation_confidence': float(relation_probs[relation_pred])
        }
        
    except Exception as e:
        logging.error(f"EBSA tahmin hatası: {str(e)}")
        return {
            'sentiment': 1,  # Default neutral
            'relation': 0,   # Default indirect
            'sentiment_confidence': 0.0,
            'relation_confidence': 0.0
        }

# ==================== PIPELINE EXECUTION ====================
def run_pipeline_on_texts(texts, ner_model, ner_tokenizer, ebsa_model, ebsa_tokenizer):
    """Pipeline'ı text'ler üzerinde çalıştır: NER → EBSA"""
    try:
        logging.info("Pipeline başlatılıyor: NER → EBSA")
        
        pipeline_results = []
        
        for text_idx, text_item in enumerate(texts):
            text = text_item['text']
            
            logging.info(f"Text {text_idx + 1}/{len(texts)} işleniyor...")
            
            # 1. NER ile PERSON entity'lerini bul
            ner_entities = predict_ner_entities(text, ner_model, ner_tokenizer)
            
            # 2. Her bulunan entity için EBSA sentiment tahmini yap
            ebsa_predictions = []
            for entity in ner_entities:
                sentiment_result = predict_ebsa_sentiment(text, entity, ebsa_model, ebsa_tokenizer)
                
                ebsa_predictions.append({
                    'entity_text': entity['text'],
                    'entity_start': entity['start'],
                    'entity_end': entity['end'],
                    'predicted_sentiment': sentiment_result['sentiment'],
                    'predicted_relation': sentiment_result['relation'],
                    'sentiment_confidence': sentiment_result['sentiment_confidence'],
                    'relation_confidence': sentiment_result['relation_confidence']
                })
            
            pipeline_results.append({
                'text': text,
                'original_entities': text_item['entities'],  # Ground truth
                'ner_found_entities': ner_entities,
                'ebsa_predictions': ebsa_predictions,
                'metadata': text_item.get('metadata', {})
            })
        
        logging.info("Pipeline tamamlandı")
        return pipeline_results
        
    except Exception as e:
        logging.error(f"Pipeline hatası: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# ==================== EVALUATION FUNCTIONS ====================
def calculate_overlap_ratio(gt_entity, pred_entity):
    """İki entity arasındaki örtüşme oranını hesapla"""
    gt_start, gt_end = gt_entity['start'], gt_entity['end']
    pred_start, pred_end = pred_entity['start'], pred_entity['end']
    
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    gt_length = gt_end - gt_start
    
    return overlap_length / gt_length if gt_length > 0 else 0.0

def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """Sınıflandırma metriklerini hesapla"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    except Exception as e:
        logging.error(f"Metrik hesaplama hatası: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def evaluate_pipeline_performance(pipeline_results, overlap_threshold=0.5):
    """Pipeline performansını değerlendir"""
    try:
        logging.info("Pipeline performansı değerlendiriliyor...")
        
        # NER Performance Metrics
        total_gt_entities = 0
        found_entities = 0
        
        # EBSA Performance Metrics (sadece bulunan entity'ler için)
        ebsa_sentiment_true = []
        ebsa_sentiment_pred = []
        ebsa_relation_true = []
        ebsa_relation_pred = []
        
        detailed_results = []
        
        for result in pipeline_results:
            text = result['text']
            gt_entities = result['original_entities']
            ner_entities = result['ner_found_entities']
            ebsa_predictions = result['ebsa_predictions']
            
            total_gt_entities += len(gt_entities)
            
            # NER Entity Matching
            matched_entities = []
            missed_entities = []
            
            for gt_entity in gt_entities:
                gt_target = gt_entity['target']
                gt_start = gt_entity['start']
                gt_end = gt_entity['end']
                gt_sentiment = gt_entity['sentiment']
                gt_relation = 1 if gt_entity['author_related'] else 0
                
                # Bu GT entity'nin NER tarafından bulunup bulunmadığını kontrol et
                best_match = None
                best_overlap = 0.0
                
                for ner_entity in ner_entities:
                    overlap = calculate_overlap_ratio(
                        {'start': gt_start, 'end': gt_end},
                        ner_entity
                    )
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = ner_entity
                
                if best_overlap >= overlap_threshold:
                    # Entity bulundu
                    found_entities += 1
                    
                    # Bu entity için EBSA tahminini bul
                    ebsa_prediction = None
                    for pred in ebsa_predictions:
                        if (pred['entity_start'] == best_match['start'] and 
                            pred['entity_end'] == best_match['end']):
                            ebsa_prediction = pred
                            break
                    
                    if ebsa_prediction:
                        # EBSA metrikler için ground truth ve predictions
                        ebsa_sentiment_true.append(gt_sentiment)
                        ebsa_sentiment_pred.append(ebsa_prediction['predicted_sentiment'])
                        ebsa_relation_true.append(gt_relation)
                        ebsa_relation_pred.append(ebsa_prediction['predicted_relation'])
                    
                    matched_entities.append({
                        'gt_entity': gt_entity,
                        'ner_entity': best_match,
                        'ebsa_prediction': ebsa_prediction,
                        'overlap_ratio': best_overlap,
                        'sentiment_correct': ebsa_prediction['predicted_sentiment'] == gt_sentiment if ebsa_prediction else False,
                        'relation_correct': ebsa_prediction['predicted_relation'] == gt_relation if ebsa_prediction else False
                    })
                else:
                    # Entity kaçırıldı
                    missed_entities.append({
                        'gt_entity': gt_entity,
                        'best_ner_match': best_match,
                        'overlap_ratio': best_overlap
                    })
            
            detailed_results.append({
                'text': text,
                'matched_entities': matched_entities,
                'missed_entities': missed_entities,
                'ner_found_count': len(ner_entities),
                'gt_entity_count': len(gt_entities)
            })
        
        # NER Metrics
        ner_recall = found_entities / total_gt_entities if total_gt_entities > 0 else 0.0
        ner_precision = 1.0  # Sadece GT entity'leri arıyoruz, FP yok
        ner_f1 = 2 * ner_precision * ner_recall / (ner_precision + ner_recall) if (ner_precision + ner_recall) > 0 else 0.0
        
        # EBSA Metrics (tam precision/recall/F1 hesaplaması)
        ebsa_sentiment_metrics = calculate_classification_metrics(
            ebsa_sentiment_true, ebsa_sentiment_pred, average='weighted'
        )
        
        ebsa_relation_metrics = calculate_classification_metrics(
            ebsa_relation_true, ebsa_relation_pred, average='weighted'
        )
        
        # Combined Pipeline Metrics
        correct_sentiment_predictions = sum(1 for t, p in zip(ebsa_sentiment_true, ebsa_sentiment_pred) if t == p)
        pipeline_success_rate = correct_sentiment_predictions / total_gt_entities if total_gt_entities > 0 else 0.0
        
        performance_metrics = {
            'ner_performance': {
                'total_gt_entities': total_gt_entities,
                'found_entities': found_entities,
                'missed_entities': total_gt_entities - found_entities,
                'recall': float(ner_recall),
                'precision': float(ner_precision),
                'f1_score': float(ner_f1)
            },
            'ebsa_sentiment_performance': {
                'total_predictions': len(ebsa_sentiment_true),
                'correct_predictions': correct_sentiment_predictions,
                'accuracy': ebsa_sentiment_metrics['accuracy'],
                'precision': ebsa_sentiment_metrics['precision'],
                'recall': ebsa_sentiment_metrics['recall'],
                'f1_score': ebsa_sentiment_metrics['f1_score']
            },
            'ebsa_relation_performance': {
                'total_predictions': len(ebsa_relation_true),
                'correct_predictions': sum(1 for t, p in zip(ebsa_relation_true, ebsa_relation_pred) if t == p),
                'accuracy': ebsa_relation_metrics['accuracy'],
                'precision': ebsa_relation_metrics['precision'],
                'recall': ebsa_relation_metrics['recall'],
                'f1_score': ebsa_relation_metrics['f1_score']
            },
            'pipeline_performance': {
                'end_to_end_success_rate': float(pipeline_success_rate),
                'total_ground_truth': total_gt_entities,
                'successful_predictions': correct_sentiment_predictions
            },
            'detailed_results': detailed_results
        }
        
        return performance_metrics
        
    except Exception as e:
        logging.error(f"Pipeline değerlendirme hatası: {str(e)}")
        logging.error(traceback.format_exc())
        return {}

# ==================== VISUALIZATION ====================
def visualize_pipeline_results(performance_metrics, output_dir):
    """Pipeline sonuçlarını görselleştir"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Pipeline Overview
        plt.figure(figsize=(15, 10))
        
        # NER Performance
        plt.subplot(2, 3, 1)
        ner_metrics = performance_metrics['ner_performance']
        found = ner_metrics['found_entities']
        missed = ner_metrics['missed_entities']
        
        plt.pie([found, missed], labels=['Found', 'Missed'], colors=['green', 'red'], 
                autopct='%1.1f%%', startangle=90)
        plt.title(f'NER Entity Detection\n(Total: {ner_metrics["total_gt_entities"]} entities)')
        
        # EBSA Sentiment Performance
        plt.subplot(2, 3, 2)
        sentiment_metrics = performance_metrics['ebsa_sentiment_performance']
        correct_sent = sentiment_metrics['correct_predictions']
        wrong_sent = sentiment_metrics['total_predictions'] - correct_sent
        
        if sentiment_metrics['total_predictions'] > 0:
            plt.pie([correct_sent, wrong_sent], labels=['Correct', 'Wrong'], 
                    colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
            plt.title(f'EBSA Sentiment Accuracy\n(Total: {sentiment_metrics["total_predictions"]} predictions)')
        else:
            plt.text(0.5, 0.5, 'No EBSA\nPredictions', ha='center', va='center')
            plt.title('EBSA Sentiment Accuracy')
        
        # Pipeline End-to-End
        plt.subplot(2, 3, 3)
        pipeline_metrics = performance_metrics['pipeline_performance']
        success_rate = pipeline_metrics['end_to_end_success_rate']
        
        plt.bar(['Success Rate'], [success_rate], color='steelblue', alpha=0.8)
        plt.ylim(0, 1.0)
        plt.ylabel('Rate')
        plt.title('End-to-End Pipeline Success')
        plt.text(0, success_rate + 0.02, f'{success_rate:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Detailed Metrics Comparison
        plt.subplot(2, 3, 4)
        metrics_names = ['NER\nRecall', 'NER\nF1', 'EBSA Sent\nF1', 'EBSA Rel\nF1']
        metrics_values = [
            ner_metrics['recall'],
            ner_metrics['f1_score'],
            sentiment_metrics['f1_score'],
            performance_metrics['ebsa_relation_performance']['f1_score']
        ]
        
        bars = plt.bar(metrics_names, metrics_values, alpha=0.8, 
                      color=['lightblue', 'blue', 'lightgreen', 'green'])
        plt.ylabel('Score')
        plt.title('F1 Scores Comparison')
        plt.xticks(rotation=0)
        plt.ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Performance Flow
        plt.subplot(2, 3, 5)
        flow_stages = ['GT Entities', 'NER Found', 'EBSA Correct']
        flow_values = [
            ner_metrics['total_gt_entities'],
            ner_metrics['found_entities'],
            sentiment_metrics['correct_predictions']
        ]
        
        plt.plot(flow_stages, flow_values, 'o-', linewidth=3, markersize=8, color='purple')
        plt.ylabel('Count')
        plt.title('Pipeline Flow')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, value in enumerate(flow_values):
            plt.text(i, value + max(flow_values) * 0.02, str(value), 
                    ha='center', va='bottom', fontsize=10)
        
        # Summary Stats
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, "Pipeline Summary:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f"Total GT Entities: {ner_metrics['total_gt_entities']}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"NER Found: {ner_metrics['found_entities']}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"NER F1: {ner_metrics['f1_score']:.3f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f"EBSA Sentiment F1: {sentiment_metrics['f1_score']:.3f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"EBSA Relation F1: {performance_metrics['ebsa_relation_performance']['f1_score']:.3f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f"End-to-End Success: {pipeline_metrics['end_to_end_success_rate']:.3f}", transform=plt.gca().transAxes)
        
        # Color code overall performance
        if pipeline_metrics['end_to_end_success_rate'] >= 0.8:
            plt.text(0.1, 0.1, "🎉 Excellent Pipeline!", color='green', fontweight='bold', transform=plt.gca().transAxes)
        elif pipeline_metrics['end_to_end_success_rate'] >= 0.6:
            plt.text(0.1, 0.1, "👍 Good Pipeline", color='orange', fontweight='bold', transform=plt.gca().transAxes)
        else:
            plt.text(0.1, 0.1, "🔧 Needs Improvement", color='red', fontweight='bold', transform=plt.gca().transAxes)
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pipeline_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Pipeline görselleştirmeleri kaydedildi: {output_dir}")
        
    except Exception as e:
        logging.error(f"Görselleştirme hatası: {str(e)}")
        logging.error(traceback.format_exc())

# ==================== MAIN FUNCTION ====================
def main():
    """Ana pipeline test fonksiyonu"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"pipeline_test_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Logging setup
        logger = setup_logging(output_dir)
        
        logging.info("="*80)
        logging.info("NER → EBSA PIPELINE TEST SYSTEM")
        logging.info("="*80)
        
        # Model yolları (HuggingFace)
        ebsa_model_path = "dbbiyte/CISA-BERTurk-sentiment"
        ner_model_path = "dbbiyte/MemoirNER-BERTurk"
        
        # Test data yolu (repo içindeki bundled dataset)
        script_dir = Path(__file__).parent.parent
        test_data_path = str(script_dir / "ottoman_sentiment_analysis" / "datasets" / "cisa_testset.json")
        
        # Alternatif: Eğer test_temo.json repo kökünde varsa onu kullan
        alt_path = str(script_dir / "test_temo.json")
        if os.path.exists(alt_path):
            test_data_path = alt_path
        
        # ==================== LOAD DATA ====================
        logging.info("\n" + "="*60)
        logging.info("PHASE 1: DATA LOADING")
        logging.info("="*60)
        
        test_texts = load_test_data(test_data_path)
        
        # ==================== LOAD MODELS ====================
        logging.info("\n" + "="*60)
        logging.info("PHASE 2: MODEL LOADING")
        logging.info("="*60)
        
        # NER Model yükle
        logging.info("NER Model yükleniyor...")
        ner_model, ner_tokenizer = load_ner_model(ner_model_path)
        
        # EBSA Model yükle
        logging.info("EBSA Model yükleniyor...")
        ebsa_model, ebsa_tokenizer = load_ebsa_model(ebsa_model_path)
        
        # ==================== RUN PIPELINE ====================
        logging.info("\n" + "="*60)
        logging.info("PHASE 3: PIPELINE EXECUTION")
        logging.info("="*60)
        
        pipeline_results = run_pipeline_on_texts(
            test_texts, ner_model, ner_tokenizer, ebsa_model, ebsa_tokenizer
        )
        
        # ==================== EVALUATE PERFORMANCE ====================
        logging.info("\n" + "="*60)
        logging.info("PHASE 4: PERFORMANCE EVALUATION")
        logging.info("="*60)
        
        performance_metrics = evaluate_pipeline_performance(pipeline_results)
        
        # ==================== PRINT RESULTS ====================
        logging.info("\n" + "="*60)
        logging.info("PIPELINE TEST RESULTS")
        logging.info("="*60)
        
        ner_perf = performance_metrics['ner_performance']
        sentiment_perf = performance_metrics['ebsa_sentiment_performance']
        relation_perf = performance_metrics['ebsa_relation_performance']
        pipeline_perf = performance_metrics['pipeline_performance']
        
        logging.info("\nNER Performance:")
        logging.info(f"Total GT Entities: {ner_perf['total_gt_entities']}")
        logging.info(f"Found by NER: {ner_perf['found_entities']}")
        logging.info(f"Missed by NER: {ner_perf['missed_entities']}")
        logging.info(f"NER Precision: {ner_perf['precision']:.4f}")
        logging.info(f"NER Recall: {ner_perf['recall']:.4f}")
        logging.info(f"NER F1 Score: {ner_perf['f1_score']:.4f}")
        
        logging.info("\nEBSA Sentiment Performance (on found entities):")
        logging.info(f"Total EBSA Predictions: {sentiment_perf['total_predictions']}")
        logging.info(f"Correct Sentiment: {sentiment_perf['correct_predictions']}")
        logging.info(f"Sentiment Accuracy: {sentiment_perf['accuracy']:.4f}")
        logging.info(f"Sentiment Precision: {sentiment_perf['precision']:.4f}")
        logging.info(f"Sentiment Recall: {sentiment_perf['recall']:.4f}")
        logging.info(f"Sentiment F1 Score: {sentiment_perf['f1_score']:.4f}")
        
        logging.info("\nEBSA Relation Performance (on found entities):")
        logging.info(f"Total EBSA Predictions: {relation_perf['total_predictions']}")
        logging.info(f"Correct Relation: {relation_perf['correct_predictions']}")
        logging.info(f"Relation Accuracy: {relation_perf['accuracy']:.4f}")
        logging.info(f"Relation Precision: {relation_perf['precision']:.4f}")
        logging.info(f"Relation Recall: {relation_perf['recall']:.4f}")
        logging.info(f"Relation F1 Score: {relation_perf['f1_score']:.4f}")
        
        logging.info("\nPipeline Overall Performance:")
        logging.info(f"End-to-End Success Rate: {pipeline_perf['end_to_end_success_rate']:.4f}")
        logging.info(f"Successful Predictions: {pipeline_perf['successful_predictions']}/{pipeline_perf['total_ground_truth']}")
        
        # ==================== DETAILED ANALYSIS ====================
        logging.info("\n" + "="*60)
        logging.info("DETAILED ANALYSIS")
        logging.info("="*60)
        
        # Missed entities analysis
        logging.info("\nMissed Entities (İlk 5):")
        missed_count = 0
        for result in performance_metrics['detailed_results']:
            if result['missed_entities']:
                missed_count += 1
                if missed_count <= 5:
                    logging.info(f"\nExample {missed_count}:")
                    logging.info(f"Text: {result['text'][:100]}...")
                    for missed in result['missed_entities']:
                        gt_entity = missed['gt_entity']
                        logging.info(f"  Missed: '{gt_entity['target']}' at ({gt_entity['start']}-{gt_entity['end']})")
                        if missed['best_ner_match']:
                            match = missed['best_ner_match']
                            logging.info(f"  Best NER match: '{match['text']}' (overlap: {missed['overlap_ratio']:.2f})")
        
        # Successful pipeline examples
        logging.info("\nSuccessful Pipeline Examples (İlk 3):")
        success_count = 0
        for result in performance_metrics['detailed_results']:
            if result['matched_entities']:
                for match in result['matched_entities']:
                    if match['sentiment_correct']:
                        success_count += 1
                        if success_count <= 3:
                            gt = match['gt_entity']
                            ner = match['ner_entity']
                            ebsa = match['ebsa_prediction']
                            sentiment_names = ['Negative', 'Neutral', 'Positive']
                            
                            logging.info(f"\nSuccess {success_count}:")
                            logging.info(f"  GT Entity: '{gt['target']}' → {sentiment_names[gt['sentiment']]}")
                            logging.info(f"  NER Found: '{ner['text']}' (overlap: {match['overlap_ratio']:.2f})")
                            logging.info(f"  EBSA Predicted: {sentiment_names[ebsa['predicted_sentiment']]} ✅")
        
        # ==================== SAVE RESULTS ====================
        logging.info("\n" + "="*60)
        logging.info("SAVING RESULTS")
        logging.info("="*60)
        
        # Comprehensive results
        comprehensive_results = {
            'timestamp': timestamp,
            'model_paths': {
                'ebsa_model': ebsa_model_path,
                'ner_model': ner_model_path
            },
            'test_data_path': test_data_path,
            'test_data_size': len(test_texts),
            'performance_metrics': performance_metrics,
            'pipeline_results': pipeline_results,
            'summary': {
                'ner_precision': float(ner_perf['precision']),
                'ner_recall': float(ner_perf['recall']),
                'ner_f1': float(ner_perf['f1_score']),
                'ebsa_sentiment_precision': float(sentiment_perf['precision']),
                'ebsa_sentiment_recall': float(sentiment_perf['recall']),
                'ebsa_sentiment_f1': float(sentiment_perf['f1_score']),
                'ebsa_sentiment_accuracy': float(sentiment_perf['accuracy']),
                'ebsa_relation_precision': float(relation_perf['precision']),
                'ebsa_relation_recall': float(relation_perf['recall']),
                'ebsa_relation_f1': float(relation_perf['f1_score']),
                'ebsa_relation_accuracy': float(relation_perf['accuracy']),
                'end_to_end_success_rate': float(pipeline_perf['end_to_end_success_rate'])
            }
        }
        
        # JSON olarak kaydet
        results_file = f"{output_dir}/pipeline_test_results.json"
        
        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        
        logging.info(f"Sonuçlar kaydedildi: {results_file}")
        
        # Görselleştirmeleri oluştur
        logging.info("Görselleştirmeler oluşturuluyor...")
        visualize_pipeline_results(performance_metrics, output_dir)
        
        # ==================== FINAL SUMMARY ====================
        logging.info("\n" + "="*80)
        logging.info("PIPELINE TEST COMPLETED!")
        logging.info("="*80)
        
        logging.info(f"\nFinal Performance Summary:")
        logging.info(f"NER F1 Score: {ner_perf['f1_score']:.4f}")
        logging.info(f"EBSA Sentiment F1: {sentiment_perf['f1_score']:.4f}")
        logging.info(f"EBSA Relation F1: {relation_perf['f1_score']:.4f}")
        logging.info(f"End-to-End Success Rate: {pipeline_perf['end_to_end_success_rate']:.4f}")
        
        logging.info(f"\nAll results saved to: {output_dir}")
        logging.info(f"- Pipeline results: {results_file}")
        logging.info(f"- Visualizations: {output_dir}/*.png")
        logging.info(f"- Logs: {output_dir}/pipeline_test.log")
        
        logging.info("="*80)
        
        return comprehensive_results
        
    except Exception as e:
        logging.error(f"Pipeline test sırasında hata: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# ==================== UTILITY FUNCTIONS ====================
if __name__ == "__main__":
    try:
        # GPU kontrolü ve bilgi
        if torch.cuda.is_available():
            print(f"GPU bulundu: {torch.cuda.get_device_name(0)}")
            print(f"GPU bellek: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("GPU bulunamadı, CPU kullanılacak")

        # Ana pipeline testini çalıştır
        print("\nStarting NER → EBSA pipeline testing...")
        results = main()
        
        print("\nPipeline test completed successfully!")
        
    except Exception as e:
        print(f"Pipeline test execution error: {str(e)}")
        import traceback
        traceback.print_exc()
