"""
Script to Evaluate CISA Model on Ibrahim Temo's Memoir Dataset
==============================================================

This script demonstrates how to load the CISA model and evaluate it on the test dataset.

TWO EVALUATION MODES:
1. Direct Mode (--no-ner): Uses ground truth entity positions from test set
2. Pipeline Mode (default): Uses NER model to find entities first, then CISA for sentiment
"""

import os
import json
import logging
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ottoman_sentiment_analysis.models.cisa import CISAPredictor, CISA_CONFIG
from ottoman_sentiment_analysis.datasets import load_cisa_testset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== NER MODEL ====================
class NEREntityExtractor:
    """Extract PERSON entities using NER model"""
    
    def __init__(self, ner_model_path="dbbiyte/MemoirNER-BERTurk"):
        logger.info(f"Loading NER model from {ner_model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {0: "O", 1: "PERSON", 2: "LOC", 3: "ORG"}
        logger.info("NER model loaded successfully")
    
    def extract_entities(self, text, max_length=512):
        """Extract PERSON entities from text"""
        inputs = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True,
            padding=True, truncation=True, max_length=max_length
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        predictions = predictions[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())
        offset_mapping = offset_mapping[0].cpu().tolist()
        
        # Extract PERSON entities
        person_entities = []
        current_entity = {"text": "", "start": 0, "end": 0}
        in_entity = False
        
        for idx, (token, pred, token_offset) in enumerate(zip(tokens, predictions, offset_mapping)):
            if token in ["[CLS]", "[SEP]", "[PAD]"] or token_offset[0] == token_offset[1]:
                continue
            
            pred_label = self.id2label[pred]
            
            if not in_entity and pred_label == "PERSON":
                current_entity = {
                    "text": token.replace("##", ""),
                    "start": token_offset[0],
                    "end": token_offset[1]
                }
                in_entity = True
            elif in_entity and pred_label == "PERSON":
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token if current_entity["text"] else token
                current_entity["end"] = token_offset[1]
            elif in_entity:
                current_entity["text"] = current_entity["text"].strip()
                if current_entity["text"]:
                    person_entities.append(current_entity.copy())
                in_entity = False
                
                if pred_label == "PERSON":
                    current_entity = {
                        "text": token.replace("##", ""),
                        "start": token_offset[0],
                        "end": token_offset[1]
                    }
                    in_entity = True
        
        if in_entity and current_entity["text"].strip():
            current_entity["text"] = current_entity["text"].strip()
            person_entities.append(current_entity)
        
        return person_entities

def calculate_overlap_ratio(gt_entity, pred_entity):
    """Calculate overlap ratio between ground truth and predicted entity"""
    gt_start, gt_end = gt_entity['start'], gt_entity['end']
    pred_start, pred_end = pred_entity['start'], pred_entity['end']
    
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start
    gt_length = gt_end - gt_start
    
    return overlap_length / gt_length if gt_length > 0 else 0.0

def evaluate_cisa_model(model_path, dataset_path=None, output_dir="results", use_ner=True, ner_model_path="dbbiyte/bert-base-turkish-ner-cased", overlap_threshold=0.5):
    """
    Evaluates the CISA model on the test dataset.
    
    Args:
        model_path: Path to CISA model
        dataset_path: Path to test dataset
        output_dir: Directory to save results
        use_ner: If True, use NER model to find entities (pipeline mode). If False, use ground truth entities (direct mode)
        ner_model_path: Path to NER model (only used if use_ner=True)
        overlap_threshold: Minimum overlap ratio to consider NER entity as matching ground truth
    """
    # 1. Load CISA Model
    logger.info(f"Loading CISA model from {model_path}...")
    predictor = CISAPredictor(model_path)
    
    # 2. Load NER Model (if pipeline mode)
    ner_extractor = None
    if use_ner:
        ner_extractor = NEREntityExtractor(ner_model_path)
    
    # 3. Load Dataset
    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        logger.info("Loading bundled CISA testset (Temo's Memoir)...")
        test_data = load_cisa_testset()
        
    logger.info(f"Loaded {len(test_data)} test examples.")
    
    # 4. Predict
    if use_ner:
        logger.info("Running predictions in PIPELINE MODE (NER → CISA)...")
    else:
        logger.info("Running predictions in DIRECT MODE (Ground Truth → CISA)...")
    
    predictions = []
    true_labels = []
    results = []
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # NER statistics
    total_gt_entities = 0
    found_entities = 0
    missed_entities = []
    
    for item in test_data:
        text = item.get('text', '')
        gt_entities = item.get('entities', [])
        
        total_gt_entities += len(gt_entities)
        
        if use_ner:
            # PIPELINE MODE: Use NER to find entities
            ner_entities = ner_extractor.extract_entities(text)
            
            # Match NER entities with ground truth
            for gt_entity in gt_entities:
                gt_start = gt_entity.get('start', 0)
                gt_end = gt_entity.get('end', 0)
                gt_target = gt_entity.get('target', '')
                true_label = gt_entity.get('sentiment', None)
                
                if true_label is None:
                    continue
                
                # Find best matching NER entity
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
                    # Entity found by NER
                    found_entities += 1
                    
                    # Predict sentiment using CISA
                    try:
                        pred_result = predictor.predict(
                            text, 
                            entity=best_match['text'],
                            entity_start=best_match['start'],
                            entity_end=best_match['end']
                        )
                        
                        predicted_label_id = -1
                        if pred_result['sentiment'] == 'negative': predicted_label_id = 0
                        elif pred_result['sentiment'] == 'neutral': predicted_label_id = 1
                        elif pred_result['sentiment'] == 'positive': predicted_label_id = 2
                        
                        predictions.append(predicted_label_id)
                        true_labels.append(true_label)
                        
                        results.append({
                            'text': text[:50] + "...",
                            'gt_entity': gt_target,
                            'ner_entity': best_match['text'],
                            'overlap': best_overlap,
                            'true_label': label_map.get(true_label),
                            'predicted_label': pred_result['sentiment'],
                            'confidence': pred_result['confidence'],
                            'correct': predicted_label_id == true_label
                        })
                    except Exception as e:
                        logger.warning(f"Prediction failed for entity '{best_match['text']}': {e}")
                else:
                    # Entity missed by NER
                    missed_entities.append({
                        'text': text[:50] + "...",
                        'entity': gt_target,
                        'best_overlap': best_overlap
                    })
        else:
            # DIRECT MODE: Use ground truth entities
            for entity_info in gt_entities:
                entity_text = entity_info.get('target', '')
                true_label = entity_info.get('sentiment', None)
                
                if true_label is None:
                    continue
                
                found_entities += 1
                    
                # Perform prediction
                try:
                    entity_start = entity_info.get('start')
                    entity_end = entity_info.get('end')
                    pred_result = predictor.predict(
                        text, 
                        entity=entity_text,
                        entity_start=entity_start,
                        entity_end=entity_end
                    )
                    
                    predicted_label_id = -1
                    if pred_result['sentiment'] == 'negative': predicted_label_id = 0
                    elif pred_result['sentiment'] == 'neutral': predicted_label_id = 1
                    elif pred_result['sentiment'] == 'positive': predicted_label_id = 2
                    
                    predictions.append(predicted_label_id)
                    true_labels.append(true_label)
                    
                    results.append({
                        'text': text[:50] + "...",
                        'entity': entity_text,
                        'true_label': label_map.get(true_label),
                        'predicted_label': pred_result['sentiment'],
                        'confidence': pred_result['confidence'],
                        'correct': predicted_label_id == true_label
                    })
                except Exception as e:
                    logger.warning(f"Prediction failed for entity '{entity_text}': {e}")

    # 5. Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
    
    logger.info("\\n" + "="*50)
    if use_ner:
        logger.info(f"Evaluation Results - PIPELINE MODE (NER → CISA)")
        logger.info("="*50)
        logger.info(f"NER Performance:")
        logger.info(f"  Total GT Entities: {total_gt_entities}")
        logger.info(f"  Found by NER: {found_entities}")
        logger.info(f"  Missed by NER: {len(missed_entities)}")
        logger.info(f"  NER Recall: {found_entities/total_gt_entities:.4f}")
        logger.info("-" * 50)
    else:
        logger.info(f"Evaluation Results - DIRECT MODE (Ground Truth → CISA)")
        logger.info("="*50)
    
    logger.info(f"CISA Sentiment Performance (on {'found' if use_ner else 'all'} entities):")
    logger.info(f"  Total Predictions: {len(predictions)}")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    if use_ner:
        end_to_end_success = sum(1 for r in results if r['correct']) / total_gt_entities if total_gt_entities > 0 else 0.0
        logger.info(f"  End-to-End Success: {end_to_end_success:.4f}")
    
    logger.info("-" * 50)
    
    print("\\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"], zero_division=0))
    
    # 6. Save Results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"cisa_evaluation_results_{'pipeline' if use_ner else 'direct'}.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")
    
    if use_ner and missed_entities:
        df_missed = pd.DataFrame(missed_entities)
        missed_path = os.path.join(output_dir, "missed_entities_by_ner.csv")
        df_missed.to_csv(missed_path, index=False)
        logger.info(f"Missed entities saved to {missed_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    mode_str = 'Pipeline' if use_ner else 'Direct'
    plt.title(f'CISA Confusion Matrix ({mode_str} Mode)')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{'pipeline' if use_ner else 'direct'}.png"))
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix_{'pipeline' if use_ner else 'direct'}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CISA model on testset")
    parser.add_argument("--model_path", type=str, default="dbbiyte/CISA-BERTurk-sentiment", help="Path to trained model or HuggingFace repo")
    parser.add_argument("--data_path", type=str, default=None, help="Path to custom json dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--no-ner", action="store_true", help="Skip NER and use ground truth entities (Direct Mode)")
    parser.add_argument("--ner_model", type=str, default="dbbiyte/MemoirNER-BERTurk", help="Path to NER model (only for Pipeline Mode)")
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Minimum overlap ratio for NER entity matching")
    
    args = parser.parse_args()
    
    evaluate_cisa_model(
        args.model_path, 
        args.data_path, 
        args.output_dir,
        use_ner=not args.no_ner,
        ner_model_path=args.ner_model,
        overlap_threshold=args.overlap_threshold
    )

