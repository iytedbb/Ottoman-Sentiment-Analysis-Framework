"""
Script to Evaluate CISA Model on Ibrahim Temo's Memoir Dataset
==============================================================

This script demonstrates how to load the CISA model and evaluate it on the test dataset.
"""

import os
import json
import logging
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ottoman_sentiment_analysis.models.cisa import CISAPredictor, CISA_CONFIG
from ottoman_sentiment_analysis.datasets import load_cisa_testset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_cisa_model(model_path, dataset_path=None, output_dir="results"):
    """
    Evaluates the CISA model on the test dataset.
    """
    # 1. Load Model
    logger.info(f"Loading CISA model from {model_path}...")
    predictor = CISAPredictor(model_path)
    
    # 2. Load Dataset
    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        logger.info("Loading bundled CISA testset (Temo's Memoir)...")
        test_data = load_cisa_testset()
        
    logger.info(f"Loaded {len(test_data)} test examples.")
    
    # 3. Predict
    logger.info("Running predictions...")
    
    predictions = []
    true_labels = []
    results = []
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    for item in test_data:
        text = item.get('text', '')
        # Handle different dataset formats if necessary
        entities = item.get('entities', [])
        
        # If dataset structure is flat (per entity), great. If nested, flatten.
        # Assuming test_temo.json structure: list of objects with 'text' and 'entities' list.
        
        for entity_info in entities:
            entity_text = entity_info.get('target', '')  # In test_temo.json, 'target' is the entity name
            true_label = entity_info.get('sentiment', None)  # 'sentiment' is the label ID (0/1/2)
            
            if true_label is None:
                continue
                
            # Perform prediction
            pred_result = predictor.predict(text, entity=entity_text)
            
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

    # 4. Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    logger.info("\n" + "="*50)
    logger.info(f"Evaluation Results on CISA Testset")
    logger.info("="*50)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("-" * 50)
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Negative", "Neutral", "Positive"]))
    
    # 5. Save Results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cisa_evaluation_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('CISA Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CISA model on testset")
    parser.add_argument("--model_path", type=str, default="dbbiyte/CISA-BERTurk-sentiment", help="Path to trained model or HuggingFace repo")
    parser.add_argument("--data_path", type=str, default=None, help="Path to custom json dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    evaluate_cisa_model(args.model_path, args.data_path, args.output_dir)
