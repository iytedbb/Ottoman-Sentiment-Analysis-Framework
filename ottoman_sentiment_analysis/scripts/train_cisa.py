"""
Train CISA Model
================
Script to train the Position-Aware Dual-Encoder CISA model.

Expected Data Format (JSON):
----------------------------
[
  {
    "text": "Full sentence text here...",
    "entities": [
      {
        "target": "Entity Name",
        "sentiment": 2,         // 0: Negative, 1: Neutral, 2: Positive
        "author_related": true, // Relation type (true: Direct, false: Indirect)
        "start": 0,
        "end": 10
      }
    ]
  }
]

Usage:
    python scripts/train_cisa.py --data_path data/train.json --output_dir saved_models/cisa
"""

import argparse
import sys
import os
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from package
try:
    from ottoman_sentiment_analysis.models.cisa import train_cisa_model
except ImportError as e:
    logging.error(f"Import failed: {e}")
    # Fallback to local import if package not installed
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ottoman_sentiment_analysis'))
    try:
        from ottoman_sentiment_analysis.models.cisa.train import train_cisa_model
    except ImportError as e2:
        logging.error(f"Fallback import also failed: {e2}")
        raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CISA Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--model_name", type=str, default=None, help="Pretrained model name (default: from config)")
    parser.add_argument("--output_dir", type=str, default="cisa_model_output", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
        
    logging.info(f"Starting CISA training with data: {args.data_path}")
    train_cisa_model(args.data_path, args.model_name, args.output_dir)
