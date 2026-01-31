"""
Train Sentiment Model
=====================
Train Classical Sentiment Analysis model (HistTurk-Sentiment).

Expected Data Format (JSON):
----------------------------
[
  {
    "text": "Bu mekan çok güzel.",
    "label": 2  // 0: Negative, 1: Neutral, 2: Positive
  },
  {
    "text": "Hiç beğenmedim.",
    "label": 0
  }
]

Usage:
    python scripts/train_sentiment.py --data_path data/sentiment.json --output_dir saved_models/sentiment
"""

import argparse
import sys
import os
import logging

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from package
try:
    from ottoman_sentiment_analysis.models.sentiment import train_sentiment_model
except ImportError as e:
    logging.error(f"Import failed: {e}")
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ottoman_sentiment_analysis'))
    try:
        from ottoman_sentiment_analysis.models.sentiment.train import train_sentiment_model
    except ImportError as e2:
        logging.error(f"Fallback import also failed: {e2}")
        raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--model_name", type=str, default=None, help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="sentiment_model_output", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    logging.info(f"Starting Sentiment training with data: {args.data_path}")
    train_sentiment_model(args.data_path, args.model_name, args.output_dir)
