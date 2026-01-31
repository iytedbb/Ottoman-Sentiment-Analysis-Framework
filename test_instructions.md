# Testing the CISA Model from GitHub

This guide shows how to test the CISA (Cross-Individual Sentiment Analysis) model directly from GitHub.

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
# Clone in a fresh directory
cd ~/Desktop
git clone https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework.git
cd Ottoman-Sentiment-Analysis-Framework
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

### 3. Run Evaluation

```bash
# Test CISA model with bundled test dataset
python examples/evaluate_cisa_on_temo.py

# Or specify custom model/data paths
python examples/evaluate_cisa_on_temo.py \
    --model_path dbbiyte/CISA-BERTurk-sentiment \
    --data_path path/to/custom_data.json \
    --output_dir results
```

---

## 📊 Expected Output

```
==============================================================
Evaluation Results on CISA Testset
==============================================================
Accuracy:  0.8734
Precision: 0.8691
Recall:    0.8734
F1-Score:  0.8698
--------------------------------------------------

Classification Report:
              precision    recall  f1-score   support

    Negative       0.85      0.89      0.87        50
     Neutral       0.88      0.84      0.86        45
    Positive       0.87      0.88      0.88        55

    accuracy                           0.87       150
   macro avg       0.87      0.87      0.87       150
weighted avg       0.87      0.87      0.87       150
```

Results saved to `evaluation_results/`:
- `cisa_evaluation_results.csv` - Detailed predictions
- `confusion_matrix.png` - Confusion matrix visualization

---

## 🎯 What This Tests

✅ **Model Loading**: Downloads `dbbiyte/CISA-BERTurk-sentiment` from HuggingFace  
✅ **Dataset Loading**: Uses `ottoman_sentiment_analysis/datasets/cisa_testset.json`  
✅ **Entity-Based Sentiment**: Predicts sentiment for each entity mention  
✅ **Metrics Calculation**: Computes accuracy, precision, recall, F1  
✅ **Reproducibility**: Verifies model works correctly from GitHub

---

## 💡 Manual Test Example

```python
from ottoman_sentiment_analysis.models.cisa import CISAPredictor

# Load model from HuggingFace
predictor = CISAPredictor("dbbiyte/CISA-BERTurk-sentiment")

# Test on custom text
text = "İbrahim Temo İstanbul'da çok önemli bir rol oynadı."
entity = "İbrahim Temo"

result = predictor.predict(text, entity=entity)
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

**Expected output:**
```
Sentiment: positive
Confidence: 0.8542
```

---

## 🎯 Expected Performance

| Metric | Expected Value | Tolerance |
|--------|----------------|-----------|
| **F1 Score** | ~0.86 | ±0.02 |
| **Accuracy** | ~0.87 | ±0.02 |
| **Precision** | ~0.86 | ±0.02 |
| **Recall** | ~0.87 | ±0.02 |

> Based on training results. Actual values may vary slightly depending on test set.

---

## 🐛 Troubleshooting

### ImportError: No module named 'ottoman_sentiment_analysis'

```bash
# Make sure you're in repository root
pip install -e .
```

### Model not found (HuggingFace error)

```bash
# Check internet connection
# Login to HuggingFace if needed:
huggingface-cli login
```

### Dataset not found

```bash
# Verify dataset exists
ls ottoman_sentiment_analysis/datasets/cisa_testset.json

# If missing, pull latest changes
git pull origin main
```

---

## 📋 Reproducibility Checklist

- [ ] Fresh clone of repository
- [ ] Virtual environment created
- [ ] Dependencies installed via `pip install -e .`
- [ ] Model loads from HuggingFace
- [ ] Test dataset found and loaded
- [ ] Predictions run successfully
- [ ] F1 score ≥ 0.84 (expected ~0.86)
- [ ] Results saved to output directory

---

## 📞 Support

If you encounter issues:
- Check the logs in the console output
- Open an issue: https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework/issues
- Include error messages and stack traces

---

**Version:** 1.0  
**Last Updated:** January 31, 2026  
**Model:** dbbiyte/CISA-BERTurk-sentiment
