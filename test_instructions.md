# GitHub Repository Testing Guide
## Ottoman Sentiment Analysis Framework

This document explains how to test whether the GitHub repository is working correctly.

---

## 🚀 Quick Test (Recommended)

### 1. Clone the Repository

```bash
# Test in a new folder (to avoid mixing with local changes)
cd ~/Desktop
git clone https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework.git
cd Ottoman-Sentiment-Analysis-Framework
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install package
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

### 3. Run Test Script

```bash
# Test all models
python examples/test_github_repository.py --all

# Test only CISA model
python examples/test_github_repository.py --test-cisa

# Test only NER model
python examples/test_github_repository.py --test-ner

# Test only Sentiment model
python examples/test_github_repository.py --test-sentiment
```

---

## 📊 Expected Output

### ✅ Successful Test Output:

```
==============================================================
GitHub Repository Test Suite
Ottoman Sentiment Analysis Framework
==============================================================

==============================================================
Checking repository structure...
==============================================================
✅ Found: ottoman_sentiment_analysis
✅ Found: ottoman_sentiment_analysis/models
✅ Found: ottoman_sentiment_analysis/models/cisa
✅ Found: ottoman_sentiment_analysis/models/ner
✅ Found: ottoman_sentiment_analysis/models/sentiment
✅ Found: ottoman_sentiment_analysis/datasets

✅ Repository structure verified!

==============================================================
Testing CISA Model
==============================================================
Loading CISA model from HuggingFace: dbbiyte/CISA-BERTurk-sentiment...
✅ Model loaded successfully!
Loading CISA test dataset...
✅ Loaded 150 test examples
Running predictions...

==============================================================
CISA Model Results
==============================================================
Accuracy:  0.8734
Precision: 0.8691
Recall:    0.8734
F1 Score:  0.8698
==============================================================

✅ CISA PASSED! F1=0.8698 (Expected: 0.8600)

==============================================================
TEST SUMMARY
==============================================================
CISA           : ✅ PASSED
NER            : ✅ PASSED
Sentiment      : ✅ PASSED
==============================================================

🎉 ALL TESTS PASSED! Repository is working correctly.
```

---

## 🔍 Manual Testing Options

### A. Manual CISA Model Test

```python
from ottoman_sentiment_analysis.models.cisa import CISAPredictor

# Load model (from HuggingFace)
predictor = CISAPredictor("dbbiyte/CISA-BERTurk-sentiment")

# Make prediction
text = "İbrahim Temo İstanbul'da çok önemli işler yaptı."
entity = "İbrahim Temo"

result = predictor.predict(text, entity=entity)
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### B. Manual NER Model Test

```python
from ottoman_sentiment_analysis.models.ner import NERPredictor

# Load model (from HuggingFace)
predictor = NERPredictor("dbbiyte/NER-BERTurk")

# Make prediction
text = "Mahmut Şevket Paşa İstanbul'da Harbiye Nezareti'nde görev yaptı."
entities = predictor.predict(text)

for ent in entities:
    print(f"{ent['text']:20s} -> {ent['type']:10s} (conf: {ent['score']:.4f})")
```

### C. Manual Sentiment Model Test

```python
from ottoman_sentiment_analysis.models.sentiment import SentimentPredictor

# Load model (from HuggingFace)
predictor = SentimentPredictor("dbbiyte/Sentiment-BERTurk")

# Make prediction
text = "Bu çok güzel bir eserdir."
result = predictor.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

---

## 📁 Test Dataset Locations

The test script uses the following datasets:

- **CISA:** `ottoman_sentiment_analysis/datasets/cisa_testset.json`
- **NER:** `ottoman_sentiment_analysis/datasets/ner_testset.json`
- **Sentiment:** (Sample data in script)

### Check Dataset Format:

```bash
# Check CISA test dataset
cat ottoman_sentiment_analysis/datasets/cisa_testset.json | head -20

# Check NER test dataset  
cat ottoman_sentiment_analysis/datasets/ner_testset.json | head -20
```

---

## 🎯 Expected Performance Metrics

| Model | Metric | Expected Value | Tolerance |
|-------|--------|----------------|-----------|
| **CISA** | F1 Score | ~0.86 | ±0.02 |
| **NER** | Entity F1 | ~0.92 | ±0.02 |
| **Sentiment** | F1 Score | ~0.93 | ±0.02 |

> **Note:** These values are based on results obtained during training. Results may vary if test dataset is different.

---

## 🐛 Troubleshooting

### ImportError: No module named 'ottoman_sentiment_analysis'

**Solution:**
```bash
# Make sure you're in repository root
pip install -e .
```

### ModuleNotFoundError: transformers, torch, etc.

**Solution:**
```bash
pip install -r requirements.txt
```

### Model file not found (HuggingFace error)

**Solution:**
```bash
# Check your internet connection
# If HuggingFace token needed:
huggingface-cli login
```

### Dataset file not found

**Solution:**
```bash
# Make sure you cloned the complete repository
git pull origin main

# Check dataset folder
ls -la ottoman_sentiment_analysis/datasets/
```

---

## 📋 Checklist: Reproducibility Test

Follow these steps to verify that the GitHub code is 100% reproducible:

- [ ] **1. Clean Environment**
  - [ ] Repository cloned in a new folder
  - [ ] Fresh virtual environment created
  - [ ] Installed with `pip install -e .`

- [ ] **2. CISA Model**
  - [ ] Model loads from HuggingFace
  - [ ] Test dataset loads
  - [ ] Predictions work
  - [ ] F1 score computed
  - [ ] F1 ≥ 0.84 (expected: ~0.86)

- [ ] **3. NER Model**
  - [ ] Model loads from HuggingFace
  - [ ] Test dataset loads
  - [ ] Entity extraction works
  - [ ] Metrics computed
  - [ ] Entity F1 ≥ 0.90 (expected: ~0.92)

- [ ] **4. Sentiment Model**
  - [ ] Model loads from HuggingFace
  - [ ] Predictions work
  - [ ] Metrics computed
  - [ ] F1 ≥ 0.91 (expected: ~0.93)

- [ ] **5. Code Quality**
  - [ ] No import errors
  - [ ] Config files readable
  - [ ] Logger works properly
  - [ ] Exception handling present

---

## 💡 Tips

1. **First test may be slow** - Downloading HuggingFace models takes time (~1-2 GB).

2. **Use cache** - Downloaded models are cached in `~/.cache/huggingface/`.

3. **GPU usage** - CUDA is used automatically if available, otherwise runs on CPU (slower).

4. **Verbose logging** - For more detailed output:
   ```bash
   python examples/test_github_repository.py --all --verbose
   ```

5. **Quick test only** - If full test takes too long, test just CISA:
   ```bash
   python examples/test_github_repository.py --test-cisa
   ```

---

## 📞 Support

If you encounter issues:

1. **Check the log files**
2. **Open a GitHub Issue**: https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework/issues
3. **Share the stack trace**

---

**Test Script Version:** 1.0  
**Last Updated:** January 31, 2026  
**Maintainer:** @iytedbb
