# Ottoman Sentiment Analysis Framework - Test Instructions

This document provides step-by-step instructions for testing the CISA (Cross-Individual Sentiment Analysis) model.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dbbiyte/Ottoman-Sentiment-Analysis-Framework.git
cd Ottoman-Sentiment-Analysis-Framework
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Test

```bash
python examples/test.py
```

---

## 📋 Test Modes

The script can run in two different modes:

### Pipeline Mode (Default)
NER model first finds entities, then CISA performs sentiment analysis.

```bash
python examples/test.py
```

---

## ⚙️ Command Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `dbbiyte/CISA-BERTurk-sentiment` | CISA model path or HuggingFace repo |
| `--data_path` | `None` (bundled dataset) | Custom JSON dataset path |
| `--output_dir` | `evaluation_results` | Directory to save results |
| `--no-ner` | `False` | Use ground truth entities |
| `--ner_model` | `dbbiyte/MemoirNER-BERTurk` | NER model path |
| `--overlap_threshold` | `0.5` | NER matching threshold |

---

## 📁 Output Files

After test completion, in `evaluation_results/` folder:

- `cisa_evaluation_results_pipeline.csv` - Detailed prediction results
- `confusion_matrix_pipeline.png` - Confusion matrix visualization
- `missed_entities_by_ner.csv` - Entities missed by NER (pipeline mode only)

---

## 🔧 Testing with Custom Dataset

To use your own JSON dataset:

```bash
python examples/test.py --data_path path/to/your/dataset.json
```

**Sentiment Labels:**
- `0` = Negative
- `1` = Neutral
- `2` = Positive

---

## 🐍 Python Usage

```python
from ottoman_sentiment_analysis.models.cisa import CISAPredictor

# Load model
predictor = CISAPredictor("dbbiyte/CISA-BERTurk-sentiment")

# Make prediction
text = "Ali Bey'in vefatı bizleri elem-i azîme sevk etmişti; kendisi ile senelerce müşterek mesâimiz mevcuttu"
entity = "Ali Bey"

result = predictor.predict(text, entity)
print(result)
# {'sentiment': 'positive', 'label': 2, 'confidence': 0.89, 'relation': 1, ...}
```

---

## 📝 Notes

- On first run, models will be downloaded from HuggingFace (~500MB)
- GPU is automatically used if available (CUDA)
- Test dataset is from Ibrahim Temo's memoir

---

## 🆘 Troubleshooting

### "Model not found" error
```bash
huggingface-cli login
```

### CUDA out of memory
```bash
python examples/test.py --no-ner  # Use CISA only
```

### Import error
```bash
pip install -e .  # Install package in development mode
```
