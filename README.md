# Ottoman Sentiment Analysis Framework 
by The OSPA Project

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![TÜBİTAK](https://img.shields.io/badge/TÜBİTAK-323K372-red)

A comprehensive NLP framework for analyzing Late Ottoman Turkish memoirs (1900-1950) using Named Entity Recognition (NER), Classical Sentiment Analysis, and Cross-Individual Sentiment Analysis (CISA).

**Supported by TÜBİTAK Project No: 323K372**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Models](#models)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Testing

For detailed testing instructions, see [test_instructions.md](test_instructions.md).

**Quick Test:**
```bash
python examples/test.py
```

## Overview

This framework provides state-of-the-art NLP models specifically trained for historical Turkish texts from the Late Ottoman period (1900-1950). It includes three main components:

1. **Named Entity Recognition (NER)**: Extracts persons, locations, and organizations
2. **Classical Sentiment Analysis**: Analyzes overall sentiment of text
3. **Cross-Individual Sentiment Analysis (CISA)**: Analyzes author's sentiment toward specific individuals

### What is CISA?

CISA (Cross-Individual Sentiment Analysis), is a novel task that analyzes the author's sentiment toward specific individuals mentioned in text, rather than the overall sentiment of the text.

**Example:**

```
Text: "Ali Bey'in vefatı bizleri hüzne boğmuştu"
Translation: "Ali Bey's death deeply saddened us"

Classical Sentiment Analysis → NEGATIVE (sad text)
CISA for "Ali Bey" → POSITIVE (author's respect and affection for Ali Bey)
```

This distinction is crucial for analyzing historical memoirs where authors often express positive sentiments about individuals even in tragic contexts.

## Features

- ✅ **Modular Architecture**: Each model can be used independently
- ✅ **Pre-trained Models**: Ready-to-use models on HuggingFace
- ✅ **Advanced Techniques**: Focal Loss, R-Drop, Layer Ensemble, Dual Encoders
- ✅ **Turkish-Specific Features**: Ottoman Turkish normalization, linguistic features
- ✅ **Comprehensive Documentation**: Detailed guides and examples
- ✅ **Easy Installation**: Single-command setup
- ✅ **Academic Quality**: Published models with citation support

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework.git
cd Ottoman-Sentiment-Analysis-Framework

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Models

All models are available on HuggingFace and published in 2025:

| Model | Task | Performance | DOI | HuggingFace Link |
|-------|------|-------------|-----|------------------|
| **MemoirNER-BERTurk** | NER | F1: 95.30% (PERSON)<br>F1: 76.10% (LOC)<br>F1: 76.28% (ORG) | [10.57967/hf/6141](https://doi.org/10.57967/hf/6141) | [dbbiyte/MemoirNER-BERTurk](https://huggingface.co/dbbiyte/MemoirNER-BERTurk) |
| **HistTurk-BERTurk-Sentiment** | Classical Sentiment | Accuracy: 92.63%<br>F1: 92.62% | [10.57967/hf/6140](https://doi.org/10.57967/hf/6140) | [dbbiyte/histurk-BERTurk-sentiment](https://huggingface.co/dbbiyte/histurk-BERTurk-sentiment) |
| **CISA-BERTurk-Sentiment** | CISA | Accuracy: 87.08%<br>F1: 87.05% | [10.57967/hf/6142](https://doi.org/10.57967/hf/6142) | [dbbiyte/CISA-BERTurk-sentiment](https://huggingface.co/dbbiyte/CISA-BERTurk-sentiment) |

### Datasets

| Dataset | Purpose | Samples | HuggingFace Link |
|---------|---------|---------|-------------------|
| **CISA-testset** | CISA Evaluation | 202 sentences from İbrahim Temo's memoir | [dbbiyte/CISA-testset](https://huggingface.co/datasets/dbbiyte/CISA-testset) |

## Quick Start

### NER (Named Entity Recognition)

```python
from ottoman_sentiment_analysis.models.ner import NERPredictor

# Load model
ner = NERPredictor("dbbiyte/MemoirNER-BERTurk")

# Predict entities
text = "Mustafa Kemal Paşa İstanbul'a geldi."
entities = ner.predict(text)

print(entities)
# Output: [
#     {'text': 'Mustafa Kemal Paşa', 'label': 'PERSON', 'start': 0, 'end': 18},
#     {'text': 'İstanbul', 'label': 'LOC', 'start': 19, 'end': 27}
# ]
```

### Classical Sentiment Analysis

```python
from ottoman_sentiment_analysis.models.sentiment import SentimentPredictor

# Load model
sentiment = SentimentPredictor("dbbiyte/histurk-BERTurk-sentiment")

# Predict sentiment
text = "Bu kitap çok güzeldi, çok beğendim."
result = sentiment.predict(text)

print(result)
# Output: {'sentiment': 'positive', 'label': 2, 'confidence': 0.95}
```

### CISA (Cross-Individual Sentiment Analysis)

```python
from ottoman_sentiment_analysis.models.cisa import CISAPredictor

# Load model
cisa = CISAPredictor("dbbiyte/CISA-BERTurk-sentiment")

# Analyze sentiment toward entity
text = "Ali Bey'in vefatı bizleri hüzne boğmuştu. O büyük bir devlet adamıydı."
entity = "Ali Bey"

result = cisa.predict(text, entity)

print(result)
# Output: {'sentiment': 'positive', 'label': 2, 'confidence': 0.89}
# Note: Despite sad context, sentiment toward Ali Bey is positive (respect)
```

## Training Models

### Train NER Model

```python
from ottoman_sentiment_analysis.models.ner import train_ner_model

trainer, tokenizer = train_ner_model(
    json_file_path="path/to/ner_data.json",
    output_dir="./ner_model"
)
```

### Train Sentiment Model

```python
from ottoman_sentiment_analysis.models.sentiment import train_sentiment_model

trainer, tokenizer = train_sentiment_model(
    json_file_path="path/to/sentiment_data.json",
    output_dir="./sentiment_model"
)
```

### Train CISA Model

```python
from ottoman_sentiment_analysis.models.cisa import train_cisa_model

trainer, tokenizer = train_cisa_model(
    json_file_path="path/to/cisa_data.json",
    output_dir="./cisa_model"
)
```

## Performance

### NER Performance

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| PERSON | 95.83% | 94.78% | **95.30%** | 1,234 |
| LOC | 78.46% | 73.91% | **76.10%** | 567 |
| ORG | 80.00% | 72.73% | **76.28%** | 234 |

### Classical Sentiment Performance

- **Accuracy**: 92.63%
- **Weighted F1**: 92.62%
- **Macro F1**: 91.85%

### CISA Performance

- **Accuracy**: 87.08%
- **Weighted F1**: 87.05%
- **Macro F1**: 86.92%

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{Ottoman-Sentiment-Analysis-Framework-2026,
  title={Ottoman Sentiment Analysis Framework: Tools for Analyzing Late Ottoman Memoirs},
  author={İlter, Mustafa and Onuç, Emre and Evecen, Doğan and Erşahin, Buket and Özcan Gönülal, Yasemin and Karabulut, Sezen and Berci, İbrahim and Tekir, Selma},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework}},
  note={Supported by TÜBİTAK Project No: 323K372}
}
```

### Individual Model Citations

**MemoirNER-BERTurk (NER Model):**
```bibtex
@misc{ilter2025memoirner,
  author = {İlter, Mustafa and Onuç, Emre and Evecen, Doğan and Erşahin, Buket and Özcan Gönülal, Yasemin and Karabulut, Sezen and Berci, İbrahim and Tekir, Selma},
  title = {MemoirNER-BERTurk: Named Entity Recognition for Ottoman Turkish Memoirs},
  howpublished = {Deep Learning Model},
  doi = {10.57967/hf/6141},
  publisher = {Hugging Face},
  url = {https://huggingface.co/dbbiyte/MemoirNER-BERTurk},
  year = {2025},
}
```

**HistTurk-BERTurk-Sentiment (Classical Sentiment):**
```bibtex
@misc{ilter2025histturk,
  author = {İlter, Mustafa and Özcan Gönülal, Yasemin},
  title = {HistTurk-BERTurk-Sentiment: Tarihi Türkçe Duygu Analizi Modeli (1900-1950)},
  howpublished = {Deep Learning Model},
  publisher = {Hugging Face},
  url = {https://huggingface.co/dbbiyte/histurk-BERTurk-sentiment},
  doi = {10.57967/hf/6140},
  year = {2025},
  institution = {İzmir Yüksek Teknoloji Enstitüsü}
}
```

**CISA-BERTurk-Sentiment (CISA/CISA Model):**
```bibtex
@misc{ilter2025cisa,
  author = {İlter, Mustafa and Evecen, Doğan and Erşahin, Buket and Özcan Gönülal, Yasemin and Karabulut, Sezen and Berci, İbrahim and Onuç, Emre and Tekir, Selma},
  title = {CISA-BERTurk-Sentiment: Cross-Individual Sentiment Analysis for Historical Turkish},
  howpublished = {Deep Learning Model},
  publisher = {Hugging Face},
  url = {https://huggingface.co/dbbiyte/CISA-BERTurk-sentiment},
  doi = {10.57967/hf/6142},
  year = {2025},
}
```

**CISA-testset Dataset:**
```bibtex
@dataset{berci2025cisa_testset,
  authors = {İbrahim Berci and Sezen Karabulut and Mustafa İlter},
  title = {CISA-testset from İbrahim Temo's Memoir},
  url = {https://huggingface.co/datasets/dbbiyte/CISA-testset},
  year = {2025},
}
```

For complete citation details, see [CITATION.bib](CITATION.bib).

## License

This project is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

- ✅ You can use, modify, and share this work
- ✅ You must give appropriate credit
- ❌ You cannot use it for commercial purposes

For commercial use, please contact the authors.

## Acknowledgments

This work is supported by:

- **TÜBİTAK (The Scientific and Technological Research Council of Turkey)**  
  Project No: 323K372

### Contributors

- **Dr. Mustafa İLTER** - İzmir Institute of Technology (İYTE)  
  Digital Humanities and AI Lab

**Research Team:**
- Dr. Doğan EVECEN - İzmir Institute of Technology
- Dr. Buket ERŞAHİN - İzmir Institute of Technology  
- Dr. Yasemin ÖZCAN GÖNÜLAL - İzmir Institute of Technology
- Assoc. Prof. Selma TEKİR - İzmir Institute of Technology
- Assoc. Prof. Sezen KARABULUT - Pamukkale University
- İbrahim BERCİ - Pamukkale University
- Emre ONUÇ - Pamukkale University

## Project Structure

```
Ottoman-Sentiment-Analysis-Framework/
├── ottoman_sentiment_analysis/
│   ├── models/
│   │   ├── ner/          # Named Entity Recognition
│   │   ├── sentiment/    # Classical Sentiment Analysis
│   │   └── cisa/         # CISA Models
│   ├── utils/            # Utilities and helpers
│   ├── datasets/         # Dataset handlers
│   └── __init__.py
├── examples/
│   └── evaluate_cisa_on_temo.py  # Complete pipeline test
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
├── README.md             # This file
├── TEST_INSTRUCTIONS.md  # Testing guide
└── CITATION.bib          # Citation information
```

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework/issues](https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework/issues)
- **Email**: mustafailter@iyte.edu.tr

