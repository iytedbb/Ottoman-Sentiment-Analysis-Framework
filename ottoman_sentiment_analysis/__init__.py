"""
Ottoman Sentiment Analysis Framework
==================================

A comprehensive framework for analyzing Late Ottoman Turkish memoirs (1900-1950)
using deep learning models for NER, Sentiment Analysis, and Cross-Individual 
Sentiment Analysis (CISA).

Supported by TÜBİTAK Project No: 323K372

Modules:
    - models.ner: Named Entity Recognition
    - models.sentiment: Classical Sentiment Analysis
    - models.cisa: Cross-Individual Sentiment Analysis
    - utils: Utility functions for data processing and visualization
"""

__version__ = "1.0.0"
__author__ = "İYTE Digital Humanities and AI Lab & Pamukkale University"
__license__ = "CC BY-NC 4.0"

from . import models
from . import utils

__all__ = ["models", "utils"]
