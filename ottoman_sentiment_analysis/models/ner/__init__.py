"""
Named Entity Recognition Module
================================

NER model for extracting historical entities from Ottoman Turkish memoirs.

Entities:
    - PERSON: Historical figures (F1: 95.30%)
    - LOC: Locations (F1: 76.10%)
    - ORG: Organizations (F1: 76.28%)
"""

from .architecture import NERDataset, FocalLoss, CustomTrainer
from .config import NER_CONFIG, LABEL_MAP
from .train import train_ner_model
from .inference import NERPredictor

__all__ = [
    "NERDataset",
    "FocalLoss", 
    "CustomTrainer",
    "NER_CONFIG",
    "LABEL_MAP",
    "train_ner_model",
    "NERPredictor"
]
