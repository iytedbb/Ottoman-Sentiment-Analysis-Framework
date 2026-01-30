"""
Cross-Individual Sentiment Analysis (CISA) Module
==================================================

Cross-Individual Sentiment Analysis for historical Turkish texts.
Analyzes author's sentiment toward specific individuals mentioned in text.

Performance: 87.08% accuracy, F1: 87.05%

Architecture: DECA-CISA (Dual-Encoder Context-Aware Cross-Individual Sentiment Analysis)

Example:
    Text: "Ali Bey'in vefatı bizleri hüzne boğmuştu"
    Standard SA: Negative (sad text)
    CISA: Positive (author's respect for Ali Bey)
"""

from .architecture import (
    PositionAwareDualEncoderCISA,
    TurkishLinguisticFeatures,
    EnhancedEntityContextAttention,
    ContextualSentimentEncoder,
    AdaptiveFocalLoss,
    R_Drop
)
from .config import CISA_CONFIG
from .train import train_cisa_model
from .inference import CISAPredictor

__all__ = [
    "PositionAwareDualEncoderCISA",
    "TurkishLinguisticFeatures",
    "EnhancedEntityContextAttention",
    "ContextualSentimentEncoder",
    "AdaptiveFocalLoss",
    "R_Drop",
    "CISA_CONFIG",
    "train_cisa_model",
    "CISAPredictor"
]
