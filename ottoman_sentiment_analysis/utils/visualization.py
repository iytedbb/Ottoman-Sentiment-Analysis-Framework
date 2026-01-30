"""
Visualization Utilities
========================

Plotting functions for model evaluation and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Dict, Any


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels (list): Label names
        save_path (str, optional): Path to save figure
        figsize (tuple): Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_scores, n_classes, class_names, save_path=None, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_scores: Prediction scores
        n_classes (int): Number of classes
        class_names (list): Class names
        save_path (str, optional): Path to save figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            label=f'{class_names[i]} (AUC = {roc_auc:.2f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path=None, figsize=(12, 5)):
    """
    Plot training history (loss and metrics).
    
    Args:
        history (dict): Training history with 'train_loss', 'val_loss', etc.
        save_path (str, optional): Path to save figure
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Metrics plot
    metric_keys = [k for k in history.keys() if 'loss' not in k.lower()]
    for key in metric_keys:
        axes[1].plot(history[key], label=key.replace('_', ' ').title(), linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Training Metrics', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_entity_distribution(entity_counts: Dict[str, int], save_path=None, figsize=(10, 6)):
    """
    Plot entity type distribution.
    
    Args:
        entity_counts (dict): Mapping from entity type to count
        save_path (str, optional): Path to save figure
        figsize (tuple): Figure size
    """
    labels = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    plt.xlabel('Entity Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Entity Type Distribution', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
