"""
Data Processing Utilities
==========================

Common utilities for text processing and dataset handling.
"""

import re
import json
import hashlib
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize historical Turkish text.
    
    Removes Ottoman diacritics (â, î, û) and standardizes whitespace.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Remove Ottoman diacritics
    text = re.sub(r'â', 'a', text)
    text = re.sub(r'î', 'i', text)
    text = re.sub(r'û', 'u', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.
    
    Supports both list format and dict with 'data' key.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        list: Dataset as list of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported JSON format in {file_path}")


def split_dataset(data: List[Dict], test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    
    Ensures no duplicate samples between sets using content hashing.
    
    Args:
        data (list): Dataset to split
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
    Returns:
        tuple: (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    # Remove duplicates first
    unique_data = {}
    for item in data:
        content_hash = hashlib.md5(
            normalize_text(item.get('text', '')).lower().encode()
        ).hexdigest()
        
        if content_hash not in unique_data:
            unique_data[content_hash] = item
    
    unique_list = list(unique_data.values())
    
    # Split
    train_data, test_data = train_test_split(
        unique_list,
        test_size=test_size,
        random_state=random_state
    )
    
    return train_data, test_data


def calculate_text_statistics(data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate basic statistics for a dataset.
    
    Args:
        data (list): Dataset
        
    Returns:
        dict: Statistics including counts, averages, etc.
    """
    if not data:
        return {}
    
    text_lengths = [len(item.get('text', '').split()) for item in data]
    
    stats = {
        'total_samples': len(data),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'min_text_length': min(text_lengths),
        'max_text_length': max(text_lengths),
    }
    
    # Entity statistics if available
    if 'entities' in data[0]:
        total_entities = sum(len(item.get('entities', [])) for item in data)
        stats['total_entities'] = total_entities
        stats['avg_entities_per_sample'] = total_entities / len(data)
    
    return stats
