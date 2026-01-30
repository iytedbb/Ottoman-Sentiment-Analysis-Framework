import os
import json
import pkg_resources

def load_cisa_testset():
    """
    Loads the CISA test dataset (Ibrahim Temo's Memoir exstracts).
    
    Returns:
        list: List of dictionaries containing the test examples.
              Each example has 'text', 'entities' (list), and metadata.
    """
    # Try to find the file within the package
    try:
        file_path = pkg_resources.resource_filename(
            "ottoman_sentiment_analysis", 
            "datasets/cisa_testset.json"
        )
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
        
    # Fallback to local path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cisa_testset.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    raise FileNotFoundError("Could not find cisa_testset.json in package resources or local directory.")

def get_cisa_testset_path():
    """Returns the absolute path to the cisa_testset.json file."""
    try:
        return pkg_resources.resource_filename(
            "ottoman_sentiment_analysis", 
            "datasets/cisa_testset.json"
        )
    except:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "cisa_testset.json")
