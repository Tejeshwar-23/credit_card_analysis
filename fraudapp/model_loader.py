import joblib
import os

def load_model(file_path):
    """Loads a model from a .pkl file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {file_path}: {e}")

def load_features(file_path):
    """Loads feature names from a .pkl file."""
    if not os.path.exists(file_path):
        return None
    try:
        features = joblib.load(file_path)
        return features
    except:
        return None
