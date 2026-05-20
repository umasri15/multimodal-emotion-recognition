import os
from .config import BASE_DIR

def get_dataset_path():
    return os.path.join(BASE_DIR, "dataset")

def get_results_path():
    path = os.path.join(BASE_DIR, "Results")
    os.makedirs(path, exist_ok=True)
    return path