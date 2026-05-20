import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

MODEL_NAME = "bert-base-uncased"