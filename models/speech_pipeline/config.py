import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "TESS")
RESULTS_PATH = os.path.join(BASE_DIR, "Results", "speech")

SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001