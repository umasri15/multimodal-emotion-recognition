import os

# ======================
# ROOT PATH
# ======================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# ======================
# TRAINING SETTINGS
# ======================
BATCH_SIZE = 16
EPOCHS = 5
LR_SPEECH = 1e-3
LR_TEXT = 2e-5
LR_FUSION = 1e-4

NUM_CLASSES = 7

# ======================
# AUDIO SETTINGS
# ======================
SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128
MAX_LEN = 100

# ======================
# MODEL PATHS
# ======================
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

SPEECH_MODEL_PATH = os.path.join(RESULTS_DIR, "speech", "speech_best_model.pth")
TEXT_MODEL_PATH = os.path.join(RESULTS_DIR, "text", "text_best_model.pth")
FUSION_MODEL_PATH = os.path.join(RESULTS_DIR, "fusion", "fusion_model.pth")