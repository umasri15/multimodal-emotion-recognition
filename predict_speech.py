import torch
import numpy as np
import librosa
import sys

from models.speech_pipeline.model import SpeechEmotionModel
from utils.config import SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load trained model
# -----------------------
model = SpeechEmotionModel().to(device)
model.load_state_dict(torch.load(
    "Results/speech/speech_model.pth",
    map_location=device
))
model.eval()


# -----------------------
# Feature extraction (MATCH TRAINING STYLE)
# -----------------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # MFCC + Mel Spectrogram (same idea as training)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)

    # transpose to (time, features)
    mfcc = mfcc.T
    mel = mel.T

    # fix length to 100 frames
    mfcc = mfcc[:100]
    mel = mel[:100]

    if len(mfcc) < 100:
        pad = np.zeros((100 - len(mfcc), mfcc.shape[1]))
        mfcc = np.vstack((mfcc, pad))

    if len(mel) < 100:
        pad = np.zeros((100 - len(mel), mel.shape[1]))
        mel = np.vstack((mel, pad))

    # combine features
    features = np.hstack((mfcc, mel))  # (100, 33)

    # flatten → IMPORTANT (matches your 168-d training setup approx)
    features = features.flatten()

    # safety trim/pad to match model input size
    features = features[:168]
    if len(features) < 168:
        features = np.pad(features, (0, 168 - len(features)))

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)


if len(sys.argv) < 2:
    print("Usage: python predict_speech.py <path_to_wav>")
    sys.exit()

file_path = sys.argv[1]

with torch.no_grad():
    x = extract_features(file_path)
    out = model(x)
    pred = torch.argmax(out, dim=1)
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

print("Speech Emotion Prediction:", emotion_labels[pred.item()])

