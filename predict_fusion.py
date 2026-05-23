import torch
import numpy as np
import librosa
import sys
from transformers import BertTokenizer

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel
from utils.config import SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Load models
speech_model = SpeechEmotionModel().to(device)
speech_model.load_state_dict(torch.load("Results/speech/speech_model.pth", map_location=device))
speech_model.eval()

text_model = TextEmotionModel().to(device)
text_model.load_state_dict(torch.load("Results/text/text_best_model.pth", map_location=device))
text_model.eval()

fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(torch.load("Results/fusion/fusion_model.pth", map_location=device))
fusion_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------
# Speech feature extraction (same as speech script)
# -----------------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)

    mfcc = mfcc.T[:100]
    mel = mel.T[:100]

    if len(mfcc) < 100:
        mfcc = np.pad(mfcc, ((0, 100 - len(mfcc)), (0, 0)))

    if len(mel) < 100:
        mel = np.pad(mel, ((0, 100 - len(mel)), (0, 0)))

    features = np.hstack((mfcc, mel)).flatten()
    features = features[:168]

    if len(features) < 168:
        features = np.pad(features, (0, 168 - len(features)))

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------
# Inputs
# -----------------------
if len(sys.argv) < 3:
    print("Usage: python predict_fusion.py <audio_path> <text>")
    sys.exit()

audio_path = sys.argv[1]
text_input = sys.argv[2]

# -----------------------
# Prediction
# -----------------------
with torch.no_grad():

    speech_x = extract_features(audio_path)

    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    speech_out = speech_model(speech_x)
    text_out = text_model(
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device)
    )

    out = fusion_model(speech_out, text_out)
    pred = torch.argmax(out, dim=1)

print("Fusion Emotion Prediction:", emotion_labels[pred.item()])