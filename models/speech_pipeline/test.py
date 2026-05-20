import torch
import os

from models.speech_pipeline.model import SpeechEmotionModel
from utils.config import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpeechEmotionModel().to(device)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "Results", "speech", "speech_model.pth"),
    map_location=device
))
model.eval()

# ✅ FIX: force correct shape ALWAYS
speech_input = torch.randn(1, 168).float().to(device)

with torch.no_grad():
    output = model(speech_input)
    pred = torch.argmax(output, dim=1)

print("Speech Prediction:", pred.item())