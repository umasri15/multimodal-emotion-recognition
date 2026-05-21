import torch
import os
from transformers import BertTokenizer

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel
from utils.config import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

speech_model = SpeechEmotionModel().to(device)
speech_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "Results", "speech", "speech_model.pth"),
    map_location=device
))
speech_model.eval()

text_model = TextEmotionModel().to(device)
text_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "Results", "text", "text_best_model.pth"),
    map_location=device
))
text_model.eval()

fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "Results", "fusion", "fusion_model.pth"),
    map_location=device
))
fusion_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- INPUT ----------------
speech_input = torch.randn(1, 168).to(device)

text = "I feel amazing and excited"

enc = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

with torch.no_grad():

    speech_out = speech_model(speech_input)
    text_out = text_model(input_ids, attention_mask)

    output = fusion_model(speech_out, text_out)

    pred = torch.argmax(output, dim=1)

print("Fusion Prediction:", pred.item())

