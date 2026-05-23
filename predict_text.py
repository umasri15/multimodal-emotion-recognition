import torch
import sys
from transformers import BertTokenizer

from models.text_pipeline.model import TextEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Load tokenizer + model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TextEmotionModel().to(device)
model.load_state_dict(torch.load(
    "Results/text/text_best_model.pth",
    map_location=device
))
model.eval()

# Input text
if len(sys.argv) < 2:
    text = input("Enter text: ")
else:
    text = sys.argv[1]

inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    out = model(
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device)
    )
    pred = torch.argmax(out, dim=1)

print("Text Emotion Prediction:", emotion_labels[pred.item()])