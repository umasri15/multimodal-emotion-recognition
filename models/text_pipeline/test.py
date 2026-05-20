import torch
import os
from transformers import BertTokenizer

from models.text_pipeline.model import TextEmotionModel
from utils.config import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = TextEmotionModel().to(device)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "Results", "text", "text_best_model.pth"),
    map_location=device
))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- TEST TEXT ----------------
text = "I am very happy today"

enc = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

# ---------------- PREDICT ----------------
with torch.no_grad():
    output = model(input_ids, attention_mask)
    pred = torch.argmax(output, dim=1)

print("Text Prediction:", pred.item())