import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import BASE_DIR, BATCH_SIZE

from models.speech_pipeline.dataset import TESSDataset
from models.text_pipeline.dataset import TextEmotionDataset

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = os.path.join(BASE_DIR, "dataset", "TESS")

    speech_dataset = TESSDataset(dataset_path)
    text_dataset = TextEmotionDataset(dataset_path)

    speech_loader = DataLoader(speech_dataset, batch_size=BATCH_SIZE, shuffle=True)
    text_loader = DataLoader(text_dataset, batch_size=BATCH_SIZE, shuffle=True)

    speech_model = SpeechEmotionModel().to(device)
    speech_model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "Results", "speech", "speech_model.pth"), map_location=device)
    )
    speech_model.eval()

    text_model = TextEmotionModel().to(device)
    text_model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "Results", "text", "text_best_model.pth"), map_location=device)
    )
    text_model.eval()

    fusion_model = FusionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4)

    EPOCHS = 5
    print("Training Fusion Model...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for (speech_x, labels), text_batch in zip(speech_loader, text_loader):
            speech_x = speech_x.to(device)
            labels = labels.to(device)

            input_ids = text_batch["input_ids"].to(device)
            attention_mask = text_batch["attention_mask"].to(device)

            with torch.no_grad():
                speech_out = speech_model(speech_x)
                text_out = text_model(input_ids, attention_mask)

            outputs = fusion_model(speech_out, text_out)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} Acc {acc:.2f}%")

    save_path = os.path.join(BASE_DIR, "Results", "fusion")
    os.makedirs(save_path, exist_ok=True)

    torch.save(
        fusion_model.state_dict(),
        os.path.join(save_path, "fusion_model.pth")
    )

    print("Fusion model saved!")


if __name__ == "__main__":
    main()

