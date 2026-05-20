import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import DATASET_PATH, RESULTS_PATH, BATCH_SIZE, EPOCHS, LR
from .dataset import TESSDataset
from .model import SpeechEmotionModel


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TESSDataset(DATASET_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = SpeechEmotionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\nTraining speech model...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total

        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} Acc {acc:.2f}%")


    os.makedirs(RESULTS_PATH, exist_ok=True)

    save_path = os.path.join(RESULTS_PATH, "speech_model.pth")
    torch.save(model.state_dict(), save_path)

    print("\nSaved at:", save_path)


# 👇 THIS IS REQUIRED ON WINDOWS
if __name__ == "__main__":
    main()