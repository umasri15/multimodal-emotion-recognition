import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextEmotionDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.samples = []
        self.labels = []

        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "pleasant_surprise": 5,
            "sad": 6
        }

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self._load()

        print(f"[TEXT DATASET] Total samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("Text dataset empty — check dataset path.")

    # -------------------------
    def _extract_emotion(self, folder_name):
        name = folder_name.lower()

        for emo in self.label_map:
            if emo in name:
                return emo
        return None

    # -------------------------
    def _load(self):

        print("[TEXT DATASET] Loading from:", self.dataset_path)

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        for folder in os.listdir(self.dataset_path):

            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            emotion = self._extract_emotion(folder)

            if emotion is None:
                continue

            for file in os.listdir(folder_path):

                if not file.endswith(".wav"):
                    continue

                # pseudo text from filename
                text = file.replace(".wav", "").replace("_", " ")

                self.samples.append(text)
                self.labels.append(self.label_map[emotion])

    # -------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------
    def __getitem__(self, idx):

        text = self.samples[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }