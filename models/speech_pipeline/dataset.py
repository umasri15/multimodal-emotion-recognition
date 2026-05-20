import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from .config import SAMPLE_RATE
from .preprocess import extract_mfcc, extract_melspectrogram, pad_features


class TESSDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.file_paths = []
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

        self._load()

    def _get_label(self, folder):
        name = folder.lower()

        if "angry" in name:
            return "angry"
        elif "disgust" in name:
            return "disgust"
        elif "fear" in name:
            return "fear"
        elif "happy" in name:
            return "happy"
        elif "neutral" in name:
            return "neutral"
        elif "sad" in name:
            return "sad"
        elif "surprise" in name:
            return "pleasant_surprise"

        return None

    def _load(self):

        print("\n[DATASET] Loading...")

        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            label = self._get_label(folder)

            if label is None:
                continue

            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(folder_path, file))
                    self.labels.append(self.label_map[label])

        print("[DATASET] Total samples:", len(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]
        label = self.labels[idx]

        audio, sr = librosa.load(path, sr=SAMPLE_RATE)

        mfcc = extract_mfcc(audio, sr)
        mel = extract_melspectrogram(audio, sr)

        mfcc = pad_features(mfcc)
        mel = pad_features(mel)

        features = np.concatenate([mfcc, mel], axis=1)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )