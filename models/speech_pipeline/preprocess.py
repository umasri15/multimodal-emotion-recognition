import librosa
import numpy as np
from .config import SAMPLE_RATE, N_MFCC, N_MELS


def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T


def extract_melspectrogram(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mel).T


def pad_features(feat, max_len=200):
    if len(feat) > max_len:
        return feat[:max_len]

    pad = np.zeros((max_len - len(feat), feat.shape[1]))
    return np.vstack((feat, pad))