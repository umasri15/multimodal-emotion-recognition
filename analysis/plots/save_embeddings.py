import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.config import BASE_DIR, BATCH_SIZE

from models.speech_pipeline.dataset import TESSDataset
from models.text_pipeline.dataset import TextEmotionDataset

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)



save_dir = os.path.join(
    BASE_DIR,
    "Results",
    "embeddings"
)

os.makedirs(
    save_dir,
    exist_ok=True
)


dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "TESS"
)

speech_dataset = TESSDataset(dataset_path)

text_dataset = TextEmotionDataset(dataset_path)

speech_loader = DataLoader(
    speech_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

text_loader = DataLoader(
    text_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


speech_model = SpeechEmotionModel().to(device)

speech_model.load_state_dict(
    torch.load(
        os.path.join(
            BASE_DIR,
            "Results",
            "speech",
            "speech_model.pth"
        ),
        map_location=device
    )
)

speech_model.eval()

text_model = TextEmotionModel().to(device)

text_model.load_state_dict(
    torch.load(
        os.path.join(
            BASE_DIR,
            "Results",
            "text",
            "text_best_model.pth"
        ),
        map_location=device
    )
)

text_model.eval()

fusion_model = FusionModel().to(device)

fusion_model.load_state_dict(
    torch.load(
        os.path.join(
            BASE_DIR,
            "Results",
            "fusion",
            "fusion_model.pth"
        ),
        map_location=device
    )
)

fusion_model.eval()

print("Models loaded successfully ✔")



speech_embeddings = []
text_embeddings = []
fusion_embeddings = []

all_labels = []



with torch.no_grad():

    for (speech_x, labels), text_batch in zip(
        speech_loader,
        text_loader
    ):

        speech_x = speech_x.to(device)

        input_ids = text_batch["input_ids"].to(device)

        attention_mask = text_batch[
            "attention_mask"
        ].to(device)

        labels = labels.to(device)

       

        speech_outputs = speech_model(
            speech_x
        )

        if len(speech_outputs.shape) > 2:
            speech_outputs = speech_outputs.mean(dim=1)

        speech_outputs = speech_outputs.view(
            speech_outputs.size(0),
            -1
        )


        text_outputs = text_model(
            input_ids,
            attention_mask
        )

        
        fusion_outputs = fusion_model(
            speech_outputs,
            text_outputs
        )



        speech_embeddings.append(
            speech_outputs.cpu().numpy()
        )

        text_embeddings.append(
            text_outputs.cpu().numpy()
        )

        fusion_embeddings.append(
            fusion_outputs.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )



speech_embeddings = np.concatenate(
    speech_embeddings,
    axis=0
)

text_embeddings = np.concatenate(
    text_embeddings,
    axis=0
)

fusion_embeddings = np.concatenate(
    fusion_embeddings,
    axis=0
)

all_labels = np.concatenate(
    all_labels,
    axis=0
)


np.save(
    os.path.join(
        save_dir,
        "speech_embeddings.npy"
    ),
    speech_embeddings
)

np.save(
    os.path.join(
        save_dir,
        "text_embeddings.npy"
    ),
    text_embeddings
)

np.save(
    os.path.join(
        save_dir,
        "fusion_embeddings.npy"
    ),
    fusion_embeddings
)

np.save(
    os.path.join(
        save_dir,
        "labels.npy"
    ),
    all_labels
)

# =====================================================
# DONE
# =====================================================

print("\nEmbeddings saved successfully ✔")

print(
    "\nSaved in:\n",
    save_dir
)

print("\nShapes:")

print(
    "Speech:",
    speech_embeddings.shape
)

print(
    "Text:",
    text_embeddings.shape
)

print(
    "Fusion:",
    fusion_embeddings.shape
)