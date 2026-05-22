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

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from utils.config import BASE_DIR, BATCH_SIZE

from models.speech_pipeline.dataset import TESSDataset
from models.text_pipeline.dataset import TextEmotionDataset

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plots_dir = os.path.join(BASE_DIR, "Results", "plots")
os.makedirs(plots_dir, exist_ok=True)

dataset_path = os.path.join(BASE_DIR, "dataset", "TESS")

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

all_labels = []

speech_preds = []
text_preds = []
fusion_preds = []

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Surprise",
    "Sad"
]
with torch.no_grad():

    for (speech_x, labels), text_batch in zip(
        speech_loader,
        text_loader
    ):

        speech_x = speech_x.to(device)
        labels = labels.to(device)

        input_ids = text_batch["input_ids"].to(device)
        attention_mask = text_batch["attention_mask"].to(device)


        speech_outputs = speech_model(speech_x)

        # FIX 3D -> 2D
        if len(speech_outputs.shape) > 2:
            speech_outputs = speech_outputs.mean(dim=1)

        speech_outputs = speech_outputs.view(
            speech_outputs.size(0),
            -1
        )

        speech_pred = torch.argmax(
            speech_outputs,
            dim=1
        )


        text_outputs = text_model(
            input_ids,
            attention_mask
        )

        text_pred = torch.argmax(
            text_outputs,
            dim=1
        )

        fusion_outputs = fusion_model(
            speech_outputs,
            text_outputs
        )

        fusion_pred = torch.argmax(
            fusion_outputs,
            dim=1
        )


        all_labels.extend(
            labels.cpu().numpy()
        )

        speech_preds.extend(
            speech_pred.cpu().numpy()
        )

        text_preds.extend(
            text_pred.cpu().numpy()
        )

        fusion_preds.extend(
            fusion_pred.cpu().numpy()
        )

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 6)
)


cm_speech = confusion_matrix(
    all_labels,
    speech_preds
)

sns.heatmap(
    cm_speech,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[0]
)

axes[0].set_title(
    "Speech Model Confusion Matrix"
)

axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")


cm_text = confusion_matrix(
    all_labels,
    text_preds
)

sns.heatmap(
    cm_text,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[1]
)

axes[1].set_title(
    "Text Model Confusion Matrix"
)

axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")


cm_fusion = confusion_matrix(
    all_labels,
    fusion_preds
)

sns.heatmap(
    cm_fusion,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[2]
)

axes[2].set_title(
    "Fusion Model Confusion Matrix"
)

axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")


plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    "confusion_matrices.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"[INFO] Saved: {save_path}")
print("\nALL VISUALS GENERATED SUCCESSFULLY.")