import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.config import BASE_DIR, BATCH_SIZE

from models.speech_pipeline.dataset import TESSDataset
from models.text_pipeline.dataset import TextEmotionDataset

from models.speech_pipeline.model import SpeechEmotionModel
from models.text_pipeline.model import TextEmotionModel
from models.fusion_pipeline.model import FusionModel


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# RESULTS DIRECTORY
# =========================
plots_dir = os.path.join(BASE_DIR, "Results", "plots")
os.makedirs(plots_dir, exist_ok=True)

# =========================
# DATASET + LOADERS
# =========================
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

# =========================
# LOAD MODELS
# =========================
speech_model = SpeechEmotionModel().to(device)

speech_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "Results", "speech", "speech_model.pth"),
        map_location=device
    )
)

speech_model.eval()

text_model = TextEmotionModel().to(device)

text_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "Results", "text", "text_best_model.pth"),
        map_location=device
    )
)

text_model.eval()

fusion_model = FusionModel().to(device)

fusion_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "Results", "fusion", "fusion_model.pth"),
        map_location=device
    )
)

fusion_model.eval()

print("Models loaded successfully ✔")

# =========================
# STORAGE
# =========================
speech_embs = []
text_embs = []
fusion_embs = []
labels_all = []

# =========================
# EXTRACT EMBEDDINGS
# =========================
with torch.no_grad():

    for (speech_x, labels), text_batch in zip(speech_loader, text_loader):

        speech_x = speech_x.to(device)

        input_ids = text_batch["input_ids"].to(device)
        attention_mask = text_batch["attention_mask"].to(device)

        # =========================
        # SPEECH EMBEDDING
        # =========================
        speech_out = speech_model.net[0:5](speech_x)

        if len(speech_out.shape) > 2:
            speech_out = speech_out.mean(dim=1)

        speech_out = speech_out.view(speech_out.size(0), -1)

        # =========================
        # TEXT EMBEDDING
        # =========================
        bert_output = text_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_out = bert_output.pooler_output

        # =========================
        # FUSION EMBEDDING
        # =========================
        fusion_out = torch.cat([speech_out, text_out], dim=1)

        # =========================
        # SAVE
        # =========================
        speech_embs.append(speech_out.cpu().numpy())
        text_embs.append(text_out.cpu().numpy())
        fusion_embs.append(fusion_out.cpu().numpy())

        labels_all.append(labels.numpy())

# =========================
# NUMPY CONVERSION
# =========================
speech_embs = np.concatenate(speech_embs, axis=0)
text_embs = np.concatenate(text_embs, axis=0)
fusion_embs = np.concatenate(fusion_embs, axis=0)

labels_all = np.concatenate(labels_all, axis=0)

print("Speech embeddings:", speech_embs.shape)
print("Text embeddings:", text_embs.shape)
print("Fusion embeddings:", fusion_embs.shape)

# =========================
# NORMALIZATION
# =========================
speech_embs = StandardScaler().fit_transform(speech_embs)
text_embs = StandardScaler().fit_transform(text_embs)
fusion_embs = StandardScaler().fit_transform(fusion_embs)

# =========================
# EMOTION LABELS
# =========================
emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant Surprise",
    "Sad"
]

# =========================
# TSNE FUNCTION
# =========================
def run_tsne(embeddings, labels, title, save_name):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )

    plt.title(title, fontsize=16)

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        emotion_names,
        title="Emotion",
        loc="best"
    )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.grid(alpha=0.3)

    save_path = os.path.join(plots_dir, save_name)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {save_path}")

    plt.show()

# =========================
# RUN TSNE
# =========================
print("Running Speech t-SNE...")
run_tsne(
    speech_embs,
    labels_all,
    "Speech Features t-SNE",
    "speech_tsne.png"
)

print("Running Text t-SNE...")
run_tsne(
    text_embs,
    labels_all,
    "Text Features t-SNE",
    "text_tsne.png"
)

print("Running Fusion t-SNE...")
run_tsne(
    fusion_embs,
    labels_all,
    "Fusion Features t-SNE",
    "fusion_tsne.png"
)

print("Done ✔")