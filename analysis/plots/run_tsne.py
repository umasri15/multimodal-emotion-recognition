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
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.config import BASE_DIR


# =========================
# RESULTS DIRECTORY
# =========================

plots_dir = os.path.join(
    BASE_DIR,
    "Results",
    "plots"
)

os.makedirs(
    plots_dir,
    exist_ok=True
)

# =========================
# LOAD SAVED EMBEDDINGS
# =========================

emb_dir = os.path.join(
    BASE_DIR,
    "Results",
    "embeddings"
)

speech_embs = np.load(
    os.path.join(
        emb_dir,
        "speech_embeddings.npy"
    )
)

text_embs = np.load(
    os.path.join(
        emb_dir,
        "text_embeddings.npy"
    )
)

fusion_embs = np.load(
    os.path.join(
        emb_dir,
        "fusion_embeddings.npy"
    )
)

labels_all = np.load(
    os.path.join(
        emb_dir,
        "labels.npy"
    )
)

print("Embeddings loaded successfully ✔")

print("Speech:", speech_embs.shape)
print("Text:", text_embs.shape)
print("Fusion:", fusion_embs.shape)

# =========================
# NORMALIZATION
# =========================

speech_embs = StandardScaler().fit_transform(
    speech_embs
)

text_embs = StandardScaler().fit_transform(
    text_embs
)

fusion_embs = StandardScaler().fit_transform(
    fusion_embs
)

# =========================
# EMOTION LABELS
# =========================

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Surprise",
    "Sad"
]

# =========================
# TSNE FUNCTION
# =========================

def run_tsne(
    embeddings,
    labels,
    title,
    save_name
):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(
        embeddings
    )

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )

    plt.title(
        title,
        fontsize=16
    )

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

    save_path = os.path.join(
        plots_dir,
        save_name
    )

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