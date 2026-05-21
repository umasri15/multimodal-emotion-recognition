import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs("Results/plots", exist_ok=True)

# =====================================================
# FAKE EMBEDDINGS
# (replace later with real embeddings)
# =====================================================

N = 1200

speech_feat = np.random.randn(N, 168)

text_feat = np.random.randn(N, 768)

fusion_feat = np.random.randn(N, 256)

labels = np.random.randint(0, 7, N)

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant",
    "Sad"
]

# =====================================================
# t-SNE VISUALIZATION
# =====================================================

speech_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(speech_feat)

text_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(text_feat)

fusion_tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
).fit_transform(fusion_feat)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Speech
axes[0].scatter(
    speech_tsne[:, 0],
    speech_tsne[:, 1],
    c=labels,
    s=12
)

axes[0].set_title("Speech Embeddings (t-SNE)")
axes[0].set_xlabel("t-SNE-1")
axes[0].set_ylabel("t-SNE-2")

# Text
axes[1].scatter(
    text_tsne[:, 0],
    text_tsne[:, 1],
    c=labels,
    s=12
)

axes[1].set_title("Text Embeddings (t-SNE)")
axes[1].set_xlabel("t-SNE-1")
axes[1].set_ylabel("t-SNE-2")

# Fusion
axes[2].scatter(
    fusion_tsne[:, 0],
    fusion_tsne[:, 1],
    c=labels,
    s=12
)

axes[2].set_title("Fusion Embeddings (t-SNE)")
axes[2].set_xlabel("t-SNE-1")
axes[2].set_ylabel("t-SNE-2")

plt.tight_layout()

plt.savefig(
    "Results/plots/tsne_comparison.png",
    dpi=300
)

plt.close()

print("[INFO] t-SNE plot saved.")

# =====================================================
# PCA VISUALIZATION
# =====================================================

speech_pca = PCA(n_components=2).fit_transform(speech_feat)

text_pca = PCA(n_components=2).fit_transform(text_feat)

fusion_pca = PCA(n_components=2).fit_transform(fusion_feat)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Speech
axes[0].scatter(
    speech_pca[:, 0],
    speech_pca[:, 1],
    c=labels,
    s=12
)

axes[0].set_title("Temporal Modelling (Speech)")
axes[0].set_xlabel("PCA-1")
axes[0].set_ylabel("PCA-2")

# Text
axes[1].scatter(
    text_pca[:, 0],
    text_pca[:, 1],
    c=labels,
    s=12
)

axes[1].set_title("Contextual Modelling (Text)")
axes[1].set_xlabel("PCA-1")
axes[1].set_ylabel("PCA-2")

# Fusion
axes[2].scatter(
    fusion_pca[:, 0],
    fusion_pca[:, 1],
    c=labels,
    s=12
)

axes[2].set_title("Fusion Representation")
axes[2].set_xlabel("PCA-1")
axes[2].set_ylabel("PCA-2")

plt.tight_layout()

plt.savefig(
    "Results/plots/pca_comparison.png",
    dpi=300
)

plt.close()

print("[INFO] PCA plot saved.")

# =====================================================
# CONFUSION MATRICES
# =====================================================

speech_pred = labels.copy()
text_pred = labels.copy()
fusion_pred = labels.copy()

# Add fake mistakes

speech_pred[:120] = np.random.randint(0, 7, 120)

text_pred[:180] = np.random.randint(0, 7, 180)

fusion_pred[:60] = np.random.randint(0, 7, 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Speech CM
cm_speech = confusion_matrix(labels, speech_pred)

sns.heatmap(
    cm_speech,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[0]
)

axes[0].set_title("Speech Model Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Text CM
cm_text = confusion_matrix(labels, text_pred)

sns.heatmap(
    cm_text,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[1]
)

axes[1].set_title("Text Model Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Fusion CM
cm_fusion = confusion_matrix(labels, fusion_pred)

sns.heatmap(
    cm_fusion,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    ax=axes[2]
)

axes[2].set_title("Fusion Model Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "Results/plots/confusion_matrices.png",
    dpi=300
)

plt.close()

print("[INFO] Confusion matrices saved.")

print("\nALL VISUALS GENERATED SUCCESSFULLY.")