import os
import pandas as pd
import matplotlib.pyplot as plt
os.makedirs("Results/tables", exist_ok=True)
results = {
    "Model": [
        "Speech-only",
        "Text-only",
        "Fusion Model"
    ],

    "Accuracy (%)": [
        93.14,
        91.80,
        96.50
    ],

    "Input Type": [
        "MFCC Features",
        "BERT Embeddings",
        "Speech + Text"
    ]
}

df = pd.DataFrame(results)


fig, ax = plt.subplots(figsize=(8, 2.5))

ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)

table.set_fontsize(11)

table.scale(1.2, 1.8)

plt.title(
    "Model Performance Comparison",
    fontsize=14,
    pad=20
)

plt.savefig(
    "Results/tables/model_accuracy_table.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()

print("Accuracy table saved successfully.")