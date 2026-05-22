# Multimodal Emotion Recognition using Speech, Text, and Fusion Learning

A deep learning system that identifies emotions using three different approaches:

- **Speech Analysis**: Classifies emotions from audio recordings alone
- **Text Analysis**: Identifies emotions from written transcripts
- **Multimodal Fusion**: Combines both audio and text for better accuracy

The system works by analyzing emotional patterns in speech signals alongside the actual words being spoken, leading to more reliable emotion detection than using either source alone.

---

# 🗂️ Project Structure
```text
📦 multimodal-emotion-recognition/
│
├── 📂 analysis/
│   └── 📂 plots/
│       ├── 📄 generate_visuals.py
│       ├── 📄 generate_accuracy_table.py
│       ├── 📄 run_tsne.py
│       └── 📄 __init__.py
│
├── 📂 models/
│   │
│   ├── 📂 speech_pipeline/
│   │   ├── 📄 dataset.py
│   │   ├── 📄 feature_extraction.py
│   │   ├── 📄 model.py
│   │   ├── 📄 train.py
│   │   └── 📄 test.py
│   │
│   ├── 📂 text_pipeline/
│   │   ├── 📄 dataset.py
│   │   ├── 📄 model.py
│   │   ├── 📄 train.py
│   │   └── 📄 test.py
│   │
│   └── 📂 fusion_pipeline/
│       ├── 📄 model.py
│       ├── 📄 train.py
│       └── 📄 test.py
│
├── 📂 Results/
│   │
│   ├── 📂 plots/
│   │   ├── 📄 speech_tsne.png
│   │   ├── 📄 text_tsne.png
│   │   ├── 📄 fusion_tsne.png
│   │   ├── 📄 confusion_matrices.png
│   │   └── 📄 model_accuracy_comparison.png
│   │
│   ├── 📂 tables/
│   │   └── 📄 model_accuracy_table.png
│   │
│   ├── 📂 speech/
│   │
│   ├── 📂 text/
│   │
│   └── 📂 fusion/
│
├── 📂 utils/
│   └── 📄 config.py
│
├── 📂 report/
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

---

# 📦 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/umasri15/Multimodal-Emotion-Recognition.git

cd Multimodal-Emotion-Recognition
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🎙️ Speech Emotion Recognition

## Train Speech Model

```bash
python -m models.speech_pipeline.train
```

## Test Speech Model

```bash
python models/speech_pipeline/test.py
```

---

# 📝 Text Emotion Recognition

## Train Text Model

```bash
python -m models.text_pipeline.train
```

## Test Text Model

```bash
python models/text_pipeline/test.py
```

---

# 🔀 Fusion Emotion Recognition

## Train Fusion Model

```bash
python -m models.fusion_pipeline.train
```

## Test Fusion Model

```bash
python models/fusion_pipeline/test.py
```

---
## 📊 Visualizations & Analysis

### t-SNE Embeddings

#### Speech Embeddings
![Speech t-SNE](Results/plots/speech_tsne.png)

#### Text Embeddings
![Text t-SNE](Results/plots/text_tsne.png)

#### Fusion Embeddings
![Fusion t-SNE](Results/plots/fusion_tsne.png)

---

### Confusion Matrices

![Confusion Matrices](Results/plots/confusion_matrices.png)

---

## 📈 Model Performance

| Model | Accuracy | Input Type |
|---|---|---|
| Speech-only | 93.14% | MFCC Features |
| Text-only | 91.80% | BERT Embeddings |
| Fusion Model | 96.50% | Speech + Text |

---

### Performance Table

![Performance Table](Results/tables/model_accuracy_table.png)

# 🔗 GitHub Repository

https://github.com/umasri15/Multimodal-Emotion-Recognition

Important Note on GitHub & Project Files (Multimodal Emotion Recognition System)

The GitHub repository contains the complete source code for the Multimodal Emotion Recognition system, including speech, text, and fusion pipelines along with all training and evaluation scripts. Due to size constraints, large files such as datasets (TESS and other corpora), extracted features (MFCC/embeddings), and trained model checkpoints are not included in the repository.

For full reproducibility, a Google Drive link is provided containing the complete datasets, extracted features, trained models, and all experimental outputs.

The GitHub repo is lightweight and intended for easy setup and code review, while the Drive folder provides the complete working environment for training and evaluation.