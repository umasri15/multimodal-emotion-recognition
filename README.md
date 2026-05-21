Multimodal Emotion Recognition using Speech, Text, and Fusion Learning

Multimodal Emotion Recognition is a Deep Learning based AI system that classifies human emotions using:

🎙️ Speech-only learning
📝 Text-only learning
🔀 Multimodal Fusion learning

The system combines emotional cues from audio signals and textual transcripts to improve emotion classification performance using deep neural architectures built with PyTorch and Transformers.

The project was developed using the TESS (Toronto Emotional Speech Set) dataset and implements complete pipelines for:

Speech Emotion Recognition
Text Emotion Recognition
Multimodal Fusion-based Emotion Recognition
Evaluation & Visualization
🧠 Emotions Classified

The system predicts the following emotions:

 Angry
 Disgust
 Fear
 Happy
 Neutral
 Sad
 Surprise


🗂️ Project Structure
📦 multimodal-emotion-recognition/
│
├── 📂 analysis/
│   └── 📂 plots/
│       ├── 📄 generate_visuals.py
│       ├── 📄 generate_accuracy_table.py
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
│   │   ├── 📄 pca_comparison.png
│   │   ├── 📄 tsne_comparison.png
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


📦 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/umasri15/Multimodal-Emotion-Recognition.git

cd Multimodal-Emotion-Recognition
2️⃣ Create Virtual Environment
python -m venv venv

3️⃣ Activate Environment
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt
🔧 Main Dependencies
🧠 Deep Learning & NLP
torch
torchaudio
transformers
scikit-learn
numpy
pandas
🎙️ Audio Processing
librosa
soundfile
scipy
📊 Visualization
matplotlib
seaborn
📥 Dataset Setup

Download the TESS Dataset from Kaggle.

Dataset:

Toronto Emotional Speech Set (TESS)

Place dataset inside:

dataset/TESS/

The folder should contain all emotion subfolders with .wav audio files.

🎙️ Speech Emotion Recognition Pipeline
Feature Extraction

Speech features are extracted using:

MFCC (Mel Frequency Cepstral Coefficients)
Temporal speech representations
Audio normalization & silence trimming
Train Speech Model
python -m models.speech_pipeline.train
Test Speech Model
python models/speech_pipeline/test.py
📝 Text Emotion Recognition Pipeline
Text Processing

Text transcripts are processed using:

Tokenization
Contextual embeddings
Transformer-based representations

The project uses:

BERT embeddings

for contextual emotional understanding.

Train Text Model
python -m models.text_pipeline.train
Test Text Model
python models/text_pipeline/test.py
🔀 Multimodal Fusion Pipeline

The fusion model combines:

Speech embeddings
Text embeddings

into a unified emotional representation for final classification.

Train Fusion Model
python -m models.fusion_pipeline.train
Test Fusion Model
python models/fusion_pipeline/test.py
📊 Visualization & Analysis

The project includes dimensionality reduction and evaluation visualizations for analyzing learned emotional representations.

📈 Generate PCA / t-SNE / Confusion Matrix Visualizations
python -m analysis.plots.generate_visuals

Generated Outputs:

PCA cluster visualization
t-SNE cluster visualization
Confusion matrices
Model comparison plots

Saved inside:

Results/plots/
📋 Generate Accuracy Comparison Table
python -m analysis.plots.generate_accuracy_table

Output:

Results/tables/model_accuracy_table.png