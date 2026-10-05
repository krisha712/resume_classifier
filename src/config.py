"""
Central configuration for the Resume Classification project.
All paths are relative to the project root so the project is portable.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
EDA_DIR     = os.path.join(OUTPUTS_DIR, "eda")

# Derived artefact paths
DATASET_CSV      = os.path.join(DATA_DIR, "resume_dataset.csv")
TRAIN_CSV        = os.path.join(DATA_DIR, "train.csv")
VAL_CSV          = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV         = os.path.join(DATA_DIR, "test.csv")
TFIDF_MODEL_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LR_MODEL_PATH      = os.path.join(MODELS_DIR, "lr_classifier.joblib")
SVM_MODEL_PATH     = os.path.join(MODELS_DIR, "svm_classifier.joblib")
NN_MODEL_PATH      = os.path.join(MODELS_DIR, "nn_classifier.keras")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
CLASSICAL_MODEL_PATH = os.path.join(MODELS_DIR, "classical_model.pkl")
MODEL_RESULTS_CSV    = os.path.join(OUTPUTS_DIR, "model_results.csv")
FINAL_TEST_REPORT    = os.path.join(OUTPUTS_DIR, "final_test_report.txt")
CM_ML_PATH           = os.path.join(OUTPUTS_DIR, "confusion_matrix_ml.png")

# ── Deep-learning artefacts ────────────────────────────────────────────────────
W2V_MODEL_PATH      = os.path.join(MODELS_DIR,  "word2vec.model")
DL_TOKENIZER_PATH   = os.path.join(MODELS_DIR,  "dl_tokenizer.pkl")
DL_LABEL_ENC_PATH   = os.path.join(MODELS_DIR,  "dl_label_encoder.pkl")
DL_MODEL_PATH       = os.path.join(MODELS_DIR,  "dl_model.keras")
DL_RESULTS_CSV      = os.path.join(OUTPUTS_DIR, "dl_results.csv")
CM_DL_PATH          = os.path.join(OUTPUTS_DIR, "confusion_matrix_dl.png")

# ── Dataset ────────────────────────────────────────────────────────────────────
# Folder whose name looks like a catch-all; set to None to keep it
EXCLUDE_CATEGORIES = []   # e.g. ["Resume"] if you want to drop the misc folder

# ── Text preprocessing ─────────────────────────────────────────────────────────
MIN_TOKEN_LENGTH = 2          # drop single-char tokens
MAX_FEATURES     = 20_000     # TF-IDF vocabulary size
NGRAM_RANGE      = (1, 2)     # unigrams + bigrams

# ── Train / Val / Test split ───────────────────────────────────────────────────
TEST_SIZE  = 0.15   # 15 % held-out test
VAL_SIZE   = 0.15   # 15 % validation (from remaining train)
RANDOM_STATE = 42

# ── Neural network ─────────────────────────────────────────────────────────────
NN_EPOCHS     = 20
NN_BATCH_SIZE = 32
NN_DROPOUT    = 0.4
