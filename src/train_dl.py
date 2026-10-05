"""
Stage 5: Deep-Learning Resume Classifier.

Primary approach  : Word2Vec embeddings  →  Bidirectional LSTM
Fallback approach : TF-IDF features      →  Dense MLP
  (activated automatically when TensorFlow is unavailable)

Usage:
    python src/train_dl.py

Reads:   data/train.csv, data/validation.csv, data/test.csv
Writes:
    models/word2vec.model          (gensim Word2Vec)
    models/dl_tokenizer.pkl        (Keras Tokenizer or sklearn TF-IDF)
    models/dl_label_encoder.pkl
    models/dl_model.keras          (Keras .keras format)
    outputs/dl_results.csv
    outputs/confusion_matrix_dl.png

Leakage rules
─────────────
• Word2Vec trained on train text only.
• Keras Tokenizer vocabulary built from train text only.
• Val/test are only transformed, never fitted.
• Test evaluated exactly once, after model is finalised.
"""

import os
import sys
import logging
import warnings
import time

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF C++ logs

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MODELS_DIR, OUTPUTS_DIR,
    W2V_MODEL_PATH, DL_TOKENIZER_PATH, DL_LABEL_ENC_PATH,
    DL_MODEL_PATH, DL_RESULTS_CSV, CM_DL_PATH,
    RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 20_000
MAX_SEQ_LEN  = 400
EMBED_DIM    = 100
BATCH_SIZE   = 32
MAX_EPOCHS   = 8
PATIENCE     = 2

# Classical baseline for comparison
CLASSICAL_TEST_F1 = 0.6393

# ── Check TensorFlow availability ─────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
    log.info("TensorFlow %s detected — using BiLSTM approach.", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    log.warning("TensorFlow not found — falling back to Dense MLP on TF-IDF features.")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_splits():
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not os.path.exists(p):
            log.error("Missing: %s — run split_data.py first.", p)
            sys.exit(1)
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    test  = pd.read_csv(TEST_CSV)
    for df in (train, val, test):
        df["clean_text"] = df["clean_text"].fillna("")
    log.info("Loaded  train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


def fit_label_encoder(train_df):
    le = LabelEncoder()
    le.fit(train_df["category"])
    joblib.dump(le, DL_LABEL_ENC_PATH)
    log.info("Label encoder → %s  (%d classes)", DL_LABEL_ENC_PATH, len(le.classes_))
    return le


def score(y_true, y_pred) -> dict:
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred,    average="macro",    zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true, y_pred,        average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 4),
    }


def save_confusion_matrix(y_true, y_pred, class_names, title: str):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    size = max(12, len(class_names))
    fig, ax = plt.subplots(figsize=(size, size - 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 7},
                linewidths=0.3, linecolor="gray")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    fig.savefig(CM_DL_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", CM_DL_PATH)


def print_comparison(val_metrics, test_metrics):
    sep = "─" * 55
    print(f"\n{sep}")
    print("  DL MODEL vs CLASSICAL (LinearSVC)")
    print(sep)
    print(f"  {'Metric':<22} {'DL Val':>9} {'DL Test':>9} {'SVC Test':>9}")
    print("  " + "─" * 51)
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        svc = CLASSICAL_TEST_F1 if k == "f1_macro" else "—"
        svc_str = f"{svc:>9.4f}" if isinstance(svc, float) else f"{'—':>9}"
        print(f"  {k:<22} {val_metrics[k]:>9.4f} {test_metrics[k]:>9.4f} {svc_str}")
    print(sep + "\n")


def save_results(approach, val_metrics, test_metrics):
    rows = []
    for split, m in (("validation", val_metrics), ("test", test_metrics)):
        row = {"approach": approach, "split": split}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(DL_RESULTS_CSV, index=False)
    log.info("Saved DL results → %s", DL_RESULTS_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY APPROACH: Word2Vec + BiLSTM
# ══════════════════════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> list:
    """Fast whitespace tokeniser — no NLTK overhead."""
    return text.lower().split()


def train_word2vec(train_texts):
    from gensim.models import Word2Vec
    log.info("Tokenising training corpus for Word2Vec …")
    sentences = [simple_tokenize(t) for t in train_texts]
    log.info("Training Word2Vec (vector_size=%d, window=5, min_count=2, epochs=5) …", EMBED_DIM)
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=5,
        min_count=2,
        workers=2,
        epochs=5,
        seed=RANDOM_STATE,
    )
    w2v.save(W2V_MODEL_PATH)
    log.info("Word2Vec saved → %s  (vocab=%d)", W2V_MODEL_PATH, len(w2v.wv))
    return w2v


def build_keras_tokenizer(train_texts):
    from tensorflow.keras.preprocessing.text import Tokenizer   # type: ignore
    log.info("Building Keras tokenizer (vocab_size=%d) on training data only …", VOCAB_SIZE)
    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    joblib.dump(tok, DL_TOKENIZER_PATH)
    log.info("Tokenizer saved → %s", DL_TOKENIZER_PATH)
    return tok


def texts_to_padded(tok, texts):
    from tensorflow.keras.preprocessing.sequence import pad_sequences   # type: ignore
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


def build_embedding_matrix(tok, w2v):
    word_index = tok.word_index
    n_words    = min(VOCAB_SIZE, len(word_index)) + 1
    matrix     = np.zeros((n_words, EMBED_DIM), dtype="float32")
    hits = 0
    for word, idx in word_index.items():
        if idx >= VOCAB_SIZE:
            continue
        if word in w2v.wv:
            matrix[idx] = w2v.wv[word]
            hits += 1
    log.info("Embedding matrix: %d/%d words covered by Word2Vec (%.1f %%)",
             hits, n_words - 1, 100 * hits / max(1, n_words - 1))
    return matrix, n_words


def build_bilstm(n_words, n_classes, embed_matrix):
    import tensorflow as tf
    from tensorflow.keras import layers, models   # type: ignore

    tf.random.set_seed(RANDOM_STATE)
    inp = layers.Input(shape=(MAX_SEQ_LEN,))
    x   = layers.Embedding(
            input_dim=n_words,
            output_dim=EMBED_DIM,
            weights=[embed_matrix],
            input_length=MAX_SEQ_LEN,
            trainable=True,           # fine-tune embeddings
          )(inp)
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)
    return model


def run_bilstm(train_df, val_df, test_df, le):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   # type: ignore

    n_classes   = len(le.classes_)
    y_train     = le.transform(train_df["category"])
    y_val       = le.transform(val_df["category"])
    y_test      = le.transform(test_df["category"])

    train_texts = train_df["clean_text"].tolist()
    val_texts   = val_df["clean_text"].tolist()
    test_texts  = test_df["clean_text"].tolist()

    # Step 1: Word2Vec (train only)
    w2v = train_word2vec(train_texts)

    # Step 2: Keras tokenizer (train only)
    tok = build_keras_tokenizer(train_texts)

    # Step 3: Sequences + padding
    X_train = texts_to_padded(tok, train_texts)
    X_val   = texts_to_padded(tok, val_texts)
    X_test  = texts_to_padded(tok, test_texts)

    # Step 4: Embedding matrix
    embed_matrix, n_words = build_embedding_matrix(tok, w2v)

    # Step 5: Class weights from train only
    cw_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(cw_arr[i]) for i in range(len(cw_arr))}

    # Step 6: Build model
    model = build_bilstm(n_words, n_classes, embed_matrix)

    # Step 7: Train
    ckpt_path = os.path.join(MODELS_DIR, "_bilstm_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    log.info("Training BiLSTM (max %d epochs, batch %d) …", MAX_EPOCHS, BATCH_SIZE)
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    log.info("Training finished in %.1f s", time.time() - t0)

    # Step 8: Validation metrics
    y_pred_val  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    val_metrics = score(y_val, y_pred_val)
    log.info("Validation — Macro-F1=%.4f  Acc=%.4f", val_metrics["f1_macro"], val_metrics["accuracy"])

    # Step 9: Test metrics (once)
    y_pred_test  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_metrics = score(y_test, y_pred_test)
    log.info("Test       — Macro-F1=%.4f  Acc=%.4f", test_metrics["f1_macro"], test_metrics["accuracy"])

    # Step 10: Save model
    model.save(DL_MODEL_PATH)
    log.info("Keras model saved → %s", DL_MODEL_PATH)

    # Clean up temp checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Step 11: Training history plot
    _plot_history(history)

    return y_val, y_pred_val, y_test, y_pred_test, val_metrics, test_metrics


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []),     label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history.get("loss", []),     label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("BiLSTM Training History", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "dl_training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved training history → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK APPROACH: TF-IDF + Dense MLP (when TensorFlow is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def run_tfidf_mlp(train_df, val_df, test_df, le):
    """
    Fallback: TF-IDF (unigrams, max 20k) → Dense(128) → Dropout → Dense(24, softmax).
    Uses TensorFlow if available; otherwise raises to signal a hard failure.
    NOTE: This branch is only reached when TF IS available but BiLSTM failed.
    For a no-TF scenario we log clearly and exit.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models   # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping   # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse

    log.info("[FALLBACK] TF-IDF + Dense MLP")

    n_classes   = len(le.classes_)
    y_train     = le.transform(train_df["category"])
    y_val       = le.transform(val_df["category"])
    y_test      = le.transform(test_df["category"])

    # Fit TF-IDF on train only
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=20_000,
                            min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["clean_text"]).toarray().astype("float32")
    X_val   = tfidf.transform(val_df["clean_text"]).toarray().astype("float32")
    X_test  = tfidf.transform(test_df["clean_text"]).toarray().astype("float32")
    joblib.dump(tfidf, DL_TOKENIZER_PATH)   # save vectorizer in tokenizer slot

    # Class weights
    cw_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(cw_arr[i]) for i in range(len(cw_arr))}

    # Model
    tf.random.set_seed(RANDOM_STATE)
    inp = layers.Input(shape=(X_train.shape[1],))
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    log.info("Training Dense MLP (max %d epochs) …", MAX_EPOCHS)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                 restore_best_weights=True, verbose=1)],
        verbose=1,
    )

    y_pred_val  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)

    val_metrics  = score(y_val,  y_pred_val)
    test_metrics = score(y_test, y_pred_test)

    model.save(DL_MODEL_PATH)
    log.info("Dense MLP saved → %s", DL_MODEL_PATH)

    return y_val, y_pred_val, y_test, y_pred_test, val_metrics, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_df, val_df, test_df = load_splits()
    le = fit_label_encoder(train_df)
    class_names = list(le.classes_)

    approach = "BiLSTM (Word2Vec)"

    if not TF_AVAILABLE:
        log.error(
            "TensorFlow is not installed.\n"
            "Install it with:  pip install tensorflow\n"
            "Then re-run this script."
        )
        sys.exit(1)

    # Try primary BiLSTM approach; fall back to Dense MLP on error
    try:
        (y_val, y_pred_val,
         y_test, y_pred_test,
         val_metrics, test_metrics) = run_bilstm(train_df, val_df, test_df, le)

    except Exception as exc:
        log.warning("BiLSTM failed (%s). Falling back to TF-IDF + Dense MLP.", exc)
        approach = "Dense MLP (TF-IDF fallback)"
        (y_val, y_pred_val,
         y_test, y_pred_test,
         val_metrics, test_metrics) = run_tfidf_mlp(train_df, val_df, test_df, le)

    # ── Results ───────────────────────────────────────────────────────────────
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  DL APPROACH : {approach}")
    print(sep)
    print("\n  Validation Metrics")
    for k, v in val_metrics.items():
        print(f"    {k:<22}: {v:.4f}")
    print("\n  Test Metrics")
    for k, v in test_metrics.items():
        print(f"    {k:<22}: {v:.4f}")
    print()
    print(f"  Per-class classification report (TEST)")
    print(classification_report(y_test, y_pred_test,
                                target_names=class_names, zero_division=0))

    print_comparison(val_metrics, test_metrics)

    print(f"  BEST VALIDATION MACRO-F1 : {val_metrics['f1_macro']:.4f}")
    print(f"  FINAL TEST MACRO-F1      : {test_metrics['f1_macro']:.4f}")
    print(f"  CLASSICAL SVC TEST F1    : {CLASSICAL_TEST_F1:.4f}")
    print(sep + "\n")

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_confusion_matrix(y_test, y_pred_test, class_names,
                          title=f"Confusion Matrix (Test) — {approach}")
    save_results(approach, val_metrics, test_metrics)

    log.info("Stage 5 complete.")
