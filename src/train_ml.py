"""
Stage 4: Classical ML Baselines for Resume Classification.

Usage:
    python src/train_ml.py

Reads:   data/train.csv, data/validation.csv, data/test.csv
Writes:
    outputs/model_results.csv
    outputs/final_test_report.txt
    outputs/confusion_matrix_ml.png
    models/tfidf_vectorizer.pkl
    models/classical_model.pkl
    models/label_encoder.pkl

Experiments
───────────
  Exp-A  TF-IDF unigrams  (1,1)  ×  LR | LinearSVC | MultinomialNB
  Exp-B  TF-IDF bigrams   (1,2)  ×  LR | LinearSVC | MultinomialNB

Model selection → best Validation Macro-F1.
Test evaluation → ONCE, on the selected model only.
"""

import os
import sys
import logging
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MODELS_DIR, OUTPUTS_DIR,
    TFIDF_MODEL_PATH, CLASSICAL_MODEL_PATH, LABEL_ENCODER_PATH,
    MODEL_RESULTS_CSV, FINAL_TEST_REPORT, CM_ML_PATH,
    RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_splits():
    for path in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not os.path.exists(path):
            log.error("Missing file: %s — run split_data.py first.", path)
            sys.exit(1)

    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    test  = pd.read_csv(TEST_CSV)

    for df, name in ((train, "train"), (val, "val"), (test, "test")):
        df["clean_text"] = df["clean_text"].fillna("")

    log.info("Loaded  train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ── Label encoder ─────────────────────────────────────────────────────────────

def encode_labels(train, val, test):
    le = LabelEncoder()
    le.fit(train["category"])
    y_train = le.transform(train["category"])
    y_val   = le.transform(val["category"])
    y_test  = le.transform(test["category"])
    joblib.dump(le, LABEL_ENCODER_PATH)
    log.info("Label encoder saved → %s  (%d classes)", LABEL_ENCODER_PATH, len(le.classes_))
    return le, y_train, y_val, y_test


# ── TF-IDF configs ────────────────────────────────────────────────────────────

TFIDF_CONFIGS = {
    "Exp-A (1,1)": dict(ngram_range=(1, 1), min_df=2, max_df=0.95, sublinear_tf=True),
    "Exp-B (1,2)": dict(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                        max_features=50_000),   # bigrams need a cap to stay fast
}

# ── Classifier configs ────────────────────────────────────────────────────────

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced", max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "MultinomialNB": MultinomialNB(alpha=0.1),
    }


# ── Metrics helper ────────────────────────────────────────────────────────────

def score(y_true, y_pred) -> dict:
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred,    average="macro",    zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true, y_pred,        average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 4),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def run_experiments(X_train_raw, X_val_raw, y_train, y_val):
    """
    For every TF-IDF config × classifier:
      1. Fit TF-IDF on train only.
      2. Transform train + val.
      3. Fit classifier on train features.
      4. Score on val.
    Returns a list of result dicts and a dict of fitted pipelines.
    """
    results   = []
    pipelines = {}   # key → fitted sklearn Pipeline

    for tfidf_name, tfidf_kwargs in TFIDF_CONFIGS.items():
        log.info("Fitting TF-IDF: %s", tfidf_name)
        vectorizer = TfidfVectorizer(**tfidf_kwargs)
        X_train = vectorizer.fit_transform(X_train_raw)   # fit ONLY on train
        X_val   = vectorizer.transform(X_val_raw)

        log.info("  Vocabulary size: %d", X_train.shape[1])

        for clf_name, clf in get_classifiers().items():
            key = f"{tfidf_name} | {clf_name}"
            log.info("  Training: %s", key)

            # MultinomialNB requires non-negative features.
            # sublinear_tf with TF-IDF already gives non-negative values, but
            # just in case of any negative idf values, shift to be safe.
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict(X_val)

            metrics = score(y_val, y_pred_val)
            metrics["experiment"] = tfidf_name
            metrics["classifier"] = clf_name
            metrics["model_key"]  = key
            results.append(metrics)

            # Store the complete pipeline (vectorizer + clf) for later saving
            pipe = Pipeline([
                ("tfidf", vectorizer),
                ("clf",   clf),
            ])
            pipelines[key] = pipe

            log.info("    Val Macro-F1=%.4f  Acc=%.4f", metrics["f1_macro"], metrics["accuracy"])

    return results, pipelines


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(results_df: pd.DataFrame):
    cols = ["model_key", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    table = results_df[cols].sort_values("f1_macro", ascending=False).reset_index(drop=True)

    sep = "─" * 100
    print(f"\n{sep}")
    print("  VALIDATION SET — MODEL COMPARISON  (sorted by Macro-F1)")
    print(sep)
    header = f"  {'Model':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1-Mac':>8} {'F1-Wt':>7}"
    print(header)
    print("  " + "─" * 96)
    for _, row in table.iterrows():
        print(
            f"  {row['model_key']:<45} "
            f"{row['accuracy']:>7.4f} "
            f"{row['precision_macro']:>7.4f} "
            f"{row['recall_macro']:>7.4f} "
            f"{row['f1_macro']:>8.4f} "
            f"{row['f1_weighted']:>7.4f}"
        )
    print(sep + "\n")


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, class_names, path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    size = max(12, len(class_names))
    fig, ax = plt.subplots(figsize=(size, size - 2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, annot_kws={"size": 7},
        linewidths=0.3, linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)


# ── Final test report ─────────────────────────────────────────────────────────

def write_test_report(y_test, y_pred_test, class_names, best_key,
                      val_f1, test_metrics, path: str):
    buf = StringIO()

    def w(line=""):
        buf.write(line + "\n")

    sep = "═" * 60
    w(sep)
    w("  FINAL TEST SET REPORT")
    w(sep)
    w(f"  Best model (by Val Macro-F1) : {best_key}")
    w(f"  Best validation Macro-F1     : {val_f1:.4f}")
    w()
    w("  Test Set Metrics")
    w("  " + "─" * 40)
    for k, v in test_metrics.items():
        w(f"  {k:<22}: {v:.4f}")
    w()
    w("  Per-class Classification Report")
    w("  " + "─" * 40)
    report = classification_report(
        y_test, y_pred_test,
        target_names=class_names,
        zero_division=0,
    )
    for line in report.splitlines():
        w("  " + line)
    w(sep)

    text = buf.getvalue()
    print(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Saved test report → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load
    train_df, val_df, test_df = load_splits()

    X_train_raw = train_df["clean_text"].tolist()
    X_val_raw   = val_df["clean_text"].tolist()
    X_test_raw  = test_df["clean_text"].tolist()

    # 2. Encode labels
    le, y_train, y_val, y_test = encode_labels(train_df, val_df, test_df)
    class_names = list(le.classes_)

    # 3. Train all experiments on train, score on val
    results, pipelines = run_experiments(X_train_raw, X_val_raw, y_train, y_val)

    # 4. Comparison table + CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    print_comparison_table(results_df)
    results_df.to_csv(MODEL_RESULTS_CSV, index=False)
    log.info("Saved model results → %s", MODEL_RESULTS_CSV)

    # 5. Select best model by validation Macro-F1
    best_row  = results_df.iloc[0]
    best_key  = best_row["model_key"]
    best_val_f1 = best_row["f1_macro"]
    best_pipe = pipelines[best_key]

    print(f"  BEST VALIDATION MODEL   : {best_key}")
    print(f"  BEST VALIDATION MACRO-F1: {best_val_f1:.4f}\n")

    # 6. Evaluate ONCE on the held-out test set
    log.info("Evaluating best model on TEST set …")

    # The pipeline was trained with a fitted vectorizer; for the test set we
    # only need to call predict (transform is handled inside the pipeline).
    # However the pipeline object stores the fitted vectorizer from the last
    # experiment loop iteration, which might differ. Rebuild cleanly:
    tfidf_kwargs = TFIDF_CONFIGS[best_row["experiment"]]
    clf_name     = best_row["classifier"]

    final_vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X_train_final = final_vectorizer.fit_transform(X_train_raw)   # fit on train only
    X_test_final  = final_vectorizer.transform(X_test_raw)

    final_clf = get_classifiers()[clf_name]
    final_clf.fit(X_train_final, y_train)
    y_pred_test = final_clf.predict(X_test_final)

    test_metrics = score(y_test, y_pred_test)

    print(f"  FINAL TEST MACRO-F1     : {test_metrics['f1_macro']:.4f}\n")

    # 7. Save artefacts
    final_pipeline = Pipeline([
        ("tfidf", final_vectorizer),
        ("clf",   final_clf),
    ])
    joblib.dump(final_vectorizer, TFIDF_MODEL_PATH)
    joblib.dump(final_pipeline,   CLASSICAL_MODEL_PATH)
    log.info("Saved TF-IDF vectorizer → %s", TFIDF_MODEL_PATH)
    log.info("Saved classical pipeline → %s", CLASSICAL_MODEL_PATH)

    # 8. Confusion matrix
    save_confusion_matrix(
        y_test, y_pred_test, class_names,
        path=CM_ML_PATH,
        title=f"Confusion Matrix (Test) — {best_key}",
    )

    # 9. Full test report
    write_test_report(
        y_test, y_pred_test, class_names,
        best_key=best_key,
        val_f1=best_val_f1,
        test_metrics=test_metrics,
        path=FINAL_TEST_REPORT,
    )

    log.info("Stage 4 complete.")
