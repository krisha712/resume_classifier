"""
Shared utility helpers: metrics, plots, persistence.
"""

import os
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from config import OUTPUTS_DIR, LABEL_ENCODER_PATH

log = logging.getLogger(__name__)

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_object(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    log.info("Saved → %s", path)


def load_object(path: str):
    obj = joblib.load(path)
    log.info("Loaded ← %s", path)
    return obj


def save_label_encoder(le: LabelEncoder):
    save_object(le, LABEL_ENCODER_PATH)


def load_label_encoder() -> LabelEncoder:
    return load_object(LABEL_ENCODER_PATH)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Return a dict with all required scalar metrics."""
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "precision_macro":   precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_macro":      recall_score(y_true, y_pred,    average="macro",    zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred,        average="macro",    zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred,        average="weighted", zero_division=0),
        "precision_weighted":precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":   recall_score(y_true, y_pred,    average="weighted", zero_division=0),
    }


def print_metrics(metrics: dict, model_name: str = ""):
    header = f"── {model_name} Metrics " if model_name else "── Metrics "
    print(header + "─" * max(0, 50 - len(header)))
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print()


def print_classification_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str = "model",
                          save: bool = True):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        log.info("Saved confusion matrix → %s", path)

    return fig


def plot_class_distribution(df: pd.DataFrame, category_col: str = "category",
                             save: bool = True):
    """Bar chart of resume counts per category."""
    counts = df[category_col].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Resume Count per Category", fontsize=13)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
        fig.savefig(path, dpi=150)
        log.info("Saved class distribution → %s", path)

    return fig


def plot_training_history(history, save: bool = True):
    """Plot Keras training accuracy & loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle("Neural Network Training History", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "nn_training_history.png")
        fig.savefig(path, dpi=150)
        log.info("Saved training history → %s", path)

    return fig


# ── Error analysis ─────────────────────────────────────────────────────────────

def error_analysis(y_true, y_pred, texts, n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of misclassified examples (up to n rows)."""
    errors = pd.DataFrame({
        "true_label": y_true,
        "pred_label": y_pred,
        "text_snippet": [t[:200] for t in texts],
    })
    errors = errors[errors["true_label"] != errors["pred_label"]]
    log.info("Total misclassifications: %d / %d", len(errors), len(y_true))
    return errors.head(n).reset_index(drop=True)
