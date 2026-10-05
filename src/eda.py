"""
Exploratory Data Analysis for the Resume Classification dataset.

Usage:
    python src/eda.py

Reads:  data/resume_dataset.csv   (produced by build_dataset.py)
Writes: outputs/eda/*.png  +  outputs/eda/data_quality_summary.csv
"""

import os
import sys
import logging
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASET_CSV, EDA_DIR
from preprocessing import clean_text   # light clean only — preserves tech terms

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(EDA_DIR, exist_ok=True)

SHORT_THRESHOLD = 50   # words
FIGSIZE_WIDE    = (14, 5)
FIGSIZE_SQ      = (10, 8)
DPI             = 150


def save(fig, name: str):
    path = os.path.join(EDA_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Light clean that keeps tech terms ─────────────────────────────────────────
# We only strip URLs / emails / punctuation but keep casing for the word-freq
# chart label display; lowercased version used for counting.

def light_clean(text: str) -> str:
    """URLs, emails, digits, punctuation removed — alphabetic tokens kept."""
    return clean_text(text) if isinstance(text, str) else ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class distribution bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    counts = df["category"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = sns.color_palette("tab20", len(counts))
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Resume Count per Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Count",    fontsize=11)
    ax.bar_label(ax.containers[0], fontsize=8, padding=2)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    save(fig, "01_class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Word-count distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_wordcount_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(df["word_count"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("Word Count Distribution (all resumes)", fontsize=12)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")

    # Clip outliers for readability
    p99 = df["word_count"].quantile(0.99)
    clipped = df["word_count"].clip(upper=p99)
    axes[1].hist(clipped, bins=60, color="coral", edgecolor="white")
    axes[1].set_title(f"Word Count Distribution (clipped at p99={p99:.0f})", fontsize=12)
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    save(fig, "02_wordcount_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Character-count distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_charcount_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["character_count"], bins=60, color="mediumseagreen", edgecolor="white")
    ax.set_title("Character Count Distribution", fontsize=12)
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    save(fig, "03_charcount_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-30 word-frequency chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_word_frequency(df: pd.DataFrame):
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))

    all_text = " ".join(df["text"].fillna("").tolist())
    cleaned  = light_clean(all_text)
    tokens   = [t for t in cleaned.split() if t not in stop and len(t) > 2]
    top30    = Counter(tokens).most_common(30)

    words, freqs = zip(*top30)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(list(reversed(words)), list(reversed(freqs)),
                   color=sns.color_palette("viridis", 30))
    ax.set_title("Top 30 Most Frequent Words", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    save(fig, "04_top30_word_frequency.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Overall WordCloud
# ─────────────────────────────────────────────────────────────────────────────

def plot_wordcloud(df: pd.DataFrame):
    all_text = " ".join(df["text"].fillna("").tolist())
    cleaned  = light_clean(all_text)

    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        max_words=200,
        colormap="tab20",
        collocations=False,
    ).generate(cleaned)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Overall Resume WordCloud", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_wordcloud.png")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: top-N ngrams bar chart
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ngrams(df: pd.DataFrame, n: int, top_k: int, title: str, fname: str):
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))

    corpus = df["text"].fillna("").apply(light_clean).tolist()
    vec = CountVectorizer(
        ngram_range=(n, n),
        stop_words=list(stop),
        max_features=5000,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    X = vec.fit_transform(corpus)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top   = sorted(zip(vocab, freqs), key=lambda x: -x[1])[:top_k]

    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(list(reversed(words)), list(reversed(counts)),
            color=sns.color_palette("magma", top_k))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 6 / 7 / 8. Unigrams, Bigrams, Trigrams
# ─────────────────────────────────────────────────────────────────────────────

def plot_ngrams(df: pd.DataFrame):
    _plot_ngrams(df, 1, 20, "Top 20 Unigrams",  "06_top20_unigrams.png")
    _plot_ngrams(df, 2, 20, "Top 20 Bigrams",   "07_top20_bigrams.png")
    _plot_ngrams(df, 3, 20, "Top 20 Trigrams",  "08_top20_trigrams.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Class-wise TF-IDF important terms
# ─────────────────────────────────────────────────────────────────────────────

def plot_classwise_tfidf(df: pd.DataFrame, top_k: int = 8):
    """For each category, show the top-k TF-IDF terms."""
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))

    categories = sorted(df["category"].unique())
    n_cats = len(categories)
    ncols  = 4
    nrows  = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
    axes_flat = axes.flatten()

    # Fit TF-IDF on per-category pseudo-docs (one doc = all resumes in category)
    cat_docs = [
        " ".join(df[df["category"] == c]["text"].fillna("").apply(light_clean))
        for c in categories
    ]
    tfidf = TfidfVectorizer(
        stop_words=list(stop),
        max_features=10_000,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    tfidf.fit(cat_docs)
    vocab = tfidf.get_feature_names_out()

    for idx, (cat, doc) in enumerate(zip(categories, cat_docs)):
        vec   = tfidf.transform([doc]).toarray()[0]
        top_i = vec.argsort()[-top_k:][::-1]
        terms = [vocab[i] for i in top_i]
        scores = [vec[i] for i in top_i]

        ax = axes_flat[idx]
        ax.barh(list(reversed(terms)), list(reversed(scores)),
                color=sns.color_palette("coolwarm", top_k))
        ax.set_title(cat, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("TF-IDF", fontsize=7)

    # Hide empty subplots
    for j in range(n_cats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Class-wise Top TF-IDF Terms", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "09_classwise_tfidf.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Data-quality summary  (CSV + printed table)
# ─────────────────────────────────────────────────────────────────────────────

def save_quality_summary(df: pd.DataFrame):
    SHORT_THRESHOLD = 50

    summary = pd.DataFrame({
        "metric": [
            "total_resumes",
            "total_categories",
            "missing_text",
            "empty_text",
            f"short_resumes_lt_{SHORT_THRESHOLD}_words",
            "duplicate_text",
            "min_word_count",
            "max_word_count",
            "mean_word_count",
            "median_word_count",
        ],
        "value": [
            len(df),
            df["category"].nunique(),
            int(df["text"].isna().sum()),
            int((df["text"].fillna("").str.strip() == "").sum()),
            int((df["word_count"] < SHORT_THRESHOLD).sum()),
            int(df.duplicated(subset=["text"]).sum()),
            int(df["word_count"].min()),
            int(df["word_count"].max()),
            round(df["word_count"].mean(), 1),
            round(df["word_count"].median(), 1),
        ],
    })

    path = os.path.join(EDA_DIR, "10_data_quality_summary.csv")
    summary.to_csv(path, index=False)
    log.info("Saved → %s", path)

    print("\n── Data Quality Summary " + "─" * 30)
    print(summary.to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(DATASET_CSV):
        log.error("Dataset CSV not found at %s — run build_dataset.py first.", DATASET_CSV)
        sys.exit(1)

    log.info("Loading dataset …")
    df = pd.read_csv(DATASET_CSV)
    log.info("Loaded %d rows, %d categories", len(df), df["category"].nunique())

    log.info("[1/10] Class distribution …")
    plot_class_distribution(df)

    log.info("[2/10] Word-count distribution …")
    plot_wordcount_distribution(df)

    log.info("[3/10] Character-count distribution …")
    plot_charcount_distribution(df)

    log.info("[4/10] Top-30 word frequency …")
    plot_top_word_frequency(df)

    log.info("[5/10] WordCloud …")
    plot_wordcloud(df)

    log.info("[6-8/10] Unigrams / Bigrams / Trigrams …")
    plot_ngrams(df)

    log.info("[9/10] Class-wise TF-IDF …")
    plot_classwise_tfidf(df)

    log.info("[10/10] Data quality summary …")
    save_quality_summary(df)

    log.info("EDA complete. All outputs saved to: %s", EDA_DIR)
