"""
Build the master dataset CSV from raw PDF resumes.

Usage:
    python src/build_dataset.py

Output:
    data/resume_dataset.csv
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd

# Allow running directly from project root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import DATA_DIR, DATASET_CSV, EXCLUDE_CATEGORIES
from pdf_extractor import extract_text_from_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SHORT_RESUME_THRESHOLD = 50  # words


# ─── Build ─────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    data_path = Path(DATA_DIR)
    records = []

    category_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    for cat_dir in category_dirs:
        category = cat_dir.name
        if category in EXCLUDE_CATEGORIES:
            log.info("Skipping excluded category: %s", category)
            continue

        pdf_files = list(cat_dir.glob("*.pdf"))
        log.info("%-25s → %d PDFs", category, len(pdf_files))

        for pdf_path in pdf_files:
            text = extract_text_from_pdf(str(pdf_path))
            records.append({
                "file_name":       pdf_path.name,
                "file_path":       str(pdf_path.relative_to(data_path.parent)),
                "category":        category,
                "text":            text,
                "character_count": len(text),
                "word_count":      len(text.split()) if text.strip() else 0,
            })

    df = pd.DataFrame(records, columns=[
        "file_name", "file_path", "category", "text", "character_count", "word_count"
    ])
    return df


# ─── Data-quality report ───────────────────────────────────────────────────────

def print_quality_report(df: pd.DataFrame):
    sep = "─" * 55

    missing_text = df["text"].isna().sum()
    empty_text   = (df["text"].str.strip() == "").sum()
    short_text   = (df["word_count"] < SHORT_RESUME_THRESHOLD).sum()
    dup_text     = df.duplicated(subset=["text"]).sum()

    print(f"\n{sep}")
    print("  DATA QUALITY REPORT")
    print(sep)
    print(f"  Total resumes          : {len(df)}")
    print(f"  Total categories       : {df['category'].nunique()}")
    print(f"  Missing text (NaN)     : {missing_text}")
    print(f"  Empty text             : {empty_text}")
    print(f"  Very short (<{SHORT_RESUME_THRESHOLD} words) : {short_text}")
    print(f"  Duplicate text         : {dup_text}")
    print(f"  Min  word count        : {df['word_count'].min()}")
    print(f"  Max  word count        : {df['word_count'].max()}")
    print(f"  Mean word count        : {df['word_count'].mean():.1f}")
    print(f"  Median word count      : {df['word_count'].median():.1f}")
    print(sep)

    print("\n  Samples per category:")
    counts = df.groupby("category").size().sort_values(ascending=False)
    for cat, cnt in counts.items():
        print(f"    {cat:<30} {cnt:>4}")
    print(sep + "\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting dataset build …")
    df = build_dataset()

    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)
    log.info("Saved dataset → %s", DATASET_CSV)

    print_quality_report(df)
