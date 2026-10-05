"""
PDF text extraction using PyMuPDF (fitz).
Walks the DATA_DIR, reads every PDF, and returns a DataFrame with
columns: [file_path, category, raw_text].
"""

import os
import logging
from pathlib import Path

import fitz          # PyMuPDF
import pandas as pd

from config import DATA_DIR, EXCLUDE_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Return concatenated text from all pages of a PDF.
    Returns empty string on any extraction failure."""
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception as exc:
        log.warning("Failed to read %s: %s", pdf_path, exc)
        return ""


def build_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Walk every category sub-folder in data_dir, extract text from each PDF,
    and return a DataFrame with columns [file_path, category, raw_text].
    """
    records = []
    data_path = Path(data_dir)

    category_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    for cat_dir in category_dirs:
        category = cat_dir.name

        if category in EXCLUDE_CATEGORIES:
            log.info("Skipping excluded category: %s", category)
            continue

        pdf_files = list(cat_dir.glob("*.pdf"))
        log.info("Category %-25s → %d PDFs", category, len(pdf_files))

        for pdf_file in pdf_files:
            text = extract_text_from_pdf(str(pdf_file))
            records.append({
                "file_path": str(pdf_file.relative_to(data_path.parent)),
                "category":  category,
                "raw_text":  text,
            })

    df = pd.DataFrame(records, columns=["file_path", "category", "raw_text"])
    log.info("Dataset built: %d rows, %d categories", len(df), df["category"].nunique())
    return df


if __name__ == "__main__":
    # Quick smoke-test: extract one PDF per category and print counts
    df = build_dataset()
    print(df.groupby("category").size().sort_values(ascending=False))
