"""
Stage 3: Preprocessing + Stratified Data Split

Usage:
    python src/split_data.py

Reads:   data/resume_dataset.csv
Writes:  data/train.csv
         data/validation.csv
         data/test.csv

Strategy
────────
• Apply resume-aware preprocessing → 'clean_text' column.
• Remove rows whose clean_text is empty (report how many).
• Detect exact duplicate clean_text values.
  - Group duplicates so all copies land in the same split (no leakage).
  - Report duplicate group count.
• Stratified split at the GROUP level:
    70 % train | 15 % validation | 15 % test
• Fixed random_state=42 throughout for reproducibility.
• No TF-IDF / tokenizer / scaler fitted here — raw text only.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from config import DATASET_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from preprocessing import preprocess_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Columns to keep in the output CSVs
OUTPUT_COLS = ["file_name", "file_path", "category", "text",
               "clean_text", "character_count", "word_count"]


# ── Step 1: load ──────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_CSV):
        log.error("Dataset not found: %s  — run build_dataset.py first.", DATASET_CSV)
        sys.exit(1)
    df = pd.read_csv(DATASET_CSV)
    log.info("Loaded %d rows from %s", len(df), DATASET_CSV)
    return df


# ── Step 2: preprocess ────────────────────────────────────────────────────────

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Applying resume-aware preprocessing …")
    df = preprocess_dataframe(df)          # adds 'clean_text'
    before = len(df)

    empty_mask = df["clean_text"].str.strip() == ""
    n_empty = empty_mask.sum()
    if n_empty:
        log.warning("Removing %d rows with empty clean_text.", n_empty)
        df = df[~empty_mask].reset_index(drop=True)

    log.info("Rows after removing empty clean_text: %d  (removed %d)", len(df), n_empty)
    return df, n_empty


# ── Step 3: duplicate handling ────────────────────────────────────────────────

def assign_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a group_id to every row.
    Rows with identical clean_text share the same group_id.
    Unique rows get their own singleton group_id.
    This ensures all copies of a duplicate stay in the same split.
    """
    # Map each unique clean_text to a stable integer id
    unique_texts = df["clean_text"].unique()
    text_to_gid  = {t: i for i, t in enumerate(unique_texts)}
    df = df.copy()
    df["_group_id"] = df["clean_text"].map(text_to_gid)

    dup_groups = df.groupby("_group_id").filter(lambda x: len(x) > 1)
    n_dup_groups = dup_groups["_group_id"].nunique()
    n_dup_rows   = len(dup_groups)
    log.info("Duplicate groups: %d  (%d total rows involved)", n_dup_groups, n_dup_rows)

    return df, n_dup_groups, n_dup_rows


# ── Step 4: group-level stratified split ─────────────────────────────────────

def stratified_split(df: pd.DataFrame):
    """
    Split at the group level so duplicates never leak across splits.

    Returns train_df, val_df, test_df.
    """
    # One representative row per group → carry category label for stratification
    groups = (
        df.groupby("_group_id")
          .agg(category=("category", "first"))
          .reset_index()
    )

    # Categories with only 1 group cannot be stratified into 3 splits.
    # We put those singletons into train.
    cat_counts = groups["category"].value_counts()
    rare_cats  = cat_counts[cat_counts < 3].index.tolist()
    rare_mask  = groups["category"].isin(rare_cats)

    rare_groups  = groups[rare_mask]
    strat_groups = groups[~rare_mask]

    if len(rare_groups):
        log.warning(
            "Categories with < 3 groups (all assigned to train): %s",
            rare_cats
        )

    # ── First cut: split off test (15 %) ──────────────────────────────────────
    train_val_gids, test_gids = train_test_split(
        strat_groups["_group_id"].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strat_groups["category"].values,
    )

    # ── Second cut: split train_val into train / val ──────────────────────────
    # val is 15 % of total → 15/85 ≈ 0.1765 of the remaining train_val
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)

    train_val_df = strat_groups[strat_groups["_group_id"].isin(train_val_gids)]
    train_gids, val_gids = train_test_split(
        train_val_df["_group_id"].values,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=train_val_df["category"].values,
    )

    # Rare categories all go to train
    all_train_gids = set(train_gids) | set(rare_groups["_group_id"].values)

    # ── Expand groups back to rows ─────────────────────────────────────────────
    train_df = df[df["_group_id"].isin(all_train_gids)].copy()
    val_df   = df[df["_group_id"].isin(val_gids)].copy()
    test_df  = df[df["_group_id"].isin(test_gids)].copy()

    return train_df, val_df, test_df


# ── Step 5: verify & report ───────────────────────────────────────────────────

def verify_splits(train_df, val_df, test_df, df_total):
    all_cats = sorted(df_total["category"].unique())
    total    = len(train_df) + len(val_df) + len(test_df)

    sep = "─" * 60
    print(f"\n{sep}")
    print("  SPLIT SUMMARY")
    print(sep)
    print(f"  Total usable resumes : {total}")
    print(f"  Train                : {len(train_df):>5}  ({len(train_df)/total*100:.1f} %)")
    print(f"  Validation           : {len(val_df):>5}  ({len(val_df)/total*100:.1f} %)")
    print(f"  Test                 : {len(test_df):>5}  ({len(test_df)/total*100:.1f} %)")
    print(sep)

    # Per-category distribution table
    print(f"\n  {'Category':<30} {'Total':>6} {'Train':>7} {'Val':>6} {'Test':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
    missing_val  = []
    missing_test = []
    for cat in all_cats:
        n_total = (df_total["category"] == cat).sum()
        n_train = (train_df["category"] == cat).sum()
        n_val   = (val_df["category"]   == cat).sum()
        n_test  = (test_df["category"]  == cat).sum()
        print(f"  {cat:<30} {n_total:>6} {n_train:>7} {n_val:>6} {n_test:>6}")
        if n_val  == 0: missing_val.append(cat)
        if n_test == 0: missing_test.append(cat)

    print(sep)
    if missing_val:
        print(f"  ⚠  Categories missing from validation : {missing_val}")
    if missing_test:
        print(f"  ⚠  Categories missing from test       : {missing_test}")
    if not missing_val and not missing_test:
        print("  ✓  All categories represented in train / validation / test")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load
    df = load_dataset()

    # 2. Preprocess
    df, n_removed_empty = apply_preprocessing(df)

    # 3. Duplicate groups
    df, n_dup_groups, n_dup_rows = assign_duplicate_groups(df)

    print(f"\n  Removed empty rows    : {n_removed_empty}")
    print(f"  Duplicate groups      : {n_dup_groups}  ({n_dup_rows} rows)")
    print(f"  Kept together in split: YES (group-level stratification)\n")

    # 4. Split
    log.info("Splitting data (70 / 15 / 15) …")
    train_df, val_df, test_df = stratified_split(df)

    # Drop internal helper column
    for split in (train_df, val_df, test_df):
        split.drop(columns=["_group_id"], inplace=True, errors="ignore")

    # 5. Verify
    verify_splits(train_df, val_df, test_df, df)

    # 6. Save
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    train_df[OUTPUT_COLS].to_csv(TRAIN_CSV, index=False)
    val_df[OUTPUT_COLS].to_csv(VAL_CSV,     index=False)
    test_df[OUTPUT_COLS].to_csv(TEST_CSV,   index=False)

    log.info("Saved → %s", TRAIN_CSV)
    log.info("Saved → %s", VAL_CSV)
    log.info("Saved → %s", TEST_CSV)
    log.info("Stage 3 complete.")
