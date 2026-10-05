"""
Resume-aware text preprocessing.

Design principles
─────────────────
• Preserve technical tokens: C++, C#, .NET, SQL, AWS, TensorFlow, Java, NLP …
• Do NOT strip punctuation blindly — # + . are meaningful in tech tokens.
• Do NOT lemmatise or stem at this stage (done later inside the TF-IDF pipeline
  if at all, so the raw clean_text stays reusable for deep-learning too).
• Do NOT remove stopwords here — let TF-IDF idf-weighting handle common words.
• Keep preprocessing deterministic and fast (pure regex, no NLTK tokeniser).

Public API
──────────
  preprocess_text(text: str) -> str          main function
  preprocess_dataframe(df)   -> df           adds 'clean_text' column

Backward-compat aliases (used by eda.py / build_dataset.py)
──────────────────────────────────────────────────────────────
  clean_text(text)    → preprocess_text(text)
  preprocess(text)    → preprocess_text(text)
  preprocess_series   → wraps preprocess_text
"""

import re
import unicodedata
import logging

import pandas as pd

log = logging.getLogger(__name__)

# ── Compiled regex patterns ────────────────────────────────────────────────────

# HTML / XML tags
_HTML_RE    = re.compile(r"<[^>]+>")
# HTML entities  &amp;  &#39;  etc.
_ENTITY_RE  = re.compile(r"&(?:#\d+|[a-z]+);", re.I)
# URLs  (http / https / www)
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.I)
# Email addresses
_EMAIL_RE   = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-z]{2,}", re.I)
# Phone numbers  — various formats
_PHONE_RE   = re.compile(
    r"(\+?\d{1,3}[\s\-.]?)?"           # optional country code
    r"(\(?\d{2,4}\)?[\s\-.]?)"         # area code
    r"\d{3,4}[\s\-.]?\d{3,5}"          # subscriber number
)
# Normalise line-breaks and excessive whitespace
_NEWLINE_RE = re.compile(r"[\r\n\t\f\v]+")
_SPACE_RE   = re.compile(r" {2,}")

# Characters that carry NO meaning in resumes and are not part of tech tokens.
# We keep:  letters, digits, spaces, and  + # . - _ / @ (for tech tokens & paths)
# We remove: stray punctuation that is just noise.
_NOISE_RE   = re.compile(r"[^\w\s+#.\-_/@]")

# Collapse repeated punctuation left after cleaning  e.g. "... ..."
_PUNCT_REPEAT = re.compile(r"([+#.\-_/@])\1{2,}")


def preprocess_text(text: str) -> str:
    """
    Resume-aware normalisation pipeline.

    Steps
    ─────
    1. Guard against NaN / non-string input.
    2. Normalise Unicode (NFKC) — handles accented chars, ligatures, etc.
    3. Strip HTML tags and entities.
    4. Replace URLs, emails, phone numbers.
    5. Lowercase.
    6. Normalise whitespace (newlines → space, collapse multiple spaces).
    7. Remove characters that are pure noise while keeping tech-token chars.
    8. Final whitespace collapse and strip.
    """
    # 1. Guard
    if not isinstance(text, str) or not text.strip():
        return ""

    # 2. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 3. HTML / markup
    text = _HTML_RE.sub(" ", text)
    text = _ENTITY_RE.sub(" ", text)

    # 4. PII / noise patterns — replace with single space (don't collapse context)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Newlines and tabs → single space
    text = _NEWLINE_RE.sub(" ", text)

    # 7. Remove noisy punctuation; preserve +, #, ., -, _, /, @
    text = _NOISE_RE.sub(" ", text)

    # 8. Remove repeated punctuation  "..." → "."
    text = _PUNCT_REPEAT.sub(r"\1", text)

    # 9. Final whitespace collapse
    text = _SPACE_RE.sub(" ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocess_text to the 'text' column.
    Adds a 'clean_text' column to a copy of df.
    Does NOT modify the original DataFrame.
    """
    df = df.copy()
    df["clean_text"] = df["text"].fillna("").apply(preprocess_text)
    return df


# ── Backward-compatibility aliases ────────────────────────────────────────────
# eda.py and build_dataset.py import clean_text / preprocess / preprocess_series

def clean_text(text: str) -> str:          # noqa: F811
    return preprocess_text(text)


def preprocess(text: str) -> str:
    return preprocess_text(text)


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(preprocess_text)


# ── NLTK helpers kept for any future use ──────────────────────────────────────

def download_nltk_resources():
    """Download NLTK data required by downstream scripts (e.g. EDA n-gram charts)."""
    import nltk
    resources = [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
        ("corpora/omw-1.4",      "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


if __name__ == "__main__":
    samples = [
        "John Doe | john.doe@email.com | +1-800-555-0199",
        "Skills: Python, C++, C#, .NET, SQL, AWS, TensorFlow, NLP",
        "B.Sc. Computer Science — GPA 3.8/4.0",
        "Senior Software Engineer @ Google (2019–2023)",
        "<p>Experience in <b>machine learning</b> &amp; data science</p>",
        "",
        None,
    ]
    for s in samples:
        print(repr(preprocess_text(s)))
