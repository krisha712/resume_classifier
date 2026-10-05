"""
Resume Classification — Streamlit Demo
Run from project root:  streamlit run app.py
"""

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Make src/ importable from the project root ───────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import streamlit as st
import joblib

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered",
)

# ── Paths (inline to avoid importing config which resolves ROOT_DIR differently)
_MODELS = os.path.join(os.path.dirname(__file__), "models")
DL_MODEL_PATH    = os.path.join(_MODELS, "dl_model.keras")
DL_TOK_PATH      = os.path.join(_MODELS, "dl_tokenizer.pkl")
DL_LE_PATH       = os.path.join(_MODELS, "dl_label_encoder.pkl")
CLS_MODEL_PATH   = os.path.join(_MODELS, "classical_model.pkl")
CLS_LE_PATH      = os.path.join(_MODELS, "label_encoder.pkl")

MAX_SEQ_LEN = 400   # must match train_dl.py


# ══════════════════════════════════════════════════════════════════════════════
# Model loading  (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_dl_model():
    """Load BiLSTM model + tokenizer + label encoder. Returns (model, tok, le) or None."""
    required = [DL_MODEL_PATH, DL_TOK_PATH, DL_LE_PATH]
    if not all(os.path.exists(p) for p in required):
        return None
    try:
        import tensorflow as tf           # noqa: F401
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model   # type: ignore
        model = load_model(DL_MODEL_PATH)
        tok   = joblib.load(DL_TOK_PATH)
        le    = joblib.load(DL_LE_PATH)
        return model, tok, le
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading model…")
def load_classical_model():
    """Load classical sklearn Pipeline + label encoder."""
    if not os.path.exists(CLS_MODEL_PATH):
        return None
    try:
        pipeline = joblib.load(CLS_MODEL_PATH)
        le       = joblib.load(CLS_LE_PATH)
        return pipeline, le
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_dl(clean_text: str, model, tok, le):
    """BiLSTM inference → (top_class, top3 [(label, prob)], proba_array)"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences   # type: ignore
    seqs  = tok.texts_to_sequences([clean_text])
    X     = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    proba = model.predict(X, verbose=0)[0]          # shape (n_classes,)
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [(le.classes_[i], float(proba[i])) for i in top3_idx]
    return le.classes_[proba.argmax()], top3, proba


def predict_classical(clean_text: str, pipeline, le):
    """LinearSVC inference → (top_class, top3 [(label, score)], scores_array)"""
    scores  = pipeline.decision_function([clean_text])[0]   # (n_classes,)
    top3_idx = scores.argsort()[-3:][::-1]
    top3 = [(le.classes_[i], float(scores[i])) for i in top3_idx]
    return le.classes_[scores.argmax()], top3, scores


# ══════════════════════════════════════════════════════════════════════════════
# PDF extraction (wraps src/pdf_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_bytes(pdf_bytes: bytes) -> str:
    """Write bytes to a temp file, extract with PyMuPDF, return raw text."""
    try:
        import fitz
    except ImportError:
        st.error("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        doc   = fitz.open(tmp_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        os.unlink(tmp_path)
        return "\n".join(pages).strip()
    except Exception as exc:
        st.error(f"PDF extraction failed: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

def render_header(model_name: str):
    st.title("📄 Resume Classifier")
    st.caption(f"Model: **{model_name}**")
    st.divider()


def render_how_it_works():
    with st.expander("How it works"):
        st.markdown(
            """
```
PDF Resume
    ↓
Text Extraction       (PyMuPDF)
    ↓
Text Preprocessing    (Unicode normalisation, URL/email removal)
    ↓
Tokenization          (Keras Tokenizer / TF-IDF)
    ↓
Word2Vec Embeddings   (trained on resume corpus)
    ↓
Bidirectional LSTM    (64 units)
    ↓
Resume Category
```
            """
        )
        st.caption(
            "The model was selected after comparing classical TF-IDF-based models "
            "with a Word2Vec + BiLSTM deep-learning model."
        )


def render_performance(model_name: str, acc: str, f1_macro: str, f1_weighted: str):
    with st.expander("Model performance (held-out test set)"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy",     acc)
        col2.metric("Macro-F1",     f1_macro)
        col3.metric("Weighted-F1",  f1_weighted)
        st.caption(f"Model: {model_name} | 24 resume categories | ~2 500 resumes")


def render_prediction_dl(predicted: str, top3, word_count: int, preview: str):
    st.success(f"### Predicted Category:  {predicted}")
    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Top 3 predictions (softmax probability)**")
        for label, prob in top3:
            bar_label = f"{label}  ({prob*100:.1f} %)"
            st.progress(float(prob), text=bar_label)

    with col_b:
        st.markdown("**Resume info**")
        st.write(f"Word count: **{word_count}**")
        st.markdown("**Extracted text preview**")
        st.text(preview[:400])


def render_prediction_classical(predicted: str, top3, word_count: int, preview: str):
    st.success(f"### Predicted Category:  {predicted}")
    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Top 3 predictions (decision score — not a probability)**")
        # Normalise scores to [0,1] range for display only
        raw_scores = [s for _, s in top3]
        min_s, max_s = min(raw_scores), max(raw_scores)
        spread = max_s - min_s if max_s != min_s else 1.0
        for label, sc in top3:
            norm = (sc - min_s) / spread
            st.progress(float(norm), text=f"{label}  (score: {sc:.3f})")

    with col_b:
        st.markdown("**Resume info**")
        st.write(f"Word count: **{word_count}**")
        st.markdown("**Extracted text preview**")
        st.text(preview[:400])


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Try DL model first; fall back to classical
    dl_assets  = load_dl_model()
    use_dl     = dl_assets is not None

    if use_dl:
        model, tok, le = dl_assets
        model_name  = "Word2Vec + Bidirectional LSTM"
        acc_str     = "69.95 %"
        f1_mac_str  = "65.03 %"
        f1_wt_str   = "69.81 %"
    else:
        cls_assets = load_classical_model()
        if cls_assets is None:
            st.error(
                "No trained model found. "
                "Run `python src/train_dl.py` and/or `python src/train_ml.py` first."
            )
            return
        pipeline, le = cls_assets
        model_name  = "TF-IDF + Linear SVM"
        acc_str     = "68.35 %"
        f1_mac_str  = "63.93 %"
        f1_wt_str   = "67.24 %"

    render_header(model_name)

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload a resume PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload a PDF resume above to get a prediction.")
        render_how_it_works()
        render_performance(model_name, acc_str, f1_mac_str, f1_wt_str)
        return

    # ── Extract text ──────────────────────────────────────────────────────────
    with st.spinner("Extracting text from PDF…"):
        raw_text = extract_pdf_bytes(uploaded.read())

    if not raw_text.strip():
        st.error(
            "Could not extract text from this PDF. "
            "The file may be image-only, password-protected, or corrupted."
        )
        render_how_it_works()
        render_performance(model_name, acc_str, f1_mac_str, f1_wt_str)
        return

    # ── Preprocess (identical to training pipeline) ───────────────────────────
    from preprocessing import preprocess_text
    clean = preprocess_text(raw_text)

    if not clean.strip():
        st.error("Text was extracted but became empty after preprocessing.")
        render_how_it_works()
        render_performance(model_name, acc_str, f1_mac_str, f1_wt_str)
        return

    word_count = len(raw_text.split())
    preview    = " ".join(raw_text.split()[:80])   # first 80 words

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        if use_dl:
            predicted, top3, _ = predict_dl(clean, model, tok, le)
            render_prediction_dl(predicted, top3, word_count, preview)
        else:
            predicted, top3, _ = predict_classical(clean, pipeline, le)
            render_prediction_classical(predicted, top3, word_count, preview)

    render_how_it_works()
    render_performance(model_name, acc_str, f1_mac_str, f1_wt_str)


if __name__ == "__main__":
    main()
