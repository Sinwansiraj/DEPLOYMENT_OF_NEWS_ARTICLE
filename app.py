"""
Streamlit frontend for News Article Categorization.
Provides a clean UI for submitting articles and displaying predictions.

Run:
    streamlit run app.py --server.port 8501
"""

import logging
import os
import sys

import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.inference import load_model, predict
from src.db_utils import init_db_pool, create_tables, log_prediction, fetch_recent_predictions
from config import LABEL_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Categorizer",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { max-width: 860px; }
    .category-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .score-bar-label { font-size: 0.9rem; color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORY_COLORS = {
    "World": "#4CAF50",
    "Sports": "#2196F3",
    "Business": "#FF9800",
    "Science/Technology": "#9C27B0",
}


@st.cache_resource(show_spinner=False)
def initialize_services():
    """
    Load model and initialize database pool (cached across sessions).

    Returns:
        Tuple of (model_ok: bool, db_ok: bool).
    """
    model_ok = False
    db_ok = False

    try:
        load_model()
        model_ok = True
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Model load failed: %s", exc)

    try:
        init_db_pool()
        create_tables()
        db_ok = True
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.warning("DB init skipped (running without logging): %s", exc)

    return model_ok, db_ok


def render_prediction(result: dict) -> None:
    """
    Render the prediction result card.

    Args:
        result: Output dict from inference.predict().
    """
    label = result["predicted_label"]
    confidence = result["confidence"]
    color = CATEGORY_COLORS.get(label, "#607D8B")

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<span class="category-badge" style="background-color:{color};color:white;">'
            f"📌 {label}</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.markdown("**Confidence Breakdown**")
    for item in result["all_scores"]:
        bar_color = CATEGORY_COLORS.get(item["label"], "#607D8B")
        st.markdown(
            f'<span class="score-bar-label">{item["label"]}</span>',
            unsafe_allow_html=True,
        )
        st.progress(item["score"], text=f"{item['score'] * 100:.1f}%")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=60)
    st.title("About")
    st.markdown(
        """
        **News Article Categorizer**
        - Model: DistilBERT
        - Dataset: AG News
        - Categories:
        """
    )
    for cat, color in CATEGORY_COLORS.items():
        st.markdown(
            f'<span style="color:{color}">● {cat}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    show_history = st.checkbox("Show Recent Predictions", value=False)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📰 News Article Categorizer")
st.caption("Paste a news article below. The model will classify it into World, Sports, Business, or Science/Technology.")

# Initialize on first load
with st.spinner("⚙️ Loading model (first run may take ~30 seconds)..."):
    model_ready, db_ready = initialize_services()

if not model_ready:
    st.error(
        "❌ Model failed to load. Check logs or verify the model path / S3 configuration.",
        icon="🚨",
    )
    st.stop()

if not db_ready:
    st.warning(
        "⚠️ Database not connected — predictions will not be logged.",
        icon="⚠️",
    )

# Input form
article_text = st.text_area(
    label="Article Text",
    placeholder=(
        "e.g. Apple unveiled its latest iPhone lineup at a special event in Cupertino "
        "on Tuesday, featuring improved cameras and a new A18 chip..."
    ),
    height=220,
    help="Paste a news headline + body text for best results.",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=False):
        st.rerun()

if classify_btn:
    if not article_text.strip():
        st.warning("Please enter some article text before classifying.", icon="✏️")
    else:
        with st.spinner("Analyzing article..."):
            try:
                result = predict(article_text)
                render_prediction(result)

                if db_ready:
                    log_prediction(
                        input_text=article_text,
                        predicted_label=result["predicted_label"],
                        confidence=result["confidence"],
                        all_scores=result["all_scores"],
                    )

            except ValueError as exc:
                st.error(f"Input error: {exc}", icon="✏️")
            except RuntimeError as exc:
                st.error(f"Model error: {exc}", icon="🤖")
            except Exception as exc:
                logger.exception("Unexpected error during prediction.")
                st.error(
                    "An unexpected error occurred. Please try again.",
                    icon="🚨",
                )

# Recent predictions history
if show_history and db_ready:
    st.markdown("---")
    st.subheader("🕒 Recent Predictions")
    records = fetch_recent_predictions(limit=20)
    if records:
        for rec in records:
            with st.expander(
                f"[{rec['created_at'].strftime('%Y-%m-%d %H:%M')}] "
                f"{rec['predicted_label']} ({rec['confidence']*100:.1f}%)"
            ):
                st.text(rec["input_text"][:300] + "..." if len(rec["input_text"]) > 300 else rec["input_text"])
    else:
        st.info("No predictions logged yet.")