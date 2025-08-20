import os
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ----------------- Load Fine-Tuned Model -----------------
MODEL_PATH = os.getenv("C:/Users/sinwa/Desktop/news_article/model","/.model")
@st.cache_resource
def load_model():
    MODEL_PATH = "C:/Users/sinwa/desktop/news_article/model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs

# -----------------------------
# Category mapping
# -----------------------------
id2label = {
    0: "World 🌍",
    1: "Sports 🏅",
    2: "Business 💼",
    3: "Sci/Tech 🔬"
}

# -----------------------------
# Streamlit UI
# -----------------------------


# Title
st.markdown("<h1 style='text-align: center;'>📰 News Article Categorization App</h1>", unsafe_allow_html=True)
st.write("Enter a news headline or short description, and the model will classify it into one of the AG News categories.")

# Input text
user_input = st.text_area("📝 Enter your news Headline here:", "")

# Button
if st.button("Classify"):
    if user_input.strip():
        probs = predict(user_input)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

        category = id2label.get(pred_class, f"Unknown ({pred_class})")

        # Prediction box
        st.markdown(
            f"<div style='background-color:#1f4d3d; padding:10px; border-radius:8px;'>"
            f"<span style='font-size:18px; color:white;'><b>Prediction:</b> {category}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Confidence box
        st.markdown(
            f"<div style='background-color:#0e2f44; padding:10px; border-radius:8px; margin-top:10px;'>"
            f"<span style='font-size:18px; color:white;'><b>Confidence:</b> {confidence*100:.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter some text to classify.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:green;'>✅ Model fine-tuned on <b>AG News Dataset</b> | Deployed by <b>Sinwan_siraj</b></p>", unsafe_allow_html=True)

