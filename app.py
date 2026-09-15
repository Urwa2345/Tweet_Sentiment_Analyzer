# =========================================================
# 1. IMPORTS & PAGE CONFIG
# =========================================================
import streamlit as st
import joblib
import pickle
import json
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. CUSTOM CSS — Modern Glassmorphism Theme
# =========================================================
st.markdown("""
<style>
    /* Global Container */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #172554 0%, #0B1220 70%);
        color: #F8FAFC;
    }

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Global Typography */
    h1, h2, h3, p, span, label, div {
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        color: #94A3B8 !important;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Input & Containers */
    .stTextArea textarea {
        background-color: rgba(16, 28, 51, 0.7) !important;
        color: #F8FAFC !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* Model Selection Radio */
    div[role="radiogroup"] {
        background: rgba(16, 28, 51, 0.6);
        padding: 8px 14px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-around;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #2563EB 0%, #0284C7 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.4rem;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #0369A1 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }

    /* Result Card */
    .result-card {
        background: rgba(16, 28, 51, 0.7);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
    }
    .conf-bar-bg {
        background: #1E293B;
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        margin-bottom: 6px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL ARTIFACTS (cached — loads once per session)
# =========================================================
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    svm_model = joblib.load("svm_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    rnn_model = load_model("rnn_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)
    return tfidf, svm_model, label_encoder, rnn_model, tokenizer, config


tfidf, svm_model, label_encoder, rnn_model, tokenizer, config = load_artifacts()
MAX_LEN = config["MAX_LEN"]


# =========================================================
# 4. TEXT PREPROCESSING (must match training exactly)
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================================================
# 5. SENTIMENT DISPLAY CONFIG
# =========================================================
SENTIMENT_STYLE = {
    "Positive": {"color": "#22C55E", "emoji": "😊"},
    "Neutral": {"color": "#94A3B8", "emoji": "😐"},
    "Negative": {"color": "#EF4444", "emoji": "😠"},
}

# =========================================================
# 6. MAIN UI — Header & Model Selection
# =========================================================
st.markdown("""
<div class="header-container">
    <div class="app-title">🐦 Tweet Sentiment Analyzer</div>
    <div class="app-subtitle">Classify tweet sentiment with Classical ML or Deep Learning</div>
</div>
""", unsafe_allow_html=True)

# Centered horizontal model switcher
model_choice = st.radio(
    "Choose prediction model:",
    ["ML Model (SVM)", "Deep Learning Model (BiRNN + GloVe)"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

tweet_input = st.text_area(
    "Enter a tweet to analyze:",
    height=110,
    placeholder="e.g. Just tried the new update and it's actually really smooth!",
    label_visibility="collapsed"
)

analyze_clicked = st.button("✨ Analyze Sentiment")

# =========================================================
# 7. PREDICTION LOGIC
# =========================================================
if analyze_clicked:
    if not tweet_input.strip():
        st.warning("Please enter a tweet first.")
    else:
        cleaned = clean_text(tweet_input)

        if cleaned == "":
            st.warning("Input has no usable text after cleaning (e.g. only links/mentions). Try different text.")
        else:
            if model_choice == "ML Model (SVM)":
                vec = tfidf.transform([cleaned])
                pred_label = svm_model.predict(vec)[0]
                sentiment = label_encoder.inverse_transform([pred_label])[0]

                # LinearSVC decision function confidence proxy
                scores = svm_model.decision_function(vec)[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()

            else:
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
                probs = rnn_model.predict(padded, verbose=0)[0]
                pred_label = np.argmax(probs)
                sentiment = label_encoder.inverse_transform([pred_label])[0]

            style = SENTIMENT_STYLE.get(sentiment, {"color": "#38BDF8", "emoji": "🔍"})

            # ---- Result Card ----
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:1.35rem; font-weight:700; margin-bottom: 4px;">
                    {style['emoji']} Sentiment: 
                    <span style="color:{style['color']};">{sentiment}</span>
                </div>
                <div style="color:#64748B; font-size:0.85rem; margin-bottom: 1.2rem;">
                    Engine: <span style="color:#94A3B8;">{model_choice}</span>
                </div>
                <div style="font-weight:600; font-size:0.9rem; margin-bottom:0.6rem; color:#CBD5E1;">
                    Confidence Breakdown
                </div>
            """, unsafe_allow_html=True)

            # ---- Confidence Bars ----
            classes = label_encoder.classes_
            for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1]):
                cls_style = SENTIMENT_STYLE.get(cls, {"color": "#38BDF8"})
                pct = prob * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
                    <span style="color:#CBD5E1;">{cls}</span>
                    <span style="font-weight:600; color:#E2E8F0;">{pct:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%; background:{cls_style['color']};"></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
