import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# ---------------- NLTK SAFE SETUP ----------------
def setup_nltk():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            if r == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{r}")
        except LookupError:
            nltk.download(r)

setup_nltk()

# ---------------- LOAD FILES ----------------
model = load_model("mental_health_model.h5")

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------- NLP TOOLS ----------------
stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text, preserve_line=True)
    tokens = [t for t in tokens if t not in stop]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    return " ".join(tokens)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Mental Health NLP Model", layout="centered")

st.title("🧠 Mental Health Detection System")
st.write("AI-powered mental health state classification from text")

user_text = st.text_area("✍️ Enter your text:")

if st.button("🔍 Predict"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Clean
        cleaned = clean_text(user_text)

        # Vectorize
        vec = tfidf.transform([cleaned]).toarray()

        # Predict
        preds = model.predict(vec, verbose=0)
        class_index = np.argmax(preds, axis=1)[0]
        label = le.inverse_transform([class_index])[0]
        confidence = float(np.max(preds))

        # Output
        st.success(f"🧾 Predicted Mental State: **{label}**")
        st.info(f"📊 Confidence Score: **{confidence:.4f}**")

        # Probabilities
        st.subheader("📈 Class Probabilities")
        for i, prob in enumerate(preds[0]):
            st.write(f"**{le.inverse_transform([i])[0]}** : {prob:.4f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed for Mental Health NLP Classification | Deep Learning + TF-IDF + Streamlit")
