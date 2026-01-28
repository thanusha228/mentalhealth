import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# ---------------- LOAD FILES ----------------
model = load_model("mental_health_model.h5")

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------- NLP SETUP ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    return " ".join(tokens)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Mental Health NLP Model", layout="centered")

st.title("🧠 Mental Health Detection System")
st.write("AI model for detecting mental health state from text")

user_text = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean
        cleaned = clean_text(user_text)

        # Vectorize
        vec = tfidf.transform([cleaned]).toarray()

        # Predict
        preds = model.predict(vec)
        class_index = np.argmax(preds, axis=1)[0]
        label = le.inverse_transform([class_index])[0]
        confidence = np.max(preds)

        # Output
        st.success(f"🧾 Predicted Label: **{label}**")
        st.info(f"📊 Confidence Score: **{confidence:.4f}**")

        # Probability display
        st.subheader("Class Probabilities")
        for i, prob in enumerate(preds[0]):
            st.write(f"{le.inverse_transform([i])[0]} : {prob:.4f}")
