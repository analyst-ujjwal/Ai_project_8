import streamlit as st
from model.predict_model import SpamClassifier
from model.explain_groq import explain_prediction

# Title and description
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")
st.title("📧 Spam Email Classifier — Logistic Regression + BERT + Groq")

st.markdown("""
Upload or type an email message to check if it's spam or legitimate (ham).  
The app uses **BERT embeddings + Logistic Regression** for classification, and **Groq LLaMA** for natural-language explanations.
""")

classifier = SpamClassifier()

# User input
email_text = st.text_area("✉️ Paste or type your email text here:", height=200)

if st.button("🔍 Analyze"):
    if not email_text.strip():
        st.warning("Please enter an email text.")
    else:
        label = classifier.predict(email_text)
        st.subheader(f"**Result:** {label}")

        with st.spinner("💡 Generating AI explanation..."):
            explanation = explain_prediction(email_text, label)
        st.success("Explanation:")
        st.write(explanation)
