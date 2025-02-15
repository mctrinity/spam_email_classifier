import joblib
import tensorflow as tf
import numpy as np
import streamlit as st

# Load the trained model and vectorizer
model = tf.keras.models.load_model("models/spam_classifier_tfidf.keras")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Streamlit UI
def main():
    st.title("📩 Spam Email Classifier (TF-IDF + Deep Learning)")
    st.markdown("**Enter an email message below to check if it's Spam or Not Spam.**")

    user_input = st.text_area("✉️ Enter email message:")

    if st.button("🔍 Check Spam"):
        if user_input:
            # Transform input using TF-IDF vectorizer
            input_vectorized = vectorizer.transform([user_input]).toarray()

            # Predict
            prediction = model.predict(input_vectorized)[0][0]
            print(f"DEBUG: Raw model output: {prediction}")
            threshold = 0.15  # More sensitive to spam
            result = "🚨 **Spam**" if prediction > threshold else "✅ **Not Spam**"

            # Display result
            st.subheader(f"Result: {result}")
            confidence = prediction * 100  # Convert to percentage
            st.write(f"Confidence: {max(confidence, 0.01):.2f}%")  # Ensure at least 0.01%


        else:
            st.warning("⚠️ Please enter an email message.")

if __name__ == "__main__":
    main()
