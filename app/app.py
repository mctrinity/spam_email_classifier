import joblib
import tensorflow as tf
import numpy as np
import streamlit as st

st.set_page_config(page_title="Spam Classifier", layout="centered")

# Load the trained model and vectorizer
model = tf.keras.models.load_model("models/spam_classifier_tfidf.keras", compile=False)
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# Streamlit UI
def main():
    st.title("📩 Spam Email Classifier (TF-IDF + Deep Learning)")
    st.markdown("**Enter an email message below to check if it's Spam or Not Spam.**")

    # 🧹 Clear cache section
    with st.expander("⚙️ Troubleshooting"):
        st.markdown(
            "If you encounter an error, try refreshing the browser and click below:"
        )
        if st.button("🧹 Clear Cache"):
            st.cache_resource.clear()
            st.success("✅ Cache cleared! Please refresh the browser (F5 or Cmd+R).")

    st.markdown("---")

    # Sample message selector
    example = st.selectbox(
        "💡 Try a sample email:", ["", "Spam Example", "Ham Example"]
    )
    user_input = ""
    if example == "Spam Example":
        user_input = (
            "Subject: 🎉 Congratulations! You’re a Lucky Winner!\n"
            'From: "Rewards Department" <rewards@freelottery.com>\n\n'
            "Dear Valued Customer,\n\n"
            "You have won a $500 Amazon Gift Card!\n"
            "Click the link below to claim: http://fraudulent-link.com\n\n"
            "Hurry! This offer is only valid for 24 hours."
        )
    elif example == "Ham Example":
        user_input = "Hey, let's meet for coffee tomorrow at 10 AM."

    user_input = st.text_area(
        "✉️ Enter or edit your email message:", value=user_input, height=200
    )

    threshold = st.slider("Spam Sensitivity Threshold", 0.0, 1.0, 0.15, 0.01)

    if st.button("🔍 Check Spam"):
        if user_input:
            # Transform input using TF-IDF vectorizer
            input_vectorized = vectorizer.transform([user_input]).toarray()

            # Predict
            prediction = model.predict(input_vectorized)[0][0]
            confidence = prediction * 100
            is_spam = prediction > threshold

            # Display result
            result = "🚨 Spam" if is_spam else "✅ Not Spam"
            result_color = "red" if is_spam else "green"

            st.markdown(
                f"<div style='color: {result_color}; font-size: 24px; font-weight: bold;'>Result: {result}</div>",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{max(confidence, 0.01):.2f}%")

            # st.progress(min(prediction, 1.0))
            st.progress(float(min(prediction, 1.0)))

        else:
            st.warning("⚠️ Please enter an email message.")

    st.markdown("---")
    st.markdown(
        "📧 Created by [Scidyllics](https://github.com/mctrinity) · Powered by LSTM + TF-IDF"
    )


if __name__ == "__main__":
    main()
