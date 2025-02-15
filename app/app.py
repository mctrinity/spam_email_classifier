import joblib
import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Deep Learning Model & Tokenizer
model = tf.keras.models.load_model("models/spam_classifier_lstm.h5")
tokenizer = joblib.load("models/tokenizer.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

max_length = 100  # Must match training


# Streamlit UI
def main():
    st.title("Spam Email Classifier (Deep Learning)")
    st.write("Enter an email message to check if it is spam or not.")

    user_input = st.text_area("Enter email message:")
    if st.button("Check Spam"):
        if user_input:
            # Preprocess Input
            input_sequence = tokenizer.texts_to_sequences([user_input])
            input_padded = pad_sequences(
                input_sequence, maxlen=max_length, padding="post"
            )

            # Predict
            prediction = model.predict(input_padded)[0][0]
            result = "Spam" if prediction > 0.5 else "Not Spam"
            st.subheader(f"Result: {result} (Confidence: {prediction:.2%})")
        else:
            st.warning("Please enter an email message.")


if __name__ == "__main__":
    main()
