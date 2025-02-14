import pickle
import os
import streamlit as st

# Load the trained model and vectorizer
model_path = "models/spam_classifier.pkl"
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model files not found. Please train the model first.")
    st.stop()

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Streamlit UI
def main():
    st.title("Spam Email Classifier")
    st.write("Enter an email message to check if it is spam or not.")

    user_input = st.text_area("Enter email message:")
    if st.button("Check Spam"):
        if user_input:
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            result = "Spam" if prediction == 1 else "Not Spam"
            st.subheader(f"Result: {result}")
        else:
            st.warning("Please enter an email message.")


if __name__ == "__main__":
    main()
