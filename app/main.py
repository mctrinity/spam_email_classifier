import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load processed dataset
data_path = "data/spam_processed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Processed dataset not found. Run 'scripts/preprocess.py' first."
    )

data = pd.read_csv(data_path)
X = data["message"]
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Save model and vectorizer
model_path = "models/spam_classifier.pkl"
vectorizer_path = "models/vectorizer.pkl"

with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"Model trained and saved as '{model_path}'")
