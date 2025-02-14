import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load processed dataset
data_path = "data/spam_processed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Processed dataset not found. Run 'scripts/preprocess.py' first."
    )

data = pd.read_csv(data_path)
X = data["message"]
y = data["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_path = "models/spam_classifier.pkl"
vectorizer_path = "models/vectorizer.pkl"

with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"Model trained and saved as '{model_path}'")
