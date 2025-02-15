import pandas as pd
import os
import requests
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Create data directory if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Dataset URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms-spam-collection/spam.csv"
data_path = "data/spam.csv"

# Download dataset if not exists
if not os.path.exists(data_path):
    print("Downloading dataset...")
    response = requests.get(url)
    with open(data_path, "wb") as file:
        file.write(response.content)
    print("Dataset saved to", data_path)
else:
    print("Dataset already exists.")

# Load dataset
data = pd.read_csv(
    data_path,
    encoding="latin-1",
    usecols=[0, 1],
    names=["label", "message"],
    skiprows=1,
)

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# **Apply TF-IDF before SMOTE**
vectorizer = TfidfVectorizer(max_features=5000)  # Use only top 5000 words to reduce memory usage
X_tfidf = vectorizer.fit_transform(data["message"])  # Convert text to TF-IDF matrix
y = data["label"]

# **Apply SMOTE only if necessary**
if sum(y == 0) > sum(y == 1):  # Apply only if dataset is imbalanced
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    print("✅ Applied SMOTE. Data distribution:", Counter(y_resampled))
else:
    X_resampled, y_resampled = X_tfidf, y  # No SMOTE needed if already balanced

# Save processed dataset (Use sparse matrix format to save memory)
joblib.dump((X_resampled, y_resampled), "data/spam_processed_tfidf.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print(f"✅ Processed dataset saved successfully!")
