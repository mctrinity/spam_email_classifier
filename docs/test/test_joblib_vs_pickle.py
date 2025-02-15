import pickle
import joblib
import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load dataset
data = fetch_20newsgroups(
    subset="train", categories=["sci.space", "talk.politics.guns"]
)
X, y = data.data, data.target

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train a simple Naïve Bayes model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Measure Pickle Save Time
start = time.time()
with open("model_pickle.pkl", "wb") as f:
    pickle.dump(model, f)
pickle_time = time.time() - start

# Measure Joblib Save Time
start = time.time()
joblib.dump(model, "model_joblib.joblib")
joblib_time = time.time() - start

# Measure Pickle Load Time
start = time.time()
with open("model_pickle.pkl", "rb") as f:
    model_pickle = pickle.load(f)
pickle_load_time = time.time() - start

# Measure Joblib Load Time
start = time.time()
model_joblib = joblib.load("model_joblib.joblib")
joblib_load_time = time.time() - start

# Measure File Sizes
import os

pickle_size = os.path.getsize("model_pickle.pkl") / 1024  # KB
joblib_size = os.path.getsize("model_joblib.joblib") / 1024  # KB

# Print Results
print(f"Pickle Save Time: {pickle_time:.4f} sec")
print(f"Joblib Save Time: {joblib_time:.4f} sec")
print(f"Pickle Load Time: {pickle_load_time:.4f} sec")
print(f"Joblib Load Time: {joblib_load_time:.4f} sec")
print(f"Pickle File Size: {pickle_size:.2f} KB")
print(f"Joblib File Size: {joblib_size:.2f} KB")
