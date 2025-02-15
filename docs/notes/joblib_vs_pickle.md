# Joblib vs Pickle: Model Storage Benchmark

## **Overview**
This section compares the performance of **joblib** and **pickle** for saving and loading our trained spam classification model.

## **Test Script**
We conducted a benchmark test to compare **pickle** and **joblib** in terms of:
- **Saving time** (how fast the model is stored)
- **Loading time** (how fast the model is retrieved)
- **File size** (compressed vs uncompressed storage)

### **Test Script (`test_joblib_vs_pickle.py`)**
```python
import pickle
import joblib
import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import os

# Load dataset
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'talk.politics.guns'])
X, y = data.data, data.target

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train a Naïve Bayes model
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
pickle_size = os.path.getsize("model_pickle.pkl") / 1024  # KB
joblib_size = os.path.getsize("model_joblib.joblib") / 1024  # KB

# Print Results
print(f"Pickle Save Time: {pickle_time:.4f} sec")
print(f"Joblib Save Time: {joblib_time:.4f} sec")
print(f"Pickle Load Time: {pickle_load_time:.4f} sec")
print(f"Joblib Load Time: {joblib_load_time:.4f} sec")
print(f"Pickle File Size: {pickle_size:.2f} KB")
print(f"Joblib File Size: {joblib_size:.2f} KB")
```

## **Test Results (Dataset Size Considered)**
With the dataset we used, the test produced the following results:

| **Metric**       | **Pickle** | **Joblib** | **Which is Better?** |
|------------------|-----------|-----------|----------------|
| **Save Time**    | 0.0013 sec | **0.0006 sec** | ✅ **Joblib is faster** |
| **Load Time**    | **0.0001 sec** | 0.0004 sec | ✅ **Pickle is slightly faster** |
| **File Size**    | **836.95 KB** | 837.12 KB | ⚖️ **Almost identical** |

## **Conclusion**
✅ **Joblib is the better choice** for saving large ML models because:
- It **saves models almost 2x faster** than pickle.
- It is optimized for **NumPy arrays** (used in ML models).
- **File sizes were similar**, so joblib's compression did not make a significant difference in this case.

🔹 **Final Decision:** We chose **Joblib** as our storage method for the spam classifier.

