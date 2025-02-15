import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load processed dataset
data_path = "data/spam_processed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Processed dataset not found. Run 'scripts/preprocess.py' first."
    )

data = pd.read_csv(data_path)
X = data["message"]
y = data["label"]

# Convert labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenization & Text Preprocessing
max_words = 5000  # Vocabulary size
max_length = 100  # Max email length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding="post")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42
)

# Define LSTM Model
model = Sequential(
    [
        Embedding(input_dim=max_words, output_dim=64, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),  # Binary Classification
    ]
)

# Compile Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
epochs = 5  # Change as needed
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test)
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save Model & Tokenizer
model.save("models/spam_classifier_lstm.h5")
joblib.dump(tokenizer, "models/tokenizer.joblib")
joblib.dump(label_encoder, "models/label_encoder.joblib")

print("✅ Deep Learning Model and Tokenizer saved successfully!")
