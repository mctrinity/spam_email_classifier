import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# **Load the preprocessed dataset**
X, y = joblib.load("data/spam_processed_tfidf.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save Model
# Save Model in a compatible format
model.save("models/spam_classifier_tfidf.keras", save_format="keras")

print("✅ Model and Vectorizer saved successfully!")
