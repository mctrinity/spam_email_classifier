import pandas as pd
import os
import requests

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

data.columns = ["label", "message"]

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Save processed dataset
processed_path = "data/spam_processed.csv"
data.to_csv(processed_path, index=False)
print(f"Processed dataset saved to {processed_path}")
