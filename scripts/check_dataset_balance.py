import pandas as pd

# Load the correct processed dataset
df = pd.read_csv("data/spam_processed_tfidf.pkl")
# Count the number of spam (1) and ham (0)
print(df["label"].value_counts())
