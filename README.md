# Spam Email Classifier

## Project Overview

This project is a **Spam Email Classifier** that detects spam messages using **Deep Learning and Natural Language Processing (NLP)**.  
The project consists of **data preprocessing, model training (LSTM + TF-IDF), and deployment in a web application using Streamlit**.

✅ **Powered by Deep Learning:** Uses an LSTM neural network to process email sequences and classify messages with high accuracy.

---

## 🤖 How It Works (Model Training: LSTM + TF-IDF)

### 🔤 TF-IDF — _Term Frequency-Inverse Document Frequency_

- Converts email text into a numerical format based on how important each word is.
- Weighs words that are rare but meaningful higher than common words.
- Helps the model understand which words carry weight in an email.

📌 Think of TF-IDF as:

> "How rare and meaningful is this word for this email?"

### 🧠 LSTM — _Long Short-Term Memory_

- A type of Recurrent Neural Network (RNN) designed to process sequences (like sentences).
- Remembers the order and context of words, just like humans do.
- Helps detect spam phrases like "win a free prize" or "urgent account update."

📌 Think of LSTM as:

> "I read this email like a human — understanding the flow and meaning."

---

## 📺 Folder Structure

```
spam_email_classifier/
│── data/                     # Store dataset
│   ├─ spam.csv              # Raw dataset
│   ├─ spam_processed_balanced.csv  # Balanced preprocessed dataset
│── models/                   # Store trained model & vectorizer
│   ├─ spam_classifier_tfidf.keras   # Saved Deep Learning model
│   ├─ tfidf_vectorizer.joblib        # TF-IDF Vectorizer
│── app/                      # Deployment application
│   ├─ app.py                # Streamlit app
│── notebooks/                # Jupyter Notebook for EDA
│   ├─ exploration.ipynb     # Exploratory Data Analysis (EDA)
│── scripts/                  # Helper scripts for preprocessing/training
│   ├─ preprocess.py         # Phase 1 script (data preparation)
│   ├─ check_dataset_balance.py  # Script to check dataset distribution
│── requirements.txt          # Dependencies
│── .gitignore                # Ignore unnecessary files
│── README.md                 # Documentation
│── train_model.py            # Deep Learning Model Training
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/spam-email-classifier.git
cd spam-email-classifier
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Data Preprocessing

```bash
python scripts/preprocess.py
```

💚 **This script:**

- Downloads the dataset
- Cleans the text
- Applies **TF-IDF vectorization**
- **Balances the dataset** using **SMOTE**
- Saves processed data for model training.

### 5️⃣ Train the Model (Deep Learning)

```bash
python train_model.py
```

💚 **This script:**

- Uses **TF-IDF** for text representation.
- Trains an **LSTM-based Deep Learning model**.
- Evaluates the model's accuracy.
- Saves the trained **model (`.keras`) and vectorizer (`.joblib`)**.

### 6️⃣ Run Streamlit App Locally

To test email classification in the web UI:

```bash
streamlit run app/app.py
```

This will launch a **Streamlit web app** for email classification.

🧹 **Bonus:** The app also includes a **Clear Cache** button to refresh memory/state — helpful for resolving prediction bugs.

💡 **Note:** If you encounter an error, try refreshing the browser (F5) and click the **Clear Cache** button before retrying.

---

## 🌟 **Deployment using Streamlit Cloud**

### **Steps to Deploy on Streamlit**

1️⃣ Push all project files to **GitHub**  
2️⃣ Go to **[Streamlit Cloud](https://share.streamlit.io/)**  
3️⃣ Click **"New App"** and select your repository  
4️⃣ In the **App file path**, enter:

```
app/app.py
```

5️⃣ Click **"Deploy"**  
6️⃣ Your app will be live at a public URL (e.g., `https://spamemailclassifier.streamlit.app/`)

---

## 📩 **Testing the Classifier**

Once deployed, you can test the app by entering email samples.

### ✅ **Spam Example**

#### **Sample 1**

```
Subject: 🎉 Congratulations! You’re a Lucky Winner!
From: "Rewards Department" <rewards@freelottery.com>

Dear Valued Customer,

We are excited to inform you that **you have won a $500 Amazon Gift Card** as part of our exclusive customer loyalty program! 🎁

To claim your prize, simply **click the link below** and fill out the verification form:
🔗 [Claim Your Reward Now](http://fraudulent-link.com)

Hurry! This offer is only valid for the next **24 hours**.

Best Regards,
**The Rewards Team**
📧 Contact us at: support@freelottery.com
```

#### **Sample 2**

```
Subject: 🎁 You’ve Been Selected to Win a Free iPhone 15!

From: “Apple Rewards Center” <support@apple-rewards-center.com>

Dear Valued Customer,

Congratulations! 🎉 You’ve been **randomly selected** to receive a **brand new iPhone 15** as part of our global product testing program.

To claim your prize:
✅ Click the secure link below:
👉 http://free-iphone15-offer.com/claim-now

⚠ Hurry! This exclusive offer expires in the next **3 hours**.

There is **no cost** to you — just confirm your shipping address.

Best regards,
The Apple Rewards Team
📞 Customer Support: 1-800-FAKE-NUMBER
```

💚 **Expected Output:** 🚨 **Spam (Confidence: 99.99%)**

---

### ✅ **Ham (Not Spam) Example**

```
Subject: Meeting Agenda for Tomorrow's Product Review

From: Sarah Bennett <sarah.bennett@yourcompany.com>

Hi Team,

Just a reminder that we have our product review meeting scheduled for **10:00 AM tomorrow** in the main conference room.

Here’s the agenda:
- Sprint recap
- Upcoming features
- Bug prioritization
- Open discussion

Please review the attached notes beforehand.

Best,
Sarah
```

💚 **Expected Output:** ✅ **Not Spam (Confidence: ~0.01%)**

---

## 🧠 **Understanding Confidence Scores**

| Example Email                                    | Confidence Score | Prediction      |
| ------------------------------------------------ | ---------------- | --------------- |
| `"Win a free prize! Click here!"`                | **98.50%**       | 🚨 **Spam**     |
| `"Hey, how's your day?"`                         | **0.02%**        | ✅ **Not Spam** |
| `"Urgent! Your account is at risk. Verify now."` | **99.99%**       | 🚨 **Spam**     |

---

## 🚀 **Next Steps**

- Fine-tune LSTM model for **better accuracy**
- Experiment with **Transformers (BERT/GPT)** for spam detection
- Build a **FastAPI service** for integration into email clients
- Deploy on **AWS/GCP/Heroku** for wider usage

---

## 👤 **Author**

**Maki Dizon**  
🌐 [GitHub](https://github.com/YOUR-USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR-PROFILE)  
📧 **Contact:** maki@example.com

---

## **💡 Now You're Ready to Detect Spam with Deep Learning! 🚀🔥**
