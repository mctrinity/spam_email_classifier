# 🚀 Execution Flow of Spam Email Classifier

This document outlines the step-by-step execution flow of the **Spam Email Classifier**, from **data preparation** to **real-time spam detection** in the web app.

---

## **📌 Stage 1: Data Preprocessing (`scripts/preprocess.py`)**
### **Goal:** Prepare the dataset for training by cleaning text data.

🔹 **What Happens Here?**
1. **Loads the raw dataset** (`data/spam.csv`).
2. **Cleans text data** (removes punctuation, converts to lowercase, removes numbers).
3. **Maps labels** (`ham → 0`, `spam → 1`).
4. **Saves the processed dataset** (`data/spam_processed.csv`).

🔹 **Run This Command:**
```bash
python scripts/preprocess.py
```
🔹 **Expected Output:**
✅ `data/spam_processed.csv` is created.

---

## **📌 Stage 2: Model Training (`train_model.py`)**
### **Goal:** Train the spam classifier using Naïve Bayes and save the model.

🔹 **What Happens Here?**
1. **Loads the cleaned dataset** (`data/spam_processed.csv`).
2. **Splits data** into training and testing sets.
3. **Converts text to numerical features** using **TF-IDF Vectorization**.
4. **Trains the Naïve Bayes classifier** (`MultinomialNB()`).
5. **Evaluates model performance** (accuracy, precision, recall).
6. **Saves the trained model** (`models/spam_classifier.pkl`) and **vectorizer** (`models/vectorizer.pkl`).

🔹 **Run This Command:**
```bash
python train_model.py
```
🔹 **Expected Output:**
✅ Model accuracy is displayed.
✅ `models/spam_classifier.pkl` & `models/vectorizer.pkl` are saved.

---

## **📌 Stage 3: Running the Web App (`app.py`)**
### **Goal:** Use the trained model to classify emails in real-time via a web app.

🔹 **What Happens Here?**
1. **Loads the saved model** (`models/spam_classifier.pkl`).
2. **Loads the saved vectorizer** (`models/vectorizer.pkl`).
3. **Creates a simple UI** with **Streamlit**.
4. **Takes user input (email text)**.
5. **Converts text into numerical features** using the **TF-IDF vectorizer**.
6. **Predicts if the input is spam or not**.
7. **Displays the classification result**.

🔹 **Run This Command:**
```bash
streamlit run app/app.py
```
🔹 **Expected Output:**
✅ A web app launches in your browser (`http://localhost:8501`).
✅ Users can enter emails and see if they are **Spam** or **Not Spam**.

---

## **📌 Stage 4: Deployment to Streamlit Cloud**
### **Goal:** Make the web app accessible online.

🔹 **What Happens Here?**
1. **Push the project to GitHub**:
   ```bash
   git add .
   git commit -m "Deploying to Streamlit"
   git push origin main
   ```
2. **Go to Streamlit Cloud** ([share.streamlit.io](https://share.streamlit.io/)).
3. **Click “New App” → Select GitHub Repo**.
4. **Set the app path to `app/app.py`**.
5. **Click "Deploy"**.

🔹 **Expected Output:**
✅ Your web app is now live at a public URL!

---

## **🎯 Summary of Execution Flow**
| **Stage** | **Script** | **Purpose** | **Command** |
|-----------|-----------|-------------|-------------|
| **1️⃣ Data Preparation** | `scripts/preprocess.py` | Cleans dataset | `python scripts/preprocess.py` |
| **2️⃣ Model Training** | `train_model.py` | Trains & saves the model | `python train_model.py` |
| **3️⃣ Web App** | `app.py` | Runs a Streamlit UI for real-time predictions | `streamlit run app/app.py` |
| **4️⃣ Deployment** | Streamlit Cloud | Hosts app online | Deploy via [share.streamlit.io](https://share.streamlit.io/) |

---



