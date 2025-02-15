# Spam Email Classifier

## Project Overview
This project is a **Spam Email Classifier** that detects spam messages using **Machine Learning and Natural Language Processing (NLP)**. The project consists of data preprocessing, model training, and deployment in a web application using **Streamlit**.

## Folder Structure
```
spam_email_classifier/
│── data/                     # Store dataset
│   ├── spam.csv              # Raw dataset
│   ├── spam_processed.csv     # Preprocessed dataset
│── models/                   # Store trained model
│   ├── spam_classifier.pkl   # Saved model file
│   ├── vectorizer.pkl        # TF-IDF Vectorizer
│── app/                      # Deployment application
│   ├── main.py               # Script for classification & inference
│   ├── app.py                # Streamlit app
│── notebooks/                # Jupyter Notebook for EDA
│   ├── exploration.ipynb     # Exploratory Data Analysis (EDA)
│── scripts/                  # Helper scripts for preprocessing/training
│   ├── preprocess.py         # Phase 1 script (data preparation)
│── requirements.txt          # Dependencies
│── .gitignore                # Ignore unnecessary files
│── README.md                 # Documentation
│── train_model.py            # Script for training & saving model
```

## Setup Instructions
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
This script downloads the dataset, cleans the text, and saves a processed version.

### 5️⃣ Train the Model
```bash
python train_model.py
```
This script trains a **Naïve Bayes classifier**, evaluates its accuracy, and saves the model and vectorizer for deployment using **pickle**.

### 6️⃣ Test the Model Locally
To check if the model is working correctly, run:
```bash
python app/app.py
```
This launches a web interface where users can input emails and check if they are spam.

## Deployment using Streamlit Cloud
### **Deploying on Streamlit Cloud**
We used **Streamlit Community Cloud** for deployment. To deploy:
1. Push all project files to **GitHub**.
2. Go to **[Streamlit Cloud](https://share.streamlit.io/)**.
3. Click **"New App"** and select your repository.
4. In the **App file path**, enter:
   ```
   app/app.py
   ```
5. Click **"Deploy"**.
6. Your app will be live at a public URL (e.g., `https://your-app-name.streamlit.app`).

### **Testing the Deployed App**
Once the app is live, test it by entering email samples:
- **Spam Example:**
  ```
  Congratulations! You've won a free gift card. Click the link to claim now!
  ```
  ✅ Expected Output: **Spam**

- **Ham (Not Spam) Example:**
  ```
  Hey, let's meet for coffee tomorrow at 10 AM.
  ```
  ✅ Expected Output: **Not Spam**

## Features
✅ **Preprocessing**: Text cleaning and tokenization
✅ **Machine Learning**: Naïve Bayes classification
✅ **Model Storage**: Model and vectorizer are saved using **pickle**
✅ **Deployment**: Web app (Streamlit Cloud)
✅ **Interactive UI**: Classify emails in real time

## Next Steps
- Improve model accuracy with deep learning (e.g., LSTMs, Transformers)
- Add API support with FastAPI
- Deploy on cloud (AWS/GCP/Heroku)

## Author
**Maki Dizon**
[GitHub](https://github.com/YOUR-USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR-PROFILE)

