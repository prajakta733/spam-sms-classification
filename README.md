# 📩 Spam SMS Classification using MLflow

This project is a **Spam SMS Classification system** built using **Machine Learning and MLflow**.  
It classifies SMS messages as **SPAM** or **HAM** using a text classification model.

---

## 📌 Project Overview

Spam messages are a common problem. This project:
- Trains a machine learning model on SMS text data
- Tracks experiments using **MLflow**
- Registers the trained model
- Uses the model to make predictions on new SMS messages

---

## 📂 Project Structure

├── data/ # Dataset files
├── mlruns/ # MLflow experiment tracking
├── train.py # Model training script
├── predict.py # Model prediction script
├── requirements.txt # Project dependencies
├── mlflow.db # MLflow tracking database
├── .gitignore # Git ignored files
└── README.md # Project documentation


## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- TF-IDF Vectorizer
- Logistic Regression / Naive Bayes

# Create virtual environment

python -m venv venv
source venv/bin/activate   

# Install dependencies

pip install -r requirements.txt

# Train the Model

python train.py

# Make Predictions

python predict.py
