import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------
# Load CSV dataset
# ---------------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df = df.rename(columns={
    'v1': 'label',
    'v2': 'message'
})

# Encode labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features & target
X = df['message']
y = df['label']

# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------
# ML Pipeline with Logistic Regression
# ---------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )),
    ("clf", LogisticRegression(
        solver='liblinear',  # good for small datasets / binary classification
        max_iter=1000
    ))
])

# ---------------------------------
# MLflow Tracking
# ---------------------------------
mlflow.set_experiment("Spam_SMS_Classification_CSV_Logistic")

with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="SpamSMSClassifierCSV_Logistic"
    )

    print("Training completed")
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
