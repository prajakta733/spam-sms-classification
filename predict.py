import mlflow.pyfunc

model_uri = "models:/SpamSMSClassifierCSV/latest"
model = mlflow.pyfunc.load_model(model_uri)

messages = [
    "Congratulations! You have won a free iPhone. Click now!",
    "Hey, are we meeting at 6 pm today?"
]

predictions = model.predict(messages)

for msg, pred in zip(messages, predictions):
    print(msg)
    print("Prediction:", "SPAM" if pred == 1 else "HAM")
    print("-" * 50)
