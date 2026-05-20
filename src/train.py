from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import sys
import mlflow
import mlflow.sklearn

# start mlflow tracking
mlflow.set_experiment("continuous-training")

with mlflow.start_run():

    # load dataset
    X, y = load_iris(return_X_y=True)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    model = RandomForestClassifier()

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")

    # log metric
    mlflow.log_metric("accuracy", accuracy)

    # log model
    mlflow.sklearn.log_model(model, "model")

    # evaluation gate
    if accuracy < 0.90:
        print("Model accuracy too low. Failing pipeline.")
        sys.exit(1)

    # save local model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("Model passed evaluation and saved successfully.")