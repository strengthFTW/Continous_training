from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

# load dataset
X, y = load_iris(return_X_y=True)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

# evaluation gate
if accuracy < 0.90:
    print("Model accuracy too low. Failing pipeline.")
    sys.exit(1)

# create model folder
os.makedirs("model", exist_ok=True)

# save model
joblib.dump(model, "model/model.pkl")

print("Model passed evaluation and saved successfully.")