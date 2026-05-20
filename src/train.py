from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# load dataset
X, y = load_iris(return_X_y=True)

# train model
model = RandomForestClassifier()
model.fit(X, y)

# create model directory if not exists
os.makedirs("model", exist_ok=True)

# save trained model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully")