import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
DATA_PATH = BASE_DIR / "data" / "career_data.csv"
MODEL_PATH = BASE_DIR / "model" / "career_model.pkl"


# Load dataset
data = pd.read_csv(DATA_PATH)

# Features and target
X = data.drop("career", axis=1)
y = data["career"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("\nGenerating confusion matrix...")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
)

plt.tight_layout()
plt.show()


# Save trained model
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved at: {MODEL_PATH}")