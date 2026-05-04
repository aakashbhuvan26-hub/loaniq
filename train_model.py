"""
train_model.py
Train and save the Gradient Boosting Classifier on the loan dataset.

Usage:
    python train_model.py

Dataset:
    Download from: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
    Expected file: loan_sanction_train.csv (or loan_train.csv)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ─── Load Data ────────────────────────────────────────────────────────────────

CSV_PATH = "loan_sanction_train.csv"

if not os.path.exists(CSV_PATH):
    # Try alternate filename
    CSV_PATH = "loan_train.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset CSV not found. Download from Kaggle and place as '{CSV_PATH}'"
    )

df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── Preprocessing ────────────────────────────────────────────────────────────

# Drop Loan_ID
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

# Fill missing values
df["Gender"].fillna("Male", inplace=True)
df["Married"].fillna("Yes", inplace=True)
df["Dependents"].fillna("0", inplace=True)
df["Self_Employed"].fillna("No", inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(360, inplace=True)
df["Credit_History"].fillna(1, inplace=True)
df["blank selection"].fillna(0, inplace=True)

# Encode categoricals
le = LabelEncoder()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    df[col] = le.fit_transform(df[col])

# Dependents: convert "3+" → 3
df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

# ─── Feature / Target Split ───────────────────────────────────────────────────

FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "blank selection"
]

X = df[FEATURES].values
y = df["Loan_Status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Train ────────────────────────────────────────────────────────────────────

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
clf.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# ─── Save ─────────────────────────────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n[INFO] Model saved to model.pkl ✓")
print("[INFO] Feature order for inference:")
for i, f in enumerate(FEATURES):
    print(f"  [{i}] {f}")
