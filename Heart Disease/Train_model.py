import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load dataset
data = pd.read_csv("heart_disease_and_hospitals.csv")

feature_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose_level', 'gender']

X = data[feature_cols].copy()
y = data["heart_disease"]

X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced') # ADD THIS PART
model.fit(X_train, y_train)

#SAVE MODEL AND THE SCALER

pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Success: Model and Scaler saved!")
print(f"Trained on: {feature_cols}")

print("--- DATASET DIAGNOSTICS ---")
print(data['heart_disease'].value_counts())