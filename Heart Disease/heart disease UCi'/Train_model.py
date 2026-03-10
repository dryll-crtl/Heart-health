import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('heart_cleveland_upload.csv')

X = df.drop('condition', axis=1)
y = df['condition']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'heart_model.pkl')

print("Model trained and saved successfully as 'heart_model.pkl'!")