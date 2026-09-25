import pandas as pd
import joblib

# Load data and model
df = pd.read_csv('data/cleaned/final_predictions.csv', low_memory=False)
model = joblib.load('models/xgb_risk_model.pkl')

# Get a high-risk case
high_risk = df[df['risk_band'] == 'High'].iloc[0]
model_features = model.get_booster().feature_names

# Test prediction on actual high-risk case
input_row = high_risk[model_features].values.reshape(1, -1)
pred = model.predict_proba(input_row)[0][1]

print(f'Actual high-risk case prediction: {pred}')
print(f'Expected (from saved risk_score): {high_risk["risk_score"]/100}')
print(f'User: {high_risk["user"]}, Week: {high_risk["week"]}')
print(f'Peer deviation score: {high_risk["peer_deviation_score"]}')
