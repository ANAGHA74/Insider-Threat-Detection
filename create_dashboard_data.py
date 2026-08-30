import pandas as pd
import numpy as np
import json

# Load the real data
df = pd.read_csv('data/cleaned/user_weekly_features.csv')

# Add risk_score based on off_hours_ratio and usb_events (plausible derivation)
# Scale up to 0-100 range
df['risk_score'] = (df['off_hours_ratio'] * 50 + df['usb_events'] * 20).clip(0, 100)

# Add risk_band based on thresholds
def get_risk_band(score):
    if score < 30:
        return 'Low'
    elif score < 70:
        return 'Medium'
    else:
        return 'High'

df['risk_band'] = df['risk_score'].apply(get_risk_band)

# Add shap_top_features for High risk rows only
def generate_shap_features(row):
    if row['risk_band'] == 'High':
        # Generate plausible feature importance pairs using real feature names
        features = []
        if row['off_hours_ratio'] > 0.5:
            features.append(('off_hours_ratio', row['off_hours_ratio'] * 10))
        if row['usb_events'] > 0:
            features.append(('usb_events', row['usb_events'] * 5))
        if row['file_access_count'] > 100:
            features.append(('file_access_count', min(row['file_access_count'] / 100, 5)))
        if len(features) == 0:
            features.append(('off_hours_ratio', row['off_hours_ratio'] * 10))
        return json.dumps(features)
    return None

df['shap_top_features'] = df.apply(generate_shap_features, axis=1)

# Add dice_explanation for High risk rows only
def generate_dice_explanation(row):
    if row['risk_band'] == 'High':
        # Generate plausible counterfactual explanation
        if row['off_hours_ratio'] > 0.5:
            new_ratio = max(0.1, row['off_hours_ratio'] - 0.3)
            new_score = max(20, row['risk_score'] - 30)
            return f"Reducing off_hours_ratio from {row['off_hours_ratio']:.2f} to {new_ratio:.2f} would lower risk score from {row['risk_score']:.0f} to {new_score:.0f}"
        elif row['usb_events'] > 0:
            new_usb = 0
            new_score = max(20, row['risk_score'] - 25)
            return f"Reducing usb_events from {row['usb_events']:.0f} to {new_usb:.0f} would lower risk score from {row['risk_score']:.0f} to {new_score:.0f}"
        else:
            return f"Reducing overall activity patterns would lower risk score from {row['risk_score']:.0f} to 25"
    return None

df['dice_explanation'] = df.apply(generate_dice_explanation, axis=1)

# Save to new file (do not overwrite original)
df.to_csv('data/cleaned/dashboard_dev_data.csv', index=False)

print(f"Created dashboard_dev_data.csv with {len(df)} rows")
print(f"Risk distribution: Low={len(df[df['risk_band']=='Low'])}, Medium={len(df[df['risk_band']=='Medium'])}, High={len(df[df['risk_band']=='High'])}")
