import pandas as pd

# Use the final predictions which now includes DiCE explanations
df = pd.read_csv('data/cleaned/final_predictions.csv', low_memory=False)

# Ensure week is datetime
df['week'] = pd.to_datetime(df['week'])

# Save for dashboard
df.to_csv('data/cleaned/dashboard_dev_data.csv', index=False)

print(f"Created dashboard_dev_data.csv with {len(df)} rows")
print(f"Risk distribution: Low={len(df[df['risk_band']=='Low'])}, Medium={len(df[df['risk_band']=='Medium'])}, High={len(df[df['risk_band']=='High'])}")
if 'dice_explanation' in df.columns:
    print(f"DiCE explanations: {df['dice_explanation'].notna().sum()} cases have DiCE explanations")
else:
    print("DiCE explanations: column not present (run 06_dice_counterfactuals.py to add)")