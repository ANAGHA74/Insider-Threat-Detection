import pandas as pd

df = pd.read_csv('data/cleaned/final_predictions.csv', low_memory=False)
df['week'] = pd.to_datetime(df['week'])

# Get high-risk cases and their z-scores
high_risk = df[df['risk_band'] == 'High'].copy()

print("High-risk cases and their key z-scores:")
print("=" * 80)

for i, row in high_risk.head(5).iterrows():
    print(f"\nUser: {row['user']}, Week: {row['week']}, Risk Score: {row['risk_score']:.1f}")
    print(f"Peer Cohort: {row['peer_cohort']}")
    print(f"Peer Deviation Score: {row['peer_deviation_score']:.4f}")
    
    # Show key z-scores
    key_z_cols = [col for col in row.index if 'z_score' in col and any(feat in col for feat in ['off_hours_ratio', 'email_attachment_count', 'email_count', 'http_total_count', 'usb_events'])]
    print("Key Z-Scores:")
    for col in key_z_cols:
        print(f"  {col}: {row[col]:.4f}")
