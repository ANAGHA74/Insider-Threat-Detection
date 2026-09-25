import pandas as pd

df = pd.read_csv('data/cleaned/final_predictions.csv', low_memory=False)
df['week'] = pd.to_datetime(df['week'])
latest_week = df['week'].max()

print(f'Latest week in data: {latest_week}')
latest_data = df[df['week'] == latest_week]

print('\nLatest baseline values by cohort:')
for cohort in latest_data['peer_cohort'].unique()[:5]:
    cohort_data = latest_data[latest_data['peer_cohort'] == cohort].iloc[0]
    print(f'\n{cohort}:')
    print(f'  off_hours_ratio_rolling_mean: {cohort_data.get("off_hours_ratio_rolling_mean", 0):.3f}')
    print(f'  email_count_rolling_mean: {cohort_data.get("email_count_rolling_mean", 0):.1f}')
    print(f'  email_attachment_count_rolling_mean: {cohort_data.get("email_attachment_count_rolling_mean", 0):.1f}')
    print(f'  http_total_count_rolling_mean: {cohort_data.get("http_total_count_rolling_mean", 0):.1f}')
    print(f'  usb_events_rolling_mean: {cohort_data.get("usb_events_rolling_mean", 0):.1f}')
