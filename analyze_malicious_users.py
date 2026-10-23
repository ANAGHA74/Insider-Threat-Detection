import pandas as pd

df = pd.read_csv('data/cleaned/test_predictions.csv')

print('='*60)
print('MALICIOUS EMPLOYEES ANALYSIS')
print('='*60)

print(f"\nTotal malicious rows in test set: {df['is_malicious'].sum()}")
print(f"Total unique malicious users: {df[df['is_malicious']==1]['user'].nunique()}")

print('\n' + '='*60)
print('PER-USER ANALYSIS')
print('='*60)

malicious_users = df[df['is_malicious']==1]['user'].unique()
print(f'\nTotal malicious users: {len(malicious_users)}')

users_caught = 0
users_partially_caught = 0
users_missed = 0

for user in malicious_users:
    user_data = df[df['user']==user]
    total_weeks = len(user_data)
    high_risk_weeks = len(user_data[user_data['risk_band']=='High'])
    medium_risk_weeks = len(user_data[user_data['risk_band']=='Medium'])
    max_risk = user_data['risk_score'].max()

    if high_risk_weeks > 0:
        users_caught += 1
        status = "[CAUGHT]"
    elif medium_risk_weeks > 0:
        users_partially_caught += 1
        status = "[PARTIAL]"
    else:
        users_missed += 1
        status = "[MISSED]"

    print(f'\n{user}: {status}')
    print(f'  Total weeks: {total_weeks}')
    print(f'  High risk weeks: {high_risk_weeks}')
    print(f'  Medium risk weeks: {medium_risk_weeks}')
    print(f'  Max risk score: {max_risk:.1f}')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Total malicious users: {len(malicious_users)}')
print(f'Fully caught (High risk): {users_caught} ({users_caught/len(malicious_users)*100:.1f}%)')
print(f'Partially caught (Medium risk): {users_partially_caught} ({users_partially_caught/len(malicious_users)*100:.1f}%)')
print(f'Completely missed (Low risk): {users_missed} ({users_missed/len(malicious_users)*100:.1f}%)')
print(f'Total detected (High + Medium): {users_caught + users_partially_caught} ({(users_caught + users_partially_caught)/len(malicious_users)*100:.1f}%)')
