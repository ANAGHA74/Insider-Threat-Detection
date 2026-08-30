"""
Feature Engineering Script for CERT Insider Threat Dataset
Aggregates cleaned log files into weekly per-user features and joins ground-truth labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time

# Configuration
CLEANED_DATA_PATH = Path("data/cleaned")
RAW_DATA_PATH = Path("data/raw")
ANSWERS_PATH = Path("archive/answers")
OUTPUT_PATH = Path("data/cleaned")

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def load_cleaned_data():
    """Load all cleaned CSV files."""
    print("Loading cleaned data files...")
    
    try:
        logon_df = pd.read_csv(CLEANED_DATA_PATH / "logon_cleaned.csv")
        logon_df['date'] = pd.to_datetime(logon_df['date'], errors='coerce')
        logon_df = logon_df.dropna(subset=['date'])
        
        device_df = pd.read_csv(CLEANED_DATA_PATH / "device_cleaned.csv")
        device_df['date'] = pd.to_datetime(device_df['date'], errors='coerce')
        device_df = device_df.dropna(subset=['date'])
        
        file_df = pd.read_csv(CLEANED_DATA_PATH / "file_cleaned.csv")
        file_df['date'] = pd.to_datetime(file_df['date'], errors='coerce')
        file_df = file_df.dropna(subset=['date'])
        
        email_df = pd.read_csv(CLEANED_DATA_PATH / "email_cleaned.csv")
        email_df['date'] = pd.to_datetime(email_df['date'], errors='coerce')
        email_df = email_df.dropna(subset=['date'])
        
        http_df = pd.read_csv(CLEANED_DATA_PATH / "http_cleaned.csv")
        http_df['date'] = pd.to_datetime(http_df['date'], errors='coerce')
        http_df = http_df.dropna(subset=['date'])
        
        ldap_df = pd.read_csv(CLEANED_DATA_PATH / "ldap_cleaned.csv")
        
        print(f"Loaded: logon ({len(logon_df):,} rows)")
        print(f"Loaded: device ({len(device_df):,} rows)")
        print(f"Loaded: file ({len(file_df):,} rows)")
        print(f"Loaded: email ({len(email_df):,} rows)")
        print(f"Loaded: http ({len(http_df):,} rows)")
        print(f"Loaded: ldap ({len(ldap_df):,} rows)")
        
        return logon_df, device_df, file_df, email_df, http_df, ldap_df
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        print("Please run the data cleaning script first.")
        raise

def is_off_hours(timestamp):
    """Check if timestamp is outside 9am-6pm."""
    hour = timestamp.hour
    return hour < 9 or hour >= 18

def engineer_logon_features(logon_df):
    """Engineer weekly logon features."""
    print("\nEngineering logon features...")
    
    # Add week column
    logon_df['week'] = logon_df['date'].dt.to_period('W').dt.start_time
    
    # Off-hours flag (handle NaT values)
    logon_df['is_off_hours'] = logon_df['date'].apply(lambda x: is_off_hours(x) if pd.notna(x) else False)
    
    # Aggregate per user per week
    logon_features = logon_df.groupby(['user', 'week']).agg({
        'id': 'count',  # logon_count
        'is_off_hours': 'mean'  # off_hours_ratio
    }).rename(columns={'id': 'logon_count', 'is_off_hours': 'off_hours_ratio'}).reset_index()
    
    print(f"Logon features: {len(logon_features):,} user-weeks")
    return logon_features

def engineer_device_features(device_df):
    """Engineer weekly device features."""
    print("\nEngineering device features...")
    
    # Add week column
    device_df['week'] = device_df['date'].dt.to_period('W').dt.start_time
    
    # USB events = Connect events
    usb_events = device_df[device_df['activity'] == 'Connect'].groupby(['user', 'week']).size().reset_index(name='usb_events')
    
    # Ensure all user-weeks exist
    all_weeks = device_df.groupby(['user', 'week']).size().reset_index()[['user', 'week']]
    device_features = all_weeks.merge(usb_events, on=['user', 'week'], how='left').fillna(0)
    
    print(f"Device features: {len(device_features):,} user-weeks")
    return device_features

def engineer_file_features(file_df):
    """Engineer weekly file features."""
    print("\nEngineering file features...")
    
    # Add week column
    file_df['week'] = file_df['date'].dt.to_period('W').dt.start_time
    
    # File access count
    file_features = file_df.groupby(['user', 'week']).agg({
        'id': 'count'  # file_access_count
    }).rename(columns={'id': 'file_access_count'}).reset_index()
    
    # Copy to removable media (simplified: count all file operations as potential copies)
    # In a real implementation, you'd check filename patterns for removable media indicators
    file_features['copy_to_removable_count'] = 0  # Placeholder - would need filename analysis
    
    print(f"File features: {len(file_features):,} user-weeks")
    return file_features

def engineer_email_features(email_df):
    """Engineer weekly email features."""
    print("\nEngineering email features...")
    
    # Add week column
    email_df['week'] = email_df['date'].dt.to_period('W').dt.start_time
    
    # Aggregate per user per week
    email_features = email_df.groupby(['user', 'week']).agg({
        'id': 'count',  # email_count
        'attachments': 'sum',  # email_attachment_count
        'size': 'mean'  # avg_attachment_size
    }).rename(columns={'id': 'email_count', 'attachments': 'email_attachment_count', 'size': 'avg_attachment_size'}).reset_index()
    
    print(f"Email features: {len(email_features):,} user-weeks")
    return email_features

def engineer_http_features(http_df):
    """Engineer weekly HTTP features."""
    print("\nEngineering HTTP features...")
    
    # Add week column
    http_df['week'] = http_df['date'].dt.to_period('W').dt.start_time
    
    # Simplified upload/download detection based on URL patterns
    # Upload: URLs containing 'upload', 'put', 'post'
    # Download: URLs containing 'download', 'get'
    http_df['is_upload'] = http_df['url'].str.contains('upload|put|post', case=False, na=False).astype(int)
    http_df['is_download'] = http_df['url'].str.contains('download|get', case=False, na=False).astype(int)
    
    # Aggregate per user per week
    http_features = http_df.groupby(['user', 'week']).agg({
        'id': 'count',  # http_total_count
        'is_upload': 'sum',  # http_upload_count
        'is_download': 'sum'  # http_download_count
    }).rename(columns={'id': 'http_total_count', 'is_upload': 'http_upload_count', 'is_download': 'http_download_count'}).reset_index()
    
    print(f"HTTP features: {len(http_features):,} user-weeks")
    return http_features

def load_ground_truth():
    """Load ground truth malicious scenario labels."""
    print("\nLoading ground truth labels...")
    
    insiders_df = pd.read_csv(ANSWERS_PATH / "insiders.csv")
    
    # Parse dates with flexible format to handle inconsistencies
    insiders_df['start'] = pd.to_datetime(insiders_df['start'], format='mixed', errors='coerce')
    insiders_df['end'] = pd.to_datetime(insiders_df['end'], format='mixed', errors='coerce')
    
    # Drop rows with invalid dates
    initial_count = len(insiders_df)
    insiders_df = insiders_df.dropna(subset=['start', 'end'])
    print(f"Dropped {initial_count - len(insiders_df)} rows with invalid dates")
    
    print(f"Loaded {len(insiders_df)} malicious scenarios")
    return insiders_df

def add_malicious_labels(features_df, insiders_df):
    """Add is_malicious label based on ground truth scenarios."""
    print("\nAdding malicious labels...")
    
    # Initialize label column
    features_df['is_malicious'] = 0
    
    # Mark user-weeks that fall within any malicious scenario
    for _, row in insiders_df.iterrows():
        user = row['user']
        start_date = row['start']
        end_date = row['end']
        
        # Find user-weeks that overlap with this scenario
        mask = (features_df['user'] == user) & \
               (features_df['week'] >= start_date) & \
               (features_df['week'] <= end_date)
        
        features_df.loc[mask, 'is_malicious'] = 1
    
    malicious_count = features_df['is_malicious'].sum()
    print(f"Marked {malicious_count} user-weeks as malicious")
    
    return features_df

def merge_all_features(logon_feat, device_feat, file_feat, email_feat, http_feat, ldap_df):
    """Merge all feature tables."""
    print("\nMerging all features...")
    
    # Start with logon features as base
    merged = logon_feat.copy()
    
    # Merge other features
    merged = merged.merge(device_feat, on=['user', 'week'], how='outer')
    merged = merged.merge(file_feat, on=['user', 'week'], how='outer')
    merged = merged.merge(email_feat, on=['user', 'week'], how='outer')
    merged = merged.merge(http_feat, on=['user', 'week'], how='outer')
    
    # Merge LDAP role/department info (handle missing columns gracefully)
    ldap_cols = ['user_id', 'role', 'department', 'business_unit']
    available_cols = [col for col in ldap_cols if col in ldap_df.columns]
    ldap_subset = ldap_df[available_cols]
    merged = merged.merge(ldap_subset, left_on='user', right_on='user_id', how='left')
    if 'user_id' in merged.columns:
        merged = merged.drop(columns=['user_id'])
    
    # Fill NaN values with 0 for numeric columns (only fill columns that exist)
    all_numeric_cols = ['logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count', 
                        'copy_to_removable_count', 'email_count', 'email_attachment_count', 
                        'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count']
    numeric_cols = [col for col in all_numeric_cols if col in merged.columns]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    print(f"Merged features: {len(merged):,} user-weeks")
    return merged

def print_validation_report(df):
    """Print validation report for final features."""
    print(f"\n{'='*60}")
    print("VALIDATION REPORT: User Weekly Features")
    print(f"{'='*60}")
    print(f"Total rows (user-weeks): {len(df):,}")
    print(f"Unique users: {df['user'].nunique():,}")
    
    # Handle potential NaT in week column
    if 'week' in df.columns:
        df_week = df.dropna(subset=['week'])
        if len(df_week) > 0:
            print(f"Date range: {df_week['week'].min()} to {df_week['week'].max()}")
        else:
            print("Date range: No valid week data")
    
    print(f"\nMalicious vs Normal:")
    print(df['is_malicious'].value_counts())
    print(f"Malicious ratio: {df['is_malicious'].mean():.4f}")
    
    print(f"\nFeature statistics:")
    print(df.describe())
    
    print(f"\nSample rows (first 5):")
    print(df.head())
    print(f"{'='*60}\n")

def main():
    """Main execution function."""
    print("="*60)
    print("FEATURE ENGINEERING - WEEKLY USER FEATURES")
    print("="*60)
    
    # Load cleaned data
    logon_df, device_df, file_df, email_df, http_df, ldap_df = load_cleaned_data()
    
    # Engineer features for each data source
    logon_features = engineer_logon_features(logon_df)
    device_features = engineer_device_features(device_df)
    file_features = engineer_file_features(file_df)
    email_features = engineer_email_features(email_df)
    http_features = engineer_http_features(http_df)
    
    # Merge all features
    merged_features = merge_all_features(
        logon_features, device_features, file_features, 
        email_features, http_features, ldap_df
    )
    
    # Load ground truth and add labels
    insiders_df = load_ground_truth()
    final_features = add_malicious_labels(merged_features, insiders_df)
    
    # Print validation report
    print_validation_report(final_features)
    
    # Save final features
    print("\nSaving final features...")
    output_file = OUTPUT_PATH / "user_weekly_features.csv"
    final_features.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
