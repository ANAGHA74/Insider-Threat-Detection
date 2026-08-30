"""
Data Cleaning Script for CERT Insider Threat Dataset r4.2
Loads raw CSV files, standardizes formats, handles missing values, and saves cleaned data as parquet.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Configuration
RAW_DATA_PATH = Path("data/raw")
CLEANED_DATA_PATH = Path("data/cleaned")
LDAP_PATH = RAW_DATA_PATH / "LDAP"

# Create directories if they don't exist
CLEANED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def print_validation_report(df, file_name):
    """Print validation report for a dataframe."""
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT: {file_name}")
    print(f"{'='*60}")
    print(f"Row count: {len(df):,}")
    print(f"\nNull counts per column:")
    print(df.isnull().sum())
    
    # Check for date columns and print date range
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        for col in date_cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"\n{col} range: {df[col].min()} to {df[col].max()}")
            elif col in df.columns:
                print(f"\n{col} - not yet parsed as datetime")
    
    print(f"\nSample rows (first 3):")
    print(df.head(3))
    print(f"{'='*60}\n")

def load_and_clean_logon():
    """Load and clean logon.csv"""
    print("\n[1/6] Processing logon.csv...")
    df = pd.read_csv(RAW_DATA_PATH / "logon.csv")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    
    # Drop rows with missing user IDs
    initial_count = len(df)
    df = df.dropna(subset=['user'])
    print(f"Dropped {initial_count - len(df)} rows with missing user IDs")
    
    # Drop exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows")
    
    print_validation_report(df, "logon.csv")
    return df

def load_and_clean_device():
    """Load and clean device.csv"""
    print("\n[2/6] Processing device.csv...")
    df = pd.read_csv(RAW_DATA_PATH / "device.csv")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    
    # Drop rows with missing user IDs
    initial_count = len(df)
    df = df.dropna(subset=['user'])
    print(f"Dropped {initial_count - len(df)} rows with missing user IDs")
    
    # Drop exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows")
    
    print_validation_report(df, "device.csv")
    return df

def load_and_clean_file():
    """Load and clean file.csv"""
    print("\n[3/6] Processing file.csv...")
    # Load only needed columns, exclude 'content'
    df = pd.read_csv(RAW_DATA_PATH / "file.csv", usecols=['id', 'date', 'user', 'pc', 'filename'])
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    
    # Drop rows with missing user IDs
    initial_count = len(df)
    df = df.dropna(subset=['user'])
    print(f"Dropped {initial_count - len(df)} rows with missing user IDs")
    
    # Drop duplicates by id only (id is unique per event)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    print(f"Dropped {initial_count - len(df)} duplicate rows (by id)")
    
    print_validation_report(df, "file.csv")
    return df

def load_and_clean_email():
    """Load and clean email.csv"""
    print("\n[4/6] Processing email.csv...")
    # Load only needed columns, exclude 'content'
    df = pd.read_csv(RAW_DATA_PATH / "email.csv", usecols=['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', 'attachments'])
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    
    # Drop rows with missing user IDs
    initial_count = len(df)
    df = df.dropna(subset=['user'])
    print(f"Dropped {initial_count - len(df)} rows with missing user IDs")
    
    # Drop duplicates by id only (id is unique per event)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    print(f"Dropped {initial_count - len(df)} duplicate rows (by id)")
    
    print_validation_report(df, "email.csv")
    return df

def load_and_clean_http():
    """Load and clean http.csv"""
    print("\n[5/6] Processing http.csv...")
    # Load only needed columns, exclude 'content'
    df = pd.read_csv(RAW_DATA_PATH / "http.csv", usecols=['id', 'date', 'user', 'pc', 'url'])
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S')
    
    # Drop rows with missing user IDs
    initial_count = len(df)
    df = df.dropna(subset=['user'])
    print(f"Dropped {initial_count - len(df)} rows with missing user IDs")
    
    # Drop duplicates by id only (id is unique per event)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    print(f"Dropped {initial_count - len(df)} duplicate rows (by id)")
    
    print_validation_report(df, "http.csv")
    return df

def load_and_combine_ldap():
    """Load and combine monthly LDAP CSVs into latest role/department per employee"""
    print("\n[6/6] Processing LDAP monthly files...")
    
    ldap_files = sorted(LDAP_PATH.glob("*.csv"))
    print(f"Found {len(ldap_files)} LDAP monthly files")
    
    all_ldap_dfs = []
    for file in ldap_files:
        print(f"  Loading {file.name}...")
        df = pd.read_csv(file)
        # Extract month-year from filename
        month_year = file.stem  # e.g., "2010-01"
        df['month_year'] = month_year
        all_ldap_dfs.append(df)
    
    # Combine all monthly snapshots
    combined_ldap = pd.concat(all_ldap_dfs, ignore_index=True)
    print(f"Combined LDAP rows: {len(combined_ldap):,}")
    
    # Standardize column names
    combined_ldap.columns = combined_ldap.columns.str.lower().str.strip()
    
    # Get the latest role/department per employee (most recent month_year)
    latest_ldap = combined_ldap.sort_values('month_year').groupby('user_id').last().reset_index()
    print(f"Unique employees after taking latest snapshot: {len(latest_ldap):,}")
    
    # Drop rows with missing user IDs
    initial_count = len(latest_ldap)
    latest_ldap = latest_ldap.dropna(subset=['user_id'])
    print(f"Dropped {initial_count - len(latest_ldap)} rows with missing user IDs")
    
    print_validation_report(latest_ldap, "LDAP (combined latest)")
    return latest_ldap

def main():
    """Main execution function."""
    print("="*60)
    print("CERT INSIDER THREAT DATASET - DATA CLEANING")
    print("="*60)
    
    # Check if raw data path exists
    if not RAW_DATA_PATH.exists():
        print(f"\nERROR: Raw data path not found: {RAW_DATA_PATH}")
        print("Please ensure data is organized as: data/raw/ with CSV files")
        return
    
    # Process each file
    try:
        logon_df = load_and_clean_logon()
        device_df = load_and_clean_device()
        file_df = load_and_clean_file()
        email_df = load_and_clean_email()
        http_df = load_and_clean_http()
        ldap_df = load_and_combine_ldap()
        
        # Save cleaned files as CSV
        print("\n" + "="*60)
        print("SAVING CLEANED FILES AS CSV")
        print("="*60)
        
        logon_df.to_csv(CLEANED_DATA_PATH / "logon_cleaned.csv", index=False)
        print("Saved: logon_cleaned.csv")
        
        device_df.to_csv(CLEANED_DATA_PATH / "device_cleaned.csv", index=False)
        print("Saved: device_cleaned.csv")
        
        file_df.to_csv(CLEANED_DATA_PATH / "file_cleaned.csv", index=False)
        print("Saved: file_cleaned.csv")
        
        email_df.to_csv(CLEANED_DATA_PATH / "email_cleaned.csv", index=False)
        print("Saved: email_cleaned.csv")
        
        http_df.to_csv(CLEANED_DATA_PATH / "http_cleaned.csv", index=False)
        print("Saved: http_cleaned.csv")
        
        ldap_df.to_csv(CLEANED_DATA_PATH / "ldap_cleaned.csv", index=False)
        print("Saved: ldap_cleaned.csv")
        
        print("\n" + "="*60)
        print("DATA CLEANING COMPLETE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("Please ensure all required CSV files exist in data/raw/")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
