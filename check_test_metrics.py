"""
Evaluate on TEST SET only (same split as training)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_PATH = Path("data/cleaned/user_weekly_features_with_peer_scores.csv")
MODEL_PATH = Path("models")

# Raw behavioral features
RAW_FEATURES = [
    'logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_count', 'email_attachment_count',
    'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count'
]

def load_and_prepare_data():
    """Load and prepare data for training."""
    print("Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    df = df[df['has_sufficient_baseline'] == True].copy()
    print(f"Total rows after filtering: {len(df):,}")
    return df

def get_model_features(df):
    """Identify feature columns for the model."""
    feature_cols = RAW_FEATURES.copy()
    z_score_cols = [col for col in df.columns if col.endswith('_z_score')]
    feature_cols.extend(z_score_cols)
    if 'peer_deviation_score' in df.columns:
        feature_cols.append('peer_deviation_score')
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_cols = [col for col in feature_cols if col not in ['user', 'week', 'role', 'department', 'peer_cohort', 'business_unit', 'has_sufficient_baseline', 'is_malicious']]
    return feature_cols

def split_by_user(df):
    """Split data by user using SAME split as training."""
    print("\nSplitting data by user (same as training)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['user']))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"\nTrain set: {train_df['user'].nunique()} users, {len(train_df):,} rows")
    print(f"Test set: {test_df['user'].nunique()} users, {len(test_df):,} rows")
    print(f"Test malicious rows: {test_df['is_malicious'].sum()}")
    print(f"Test malicious users: {test_df[test_df['is_malicious'] == 1]['user'].nunique()}")
    
    return train_df, test_df

def main():
    print("="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    feature_cols = get_model_features(df)
    
    # Split (same as training)
    train_df, test_df = split_by_user(df)
    
    # Prepare features
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['is_malicious']
    
    # Load ensemble model
    print("\nLoading ensemble model...")
    ensemble = joblib.load(MODEL_PATH / "catboost_ensemble_model.pkl")
    
    # Get predictions
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.20).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print("\n" + "="*80)
    print("TEST SET METRICS (Ensemble, threshold 0.20)")
    print("="*80)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    
    # User-level detection
    malicious_users = test_df[test_df['is_malicious'] == 1]['user'].unique()
    flagged_users = test_df[y_pred == 1]['user'].unique()
    malicious_flagged = set(malicious_users).intersection(set(flagged_users))
    print(f"\nUser-level detection: {len(malicious_flagged)}/{len(malicious_users)} malicious users flagged")
    
    print("="*80)

if __name__ == "__main__":
    main()
