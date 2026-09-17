"""
XGBoost Risk Scoring Model Training
Trains a risk model using peer-cohort features and evaluates against a static baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_PATH = Path("data/cleaned/user_weekly_features_with_peer_scores.csv")
OUTPUT_PATH = Path("data/cleaned")
MODEL_PATH = Path("models")

# Create output directories
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Raw behavioral features
RAW_FEATURES = [
    'logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_count', 'email_attachment_count',
    'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count'
]

# Columns to exclude from model features
EXCLUDE_COLS = [
    'user', 'week', 'role', 'department', 'peer_cohort', 'business_unit',
    'has_sufficient_baseline', 'is_malicious'
]

def load_and_prepare_data():
    """Load and prepare data for training."""
    print("Loading data...")
    df = pd.read_csv(FEATURES_PATH)
    
    print(f"Total rows before filtering: {len(df):,}")
    
    # Filter to rows with sufficient baseline
    df = df[df['has_sufficient_baseline'] == True].copy()
    print(f"Rows after filtering (has_sufficient_baseline=True): {len(df):,}")
    
    # Drop business_unit (constant value)
    if 'business_unit' in df.columns:
        df = df.drop(columns=['business_unit'])
    
    return df

def get_model_features(df):
    """Identify feature columns for the model."""
    # Start with raw features
    feature_cols = RAW_FEATURES.copy()
    
    # Add all z-score columns
    z_score_cols = [col for col in df.columns if col.endswith('_z_score')]
    feature_cols.extend(z_score_cols)
    
    # Add peer_deviation_score
    if 'peer_deviation_score' in df.columns:
        feature_cols.append('peer_deviation_score')
    
    # Filter to columns that actually exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Exclude any columns that shouldn't be features
    feature_cols = [col for col in feature_cols if col not in EXCLUDE_COLS]
    
    print(f"Model features ({len(feature_cols)}):")
    print(feature_cols)
    
    return feature_cols

def split_by_user(df):
    """Split data by user to avoid leakage."""
    print("\nSplitting data by user...")
    
    # Get unique users
    unique_users = df['user'].unique()
    print(f"Total unique users: {len(unique_users)}")
    
    # Count malicious users
    malicious_users = df[df['is_malicious'] == 1]['user'].unique()
    print(f"Malicious users: {len(malicious_users)}")
    
    # Use GroupShuffleSplit to split by user
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    # Get train/test indices
    train_idx, test_idx = next(gss.split(df, groups=df['user']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Check malicious distribution
    train_malicious_users = train_df[train_df['is_malicious'] == 1]['user'].unique()
    test_malicious_users = test_df[test_df['is_malicious'] == 1]['user'].unique()
    
    print(f"\nTrain set:")
    print(f"  Users: {train_df['user'].nunique()}")
    print(f"  Rows: {len(train_df):,}")
    print(f"  Malicious users: {len(train_malicious_users)}")
    print(f"  Malicious rows: {train_df['is_malicious'].sum()}")
    
    print(f"\nTest set:")
    print(f"  Users: {test_df['user'].nunique()}")
    print(f"  Rows: {len(test_df):,}")
    print(f"  Malicious users: {len(test_malicious_users)}")
    print(f"  Malicious rows: {test_df['is_malicious'].sum()}")
    
    return train_df, test_df

def train_xgboost_model(train_df, feature_cols):
    """Train XGBoost model with class imbalance handling."""
    print("\nTraining XGBoost model...")
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['is_malicious']
    
    # Compute scale_pos_weight for class imbalance
    # ratio = negative_samples / positive_samples
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    scale_pos_weight = neg_samples / pos_samples
    
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")
    print(f"Negative samples: {neg_samples:,}")
    print(f"Positive samples: {pos_samples:,}")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("Model training complete")
    
    return model

def evaluate_model(model, test_df, feature_cols):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_malicious']
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nMalicious Class Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")
    
    print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    
    print(f"\nAUC Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    # Feature importance
    print(f"\nTop 15 Most Important Features:")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'cm': cm,
        'feature_importance': feature_importance
    }
    
    return metrics

def evaluate_static_baseline(test_df, raw_features):
    """Evaluate static threshold baseline."""
    print("\n" + "="*60)
    print("STATIC BASELINE EVALUATION (95th Percentile Threshold)")
    print("="*60)
    
    y_test = test_df['is_malicious']
    
    # Compute 95th percentile thresholds for each raw feature
    thresholds = {}
    for feat in raw_features:
        if feat in test_df.columns:
            thresholds[feat] = test_df[feat].quantile(0.95)
    
    # Flag as malicious if ANY feature exceeds its threshold
    baseline_pred = np.zeros(len(test_df), dtype=int)
    for feat, threshold in thresholds.items():
        baseline_pred = np.logical_or(baseline_pred, (test_df[feat] > threshold).astype(int)).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, baseline_pred)
    recall = recall_score(y_test, baseline_pred)
    f1 = f1_score(y_test, baseline_pred)
    
    cm = confusion_matrix(y_test, baseline_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC-AUC (use raw feature max as score)
    max_feature_values = test_df[raw_features].max(axis=1)
    roc_auc = roc_auc_score(y_test, max_feature_values)
    pr_auc = average_precision_score(y_test, max_feature_values)
    
    print(f"\nMalicious Class Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")
    
    print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    
    print(f"\nAUC Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    baseline_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return baseline_metrics

def print_comparison_table(model_metrics, baseline_metrics):
    """Print side-by-side comparison."""
    print("\n" + "="*60)
    print("SIDE-BY-SIDE COMPARISON: Model vs Static Baseline")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'XGBoost Model':<15} {'Static Baseline':<15} {'Improvement':<15}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1'),
        ('False Pos Rate', 'fpr'),
        ('ROC-AUC', 'roc_auc'),
        ('PR-AUC', 'pr_auc')
    ]
    
    for name, key in metrics_to_compare:
        model_val = model_metrics[key]
        baseline_val = baseline_metrics[key]
        
        if key == 'fpr':
            # Lower is better for FPR
            improvement = ((baseline_val - model_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvement_str = f"{improvement:+.1f}%"
        else:
            # Higher is better for other metrics
            improvement = ((model_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvement_str = f"{improvement:+.1f}%"
        
        print(f"{name:<20} {model_val:<15.4f} {baseline_val:<15.4f} {improvement_str:<15}")
    
    print("="*60)

def add_risk_scores(test_df, y_pred_proba):
    """Add risk scores and bands to test dataframe."""
    # Risk score: probability * 100, rounded
    test_df['risk_score'] = (y_pred_proba * 100).round(1)
    
    # Risk band
    test_df['risk_band'] = pd.cut(
        test_df['risk_score'],
        bins=[0, 30, 70, 100],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    print(f"\nRisk score distribution:")
    print(test_df['risk_band'].value_counts())
    
    return test_df

def save_outputs(model, test_df):
    """Save model and predictions."""
    print("\nSaving outputs...")
    
    # Save model
    model_file = MODEL_PATH / "xgb_risk_model.pkl"
    joblib.dump(model, model_file)
    print(f"Saved model: {model_file}")
    
    # Save test predictions
    predictions_file = OUTPUT_PATH / "test_predictions.csv"
    test_df.to_csv(predictions_file, index=False)
    print(f"Saved predictions: {predictions_file}")

def main():
    """Main execution function."""
    print("="*60)
    print("XGBOOST RISK SCORING MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get model features
    feature_cols = get_model_features(df)
    
    # Split by user
    train_df, test_df = split_by_user(df)
    
    # Train XGBoost model
    model = train_xgboost_model(train_df, feature_cols)
    
    # Evaluate model
    model_metrics = evaluate_model(model, test_df, feature_cols)
    
    # Evaluate static baseline
    baseline_metrics = evaluate_static_baseline(test_df, RAW_FEATURES)
    
    # Print comparison
    print_comparison_table(model_metrics, baseline_metrics)
    
    # Add risk scores to test set
    test_df = add_risk_scores(test_df, model_metrics['y_pred_proba'])
    
    # Save outputs
    save_outputs(model, test_df)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
