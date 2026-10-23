"""
Ensemble Risk Scoring Model Training
Tests multiple models and creates an ensemble for better predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
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

def train_single_models(train_df, feature_cols):
    """Train individual models and return them."""
    print("\n" + "="*60)
    print("TRAINING INDIVIDUAL MODELS")
    print("="*60)
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_malicious']
    
    # Compute class weights
    neg_samples = (y_train == 0).sum()
    pos_samples = (y_train == 1).sum()
    class_weight = {0: 1, 1: 3}  # Give 3x weight to positive class
    
    print(f"Class imbalance: {neg_samples/pos_samples:.2f}:1")
    print(f"Using class weight: {class_weight}")
    
    models = {}
    
    # Model 1: XGBoost (powerful gradient boosting)
    print("\nTraining XGBoost...")
    models['xgb'] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0.5
    )
    models['xgb'].fit(X_train, y_train)
    print("XGBoost trained")
    
    # Model 2: Random Forest (powerful ensemble)
    print("\nTraining Random Forest...")
    models['rf'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    models['rf'].fit(X_train, y_train)
    print("Random Forest trained")
    
    # Model 3: Logistic Regression (simple, interpretable)
    print("\nTraining Logistic Regression...")
    models['lr'] = LogisticRegression(
        class_weight=class_weight,
        random_state=42,
        max_iter=1000,
        C=1.0,
        penalty='l2',
        solver='lbfgs'
    )
    models['lr'].fit(X_train, y_train)
    print("Logistic Regression trained")
    
    return models

def evaluate_single_models(models, test_df, feature_cols):
    """Evaluate each individual model."""
    print("\n" + "="*60)
    print("EVALUATING INDIVIDUAL MODELS")
    print("="*60)
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_malicious']
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name.upper()} Model:")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Use threshold of 0.20 (20% risk score) to match High Risk band
        y_pred = (y_pred_proba >= 0.20).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  False Positive Rate: {fpr:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        
        results[name] = {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    return results

def create_ensemble(models, train_df, feature_cols):
    """Create a weighted ensemble of models."""
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_malicious']
    
    # Create voting classifier with soft voting
    # Give more weight to XGBoost and Random Forest
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', models['xgb']),
            ('rf', models['rf']),
            ('lr', models['lr'])
        ],
        voting='soft',
        weights=[3, 2, 1]  # XGBoost highest weight, LR lowest
    )
    
    print("Training ensemble (voting classifier)...")
    ensemble.fit(X_train, y_train)
    print("Ensemble trained")
    
    return ensemble

def evaluate_ensemble(ensemble, test_df, feature_cols):
    """Evaluate ensemble model."""
    print("\n" + "="*60)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*60)
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_malicious']
    
    # Get predictions
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    # Use threshold of 0.20 (20% risk score) to match High Risk band
    y_pred = (y_pred_proba >= 0.20).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nEnsemble Model Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    
    return {
        'model': ensemble,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def add_risk_scores(test_df, y_pred_proba):
    """Add risk scores and bands to test dataframe."""
    # Risk score: probability * 100, rounded
    test_df['risk_score'] = (y_pred_proba * 100).round(1)

    # Risk band - Use more appropriate thresholds based on actual distribution
    # 0-5: Low (covers 99% of normal users)
    # 5-20: Medium (elevated but not clearly malicious)
    # 20-100: High (clearly suspicious, catches 75% of malicious users)
    test_df['risk_band'] = pd.cut(
        test_df['risk_score'],
        bins=[0, 5, 20, 100],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    print(f"\nRisk score distribution:")
    print(test_df['risk_band'].value_counts())

    return test_df

def save_outputs(ensemble, test_df):
    """Save model and predictions."""
    print("\nSaving outputs...")
    
    # Save ensemble model
    model_file = MODEL_PATH / "ensemble_risk_model.pkl"
    joblib.dump(ensemble, model_file)
    print(f"Saved ensemble model: {model_file}")
    
    # Also save as xgb_risk_model.pkl for compatibility with existing scripts
    compat_file = MODEL_PATH / "xgb_risk_model.pkl"
    joblib.dump(ensemble, compat_file)
    print(f"Saved compatible model: {compat_file}")
    
    # Save test predictions
    predictions_file = OUTPUT_PATH / "test_predictions.csv"
    test_df.to_csv(predictions_file, index=False)
    print(f"Saved predictions: {predictions_file}")

def main():
    """Main execution function."""
    print("="*60)
    print("ENSEMBLE RISK SCORING MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get model features
    feature_cols = get_model_features(df)
    
    # Split by user
    train_df, test_df = split_by_user(df)
    
    # Train individual models
    models = train_single_models(train_df, feature_cols)
    
    # Evaluate individual models
    single_results = evaluate_single_models(models, test_df, feature_cols)
    
    # Create ensemble
    ensemble = create_ensemble(models, train_df, feature_cols)
    
    # Evaluate ensemble
    ensemble_results = evaluate_ensemble(ensemble, test_df, feature_cols)
    
    # Add risk scores using ensemble
    test_df = add_risk_scores(test_df, ensemble_results['y_pred_proba'])
    
    # Save outputs
    save_outputs(ensemble, test_df)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
