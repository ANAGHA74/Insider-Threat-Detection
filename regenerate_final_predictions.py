"""
Regenerate final_predictions.csv with the new CatBoost ensemble model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Configuration
FEATURES_PATH = Path("data/cleaned/user_weekly_features_with_peer_scores.csv")
OUTPUT_PATH = Path("data/cleaned")
MODEL_PATH = Path("models")

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

def main():
    """Main execution function."""
    print("="*60)
    print("REGENERATING FINAL PREDICTIONS WITH NEW ENSEMBLE MODEL")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Total rows: {len(df):,}")

    # Filter to rows with sufficient baseline
    df = df[df['has_sufficient_baseline'] == True].copy()
    print(f"Rows after filtering: {len(df):,}")

    # Get model features
    feature_cols = get_model_features(df)

    # Load the new ensemble model
    print("\nLoading ensemble model...")
    ensemble = joblib.load(MODEL_PATH / "catboost_ensemble_model.pkl")
    print("Model loaded")

    # Prepare features
    X = df[feature_cols].copy()
    X = X.fillna(0)

    # Get predictions
    print("\nGenerating predictions...")
    y_pred_proba = ensemble.predict_proba(X)[:, 1]

    # Add predictions to dataframe
    df['risk_score'] = (y_pred_proba * 100).round(1)

    # Risk band thresholds
    df['risk_band'] = pd.cut(
        df['risk_score'],
        bins=[0, 5, 20, 100],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    # Add binary prediction at threshold 0.20
    df['predicted_malicious'] = (y_pred_proba >= 0.20).astype(int)

    print(f"\nRisk score distribution:")
    print(df['risk_band'].value_counts())

    print(f"\nPrediction distribution:")
    print(df['predicted_malicious'].value_counts())

    # Save predictions
    output_file = OUTPUT_PATH / "final_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved predictions: {output_file}")

    print("\n" + "="*60)
    print("PREDICTION REGENERATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
