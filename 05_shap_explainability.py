"""
SHAP Explainability Script
Computes SHAP values for XGBoost model and generates explainable insights.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import shap
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = Path("models/xgb_risk_model.pkl")
PREDICTIONS_PATH = Path("data/cleaned/test_predictions.csv")
OUTPUT_PATH = Path("data/cleaned")
PLOTS_PATH = Path("outputs")

# Create output directories
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Raw behavioral features (must match model training script)
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

def load_model_and_data():
    """Load trained model and test predictions."""
    print("Loading model and predictions...")
    
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded predictions: {len(df):,} rows")
    
    return model, df

def get_model_features(df):
    """Identify feature columns (must match model training script)."""
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

def compute_shap_values(model, X):
    """Compute SHAP values using TreeExplainer."""
    print("\nComputing SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print(f"SHAP values computed: shape {shap_values.shape}")
    
    return explainer, shap_values

def extract_top_features_per_row(shap_values, feature_names, top_n=5):
    """Extract top N contributing features for each row."""
    print(f"\nExtracting top {top_n} features per row...")
    
    top_features_list = []
    
    for i in range(len(shap_values)):
        row_shap = shap_values[i]
        
        # Get absolute values and sort
        abs_shap = np.abs(row_shap)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]  # Top N, descending
        
        # Create feature list
        top_features = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_value = float(row_shap[idx])
            top_features.append({
                "feature": feature_name,
                "value": round(shap_value, 4)
            })
        
        top_features_list.append(top_features)
    
    return top_features_list

def add_shap_explanations(df, shap_values, feature_names):
    """Add SHAP explanations to dataframe."""
    print("\nAdding SHAP explanations...")
    
    # Extract top 5 features per row
    top_features_list = extract_top_features_per_row(shap_values, feature_names, top_n=5)
    
    # Convert to JSON strings
    shap_json_list = [json.dumps(features) for features in top_features_list]
    
    # Add to dataframe
    df['shap_top_features'] = shap_json_list
    
    # Only keep explanations for High risk rows
    high_risk_mask = df['risk_band'] == 'High'
    df.loc[~high_risk_mask, 'shap_top_features'] = None
    
    high_risk_count = high_risk_mask.sum()
    print(f"SHAP explanations added for {high_risk_count} High-risk rows")
    
    return df

def print_global_feature_importance(shap_values, feature_names, top_n=10):
    """Print globally most influential features."""
    print("\n" + "="*60)
    print(f"TOP {top_n} GLOBALLY INFLUENTIAL FEATURES")
    print("="*60)
    
    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Sort and get top N
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    print(f"\n{'Feature':<30} {'Mean |SHAP|':<15}")
    print("-" * 45)
    for idx in top_indices:
        feature_name = feature_names[idx]
        mean_shap = mean_abs_shap[idx]
        print(f"{feature_name:<30} {mean_shap:<15.4f}")
    
    print("="*60)

def print_example_explanation(df, feature_names):
    """Print one example of a flagged user's top contributors."""
    print("\n" + "="*60)
    print("EXAMPLE: HIGH-RISK USER EXPLANATION")
    print("="*60)
    
    # Get first High-risk row with SHAP explanation
    high_risk_rows = df[(df['risk_band'] == 'High') & (df['shap_top_features'].notna())]
    
    if len(high_risk_rows) == 0:
        print("No High-risk rows with SHAP explanations found")
        return
    
    example_row = high_risk_rows.iloc[0]
    
    print(f"\nUser: {example_row['user']}")
    print(f"Week: {example_row['week']}")
    print(f"Role: {example_row['role']}")
    print(f"Risk Score: {example_row['risk_score']}")
    print(f"Risk Band: {example_row['risk_band']}")
    print(f"Actual Label: {'Malicious' if example_row['is_malicious'] == 1 else 'Normal'}")
    
    print(f"\nTop 5 Contributing Features:")
    top_features = json.loads(example_row['shap_top_features'])
    for i, feat in enumerate(top_features, 1):
        direction = "increases" if feat['value'] > 0 else "decreases"
        print(f"  {i}. {feat['feature']}: {feat['value']:+.4f} ({direction} risk)")
    
    print("="*60)

def save_summary_plot(explainer, shap_values, X, feature_names):
    """Save SHAP summary plot."""
    print("\nSaving SHAP summary plot...")
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    plot_path = PLOTS_PATH / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved summary plot: {plot_path}")

def main():
    """Main execution function."""
    print("="*60)
    print("SHAP EXPLAINABILITY")
    print("="*60)
    
    # Load model and data
    model, df = load_model_and_data()
    
    # Get feature names
    feature_names = get_model_features(df)
    
    # Prepare feature matrix
    X = df[feature_names]
    
    # Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X)
    
    # Add SHAP explanations to dataframe
    df = add_shap_explanations(df, shap_values, feature_names)
    
    # Print global feature importance
    print_global_feature_importance(shap_values, feature_names, top_n=10)
    
    # Print example explanation
    print_example_explanation(df, feature_names)
    
    # Save summary plot
    save_summary_plot(explainer, shap_values, X, feature_names)
    
    # Save enhanced predictions
    output_file = OUTPUT_PATH / "test_predictions_with_shap.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved predictions with SHAP: {output_file}")
    
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
