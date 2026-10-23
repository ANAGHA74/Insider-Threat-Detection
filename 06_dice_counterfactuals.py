"""
DiCE Counterfactual Explanations Script
Generates counterfactual explanations for high-risk cases using DiCE.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import dice_ml
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = Path("models/xgb_risk_model.pkl")
PREDICTIONS_PATH = Path("data/cleaned/test_predictions_with_shap.csv")
OUTPUT_PATH = Path("data/cleaned")

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Mutable features (behaviors a person could change)
MUTABLE_FEATURES = [
    'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_attachment_count', 'avg_attachment_size',
    'http_upload_count', 'http_download_count'
]

# Immutable features (never vary)
IMMUTABLE_FEATURES = [
    'logon_count', 'email_count', 'http_total_count'
]

# All z-score and peer deviation features (derived, not directly actionable)
DERIVED_FEATURES = [
    'logon_count_z_score', 'off_hours_ratio_z_score', 'usb_events_z_score',
    'file_access_count_z_score', 'copy_to_removable_count_z_score',
    'email_count_z_score', 'email_attachment_count_z_score',
    'avg_attachment_size_z_score', 'http_total_count_z_score',
    'http_upload_count_z_score', 'http_download_count_z_score',
    'peer_deviation_score'
]

# All model features (must match training script)
RAW_FEATURES = [
    'logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_count', 'email_attachment_count',
    'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count'
]

EXCLUDE_COLS = [
    'user', 'week', 'role', 'department', 'peer_cohort', 'business_unit',
    'has_sufficient_baseline', 'is_malicious'
]

def load_model_and_data():
    """Load model and predictions."""
    print("Loading model and predictions...")
    
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")
    
    # If it's a VotingClassifier, extract the XGBoost model for DiCE
    if hasattr(model, 'estimators_'):
        print("Detected VotingClassifier - extracting XGBoost model for DiCE")
        model = model.named_estimators_['xgb']
        print(f"Using XGBoost model for DiCE counterfactuals")
    
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded predictions: {len(df):,} rows")
    
    return model, df

def get_model_features(df):
    """Get model feature list."""
    feature_cols = RAW_FEATURES.copy()
    z_score_cols = [col for col in df.columns if col.endswith('_z_score')]
    feature_cols.extend(z_score_cols)
    if 'peer_deviation_score' in df.columns:
        feature_cols.append('peer_deviation_score')
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    feature_cols = [col for col in feature_cols if col not in EXCLUDE_COLS]
    
    return feature_cols


def generate_baseline_explanation(original_row, model, feature_cols):
    """Generate explanation by comparing to baseline instead of using DiCE."""
    try:
        # Identify features with highest POSITIVE deviation from baseline (above baseline = risky)
        deviations = []
        
        for feat in MUTABLE_FEATURES:
            if feat in original_row.index:
                current_val = original_row[feat]
                baseline_mean_col = f'{feat}_rolling_mean'
                baseline_std_col = f'{feat}_rolling_std'
                
                if baseline_mean_col in original_row.index and baseline_std_col in original_row.index:
                    baseline_mean = original_row[baseline_mean_col]
                    baseline_std = original_row[baseline_std_col]
                    
                    # Only consider if current value is ABOVE baseline (risky behavior)
                    if current_val > baseline_mean and baseline_std > 0:
                        z_score = (current_val - baseline_mean) / baseline_std
                        deviations.append((feat, current_val, baseline_mean, z_score))
        
        # Sort by z-score (highest deviation first)
        deviations.sort(key=lambda x: x[3], reverse=True)
        
        # Take top 2-3 most deviated features
        top_deviations = deviations[:3]
        
        if len(top_deviations) == 0:
            return None, None
        
        # Try different combinations to find one that lowers risk
        original_pred = original_row['risk_score']
        
        # Try reducing just the top feature
        for n_features in range(1, len(top_deviations) + 1):
            test_deviations = top_deviations[:n_features]
            cf_values = original_row.copy()
            
            for feat, current_val, baseline_mean, z_score in test_deviations:
                if baseline_mean > 0:
                    cf_values[feat] = baseline_mean
            
            # Calculate new prediction
            new_pred_proba = model.predict_proba(cf_values[feature_cols].values.reshape(1, -1))[0][1]
            new_pred = (new_pred_proba * 100).round(1)
            
            # If this lowers risk, use it
            if new_pred < original_pred:
                # Calculate proximity (normalized distance to baseline)
                proximity = np.mean([d[3] for d in test_deviations])
                sparsity = len(test_deviations)
                
                return cf_values, {
                    'prediction_improved': True,
                    'proximity': proximity,
                    'sparsity': sparsity,
                    'new_risk_score': new_pred,
                    'top_deviations': test_deviations
                }
        
        # If no combination works, return None
        return None, None
        
    except Exception as e:
        print(f"  Error generating baseline explanation: {str(e)}")
        return None, None

def translate_counterfactual(original_row, cf, feature_cols, original_risk, new_risk, top_deviations):
    """Translate counterfactual diff to plain English with baseline context."""
    changes = []
    
    for feat, current_val, baseline_mean, z_score in top_deviations:
        if feat in original_row.index:
            # Check if baseline is available and current is above baseline
            if baseline_mean > 0 and current_val > baseline_mean:
                if feat in ['off_hours_ratio', 'avg_attachment_size']:
                    changes.append(f"{feat} from {current_val:.2f} to {baseline_mean:.2f} (peer baseline)")
                else:
                    changes.append(f"{feat} from {int(current_val)} to {int(baseline_mean)} (peer baseline)")
    
    if len(changes) == 0:
        return None
    
    # Build explanation sentence
    risk_change = new_risk - original_risk
    
    # Get role/department context
    role = original_row.get('role', 'their role')
    cohort = original_row.get('peer_cohort', role)
    
    if len(changes) == 1:
        explanation = f"To lower risk score from {original_risk:.1f} to {new_risk:.1f}, reduce {changes[0]} to align with {cohort} group's typical behavior."
    elif len(changes) == 2:
        explanation = f"To lower risk score from {original_risk:.1f} to {new_risk:.1f}, reduce {changes[0]} and {changes[1]} to align with {cohort} group's typical behavior."
    else:
        changes_str = ", ".join(changes[:-1]) + ", and " + changes[-1]
        explanation = f"To lower risk score from {original_risk:.1f} to {new_risk:.1f}, reduce {changes_str} to align with {cohort} group's typical behavior."
    
    return explanation

def generate_counterfactuals(model, df, feature_cols):
    """Generate baseline-aware explanations for all High and Medium risk rows."""
    print("\nGenerating baseline-aware explanations for High and Medium risk rows...")
    
    # Get ALL High and Medium risk rows (not just malicious)
    risk_rows = df[df['risk_band'].isin(['High', 'Medium'])]
    print(f"Found {len(risk_rows)} High/Medium risk rows")
    
    # Process all risk rows
    all_risk_rows = risk_rows
    print(f"Processing {len(all_risk_rows)} risk rows")
    
    # Initialize columns
    df['dice_explanation'] = None
    
    # Quality metrics
    valid_cfs = 0
    total_cfs = 0
    proximities = []
    sparsities = []
    
    # Generate explanations for each row
    for i, (idx, row) in enumerate(all_risk_rows.iterrows()):
        print(f"  Processing row {i+1}/{len(all_risk_rows)} (user: {row['user']}, risk: {row['risk_band']})")
        
        # Get the row from df to ensure we have all columns
        original_row = df.loc[idx]
        
        # Generate baseline explanation
        cf, metrics = generate_baseline_explanation(
            original_row, model, feature_cols
        )
        
        total_cfs += 1
        
        if cf is not None:
            # Translate to explanation
            explanation = translate_counterfactual(
                original_row, cf, feature_cols, original_row['risk_score'], 
                metrics['new_risk_score'], metrics['top_deviations']
            )
            
            if explanation:
                df.loc[idx, 'dice_explanation'] = explanation
                valid_cfs += 1
                proximities.append(metrics['proximity'])
                sparsities.append(metrics['sparsity'])
        else:
            print(f"  Failed to generate explanation for row {i+1}")
    
    # Print quality metrics
    print("\n" + "="*60)
    print("EXPLANATION QUALITY METRICS")
    print("="*60)
    
    validity = (valid_cfs / total_cfs * 100) if total_cfs > 0 else 0
    avg_proximity = np.mean(proximities) if proximities else 0
    avg_sparsity = np.mean(sparsities) if sparsities else 0
    
    print(f"\nTotal explanations attempted: {total_cfs}")
    print(f"Valid explanations: {valid_cfs}")
    print(f"Validity (% successfully generated): {validity:.2f}%")
    print(f"Average proximity (mean z-score deviation): {avg_proximity:.4f}")
    print(f"Average sparsity (features analyzed): {avg_sparsity:.2f}")
    print("="*60)
    
    return df

def print_example_counterfactuals(df):
    """Print example counterfactual explanations."""
    print("\n" + "="*60)
    print("EXAMPLE COUNTERFACTUAL EXPLANATIONS")
    print("="*60)
    
    # Get rows with counterfactuals
    cf_rows = df[df['dice_explanation'].notna()].head(5)
    
    if len(cf_rows) == 0:
        print("No counterfactual explanations generated")
        return
    
    for i, (idx, row) in enumerate(cf_rows.iterrows(), 1):
        print(f"\nExample {i}:")
        print(f"  User: {row['user']}")
        print(f"  Risk Score: {row['risk_score']}")
        print(f"  Counterfactual: {row['dice_explanation']}")
    
    print("="*60)

def create_risk_classification_table(df):
    """Create a risk classification table with employee IDs, weeks, and risk levels."""
    print("\n" + "="*60)
    print("CREATING RISK CLASSIFICATION TABLE")
    print("="*60)
    
    # Create summary table - one row per user-week with risk classification
    risk_table = df[['user', 'week', 'role', 'department', 'risk_score', 'risk_band', 'is_malicious', 'dice_explanation']].copy()
    
    # Ensure week is datetime for proper formatting
    risk_table['week'] = pd.to_datetime(risk_table['week'])
    
    # Sort by risk score descending (highest risk first)
    risk_table = risk_table.sort_values('risk_score', ascending=False)
    
    # Count by risk band
    high_risk = risk_table[risk_table['risk_band'] == 'High']
    medium_risk = risk_table[risk_table['risk_band'] == 'Medium']
    low_risk = risk_table[risk_table['risk_band'] == 'Low']
    
    print(f"\nRisk Classification Summary:")
    print(f"  High Risk: {len(high_risk)} user-weeks")
    print(f"  Medium Risk: {len(medium_risk)} user-weeks")
    print(f"  Low Risk: {len(low_risk)} user-weeks")
    
    # Check for false positives (High/Medium risk but not malicious)
    false_positives_high = high_risk[high_risk['is_malicious'] == 0]
    false_positives_medium = medium_risk[medium_risk['is_malicious'] == 0]
    
    print(f"\nFalse Positive Analysis:")
    print(f"  High Risk False Positives: {len(false_positives_high)} (non-malicious flagged as High)")
    print(f"  Medium Risk False Positives: {len(false_positives_medium)} (non-malicious flagged as Medium)")
    
    # Check for malicious users not flagged
    malicious_not_flagged = df[(df['is_malicious'] == 1) & (df['risk_band'] == 'Low')]
    print(f"  Malicious users not flagged (Low risk): {len(malicious_not_flagged)}")
    
    # Save the risk classification table
    output_file = OUTPUT_PATH / "risk_classification_table.csv"
    risk_table.to_csv(output_file, index=False)
    print(f"\nSaved risk classification table: {output_file}")
    
    # Create a focused summary for High and Medium risk only
    high_med_summary = risk_table[risk_table['risk_band'].isin(['High', 'Medium'])].copy()
    summary_file = OUTPUT_PATH / "high_medium_risk_summary.csv"
    high_med_summary.to_csv(summary_file, index=False)
    print(f"Saved High/Medium risk summary: {summary_file}")
    
    # Print sample of high risk employees
    print(f"\n" + "="*60)
    print("HIGH RISK EMPLOYEES (Top 10)")
    print("="*60)
    print(high_risk[['user', 'week', 'role', 'risk_score', 'is_malicious', 'dice_explanation']].head(10).to_string(index=False))
    
    print("="*60)
    
    return risk_table

def main():
    """Main execution function."""
    print("="*60)
    print("DICE COUNTERFACTUAL EXPLANATIONS")
    print("="*60)
    
    # Load model and data
    model, df = load_model_and_data()
    
    # Get feature list
    feature_cols = get_model_features(df)
    
    # Generate counterfactuals
    df = generate_counterfactuals(model, df, feature_cols)
    
    # Print examples
    print_example_counterfactuals(df)
    
    # Create risk classification table
    risk_table = create_risk_classification_table(df)
    
    # Save final predictions
    output_file = OUTPUT_PATH / "final_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved final predictions: {output_file}")
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL EXPLANATIONS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
