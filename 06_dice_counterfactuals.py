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

def get_feature_ranges(df, feature_cols):
    """Get min/max ranges for features from training data."""
    print("\nComputing feature ranges...")
    
    ranges = {}
    for feat in feature_cols:
        if feat in df.columns:
            ranges[feat] = [df[feat].min(), df[feat].max()]
    
    print("Feature ranges computed")
    return ranges

def setup_dice(model, df, feature_cols, feature_ranges):
    """Set up DiCE explainer."""
    print("\nSetting up DiCE explainer...")
    
    # Prepare data for DiCE (include target column)
    X = df[feature_cols + ['is_malicious']].copy()
    
    # Create DiCE data object
    d = dice_ml.Data(
        dataframe=X,
        continuous_features=feature_cols,
        outcome_name='is_malicious'
    )
    
    # Create DiCE model object
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    
    # Create DiCE explainer
    exp = dice_ml.Dice(d, m, method='genetic')
    
    print("DiCE explainer setup complete")
    return exp, X

def generate_counterfactual_for_row(exp, X, row_idx, original_row, feature_cols, feature_ranges, model):
    """Generate counterfactual for a single row."""
    try:
        # Get the instance (exclude target column for instance)
        instance = X.iloc[[row_idx]][feature_cols].copy()
        
        # Generate counterfactuals
        cf_exp = exp.generate_counterfactuals(
            instance,
            total_CFs=3,
            desired_class="opposite",
            features_to_vary=MUTABLE_FEATURES,
            permitted_range=feature_ranges
        )
        
        # Extract counterfactuals
        cfs = cf_exp.cf_examples_list[0].final_cfs_df
        
        if cfs is None or len(cfs) == 0:
            return None, None
        
        # Get the first counterfactual
        cf = cfs.iloc[0]
        
        # Calculate new prediction using original model
        original_pred = original_row['risk_score']
        new_pred_proba = model.predict_proba(cf[feature_cols].values.reshape(1, -1))[0][1]
        new_pred = (new_pred_proba * 100).round(1)
        
        # Check if prediction actually flipped
        prediction_flipped = (original_row['risk_band'] == 'High' and new_pred < 30)
        
        # Calculate proximity (normalized distance)
        original_values = instance.iloc[0][MUTABLE_FEATURES].values
        cf_values = cf[MUTABLE_FEATURES].values
        ranges_array = np.array([feature_ranges[f][1] - feature_ranges[f][0] for f in MUTABLE_FEATURES])
        ranges_array[ranges_array == 0] = 1  # Avoid division by zero
        proximity = np.mean(np.abs(original_values - cf_values) / ranges_array)
        
        # Calculate sparsity (number of features changed)
        changes = np.abs(original_values - cf_values) > 1e-6
        sparsity = np.sum(changes)
        
        return cf, {
            'prediction_flipped': prediction_flipped,
            'proximity': proximity,
            'sparsity': sparsity,
            'new_risk_score': new_pred
        }
        
    except Exception as e:
        print(f"  Error generating CF for row {row_idx}: {str(e)}")
        return None, None

def translate_counterfactual(original_row, cf, feature_cols, original_risk, new_risk):
    """Translate counterfactual diff to plain English."""
    changes = []
    
    for feat in MUTABLE_FEATURES:
        if feat in original_row.index and feat in cf.index:
            original_val = original_row[feat]
            cf_val = cf[feat]
            
            # Check if significantly different
            if abs(original_val - cf_val) > 1e-6:
                if feat in ['off_hours_ratio', 'avg_attachment_size']:
                    # Format as percentage or decimal
                    changes.append(f"{feat} from {original_val:.2f} to {cf_val:.2f}")
                else:
                    # Format as integer
                    changes.append(f"{feat} from {int(original_val)} to {int(cf_val)}")
    
    if len(changes) == 0:
        return None
    
    # Build explanation sentence
    if len(changes) == 1:
        explanation = f"Reducing {changes[0]} would lower this user's risk score from {original_risk:.1f} to {new_risk:.1f}."
    elif len(changes) == 2:
        explanation = f"Reducing {changes[0]} and {changes[1]} would lower this user's risk score from {original_risk:.1f} to {new_risk:.1f}."
    else:
        changes_str = ", ".join(changes[:-1]) + ", and " + changes[-1]
        explanation = f"Reducing {changes_str} would lower this user's risk score from {original_risk:.1f} to {new_risk:.1f}."
    
    return explanation

def generate_counterfactuals(model, df, feature_cols):
    """Generate counterfactuals for high-risk rows."""
    print("\nGenerating counterfactuals for High-risk rows...")
    
    # Get feature ranges
    feature_ranges = get_feature_ranges(df, feature_cols)
    
    # Set up DiCE
    exp, X = setup_dice(model, df, feature_cols, feature_ranges)
    
    # Get High-risk rows
    high_risk_rows = df[df['risk_band'] == 'High'].head(20)
    print(f"Processing {len(high_risk_rows)} High-risk rows (capped at 20)")
    
    # Initialize columns
    df['dice_explanation'] = None
    
    # Quality metrics
    valid_cfs = 0
    total_cfs = 0
    proximities = []
    sparsities = []
    
    # Generate CFs for each row
    for i, (idx, row) in enumerate(high_risk_rows.iterrows()):
        print(f"  Processing row {i+1}/{len(high_risk_rows)} (user: {row['user']})")
        
        # Find corresponding index in X
        row_idx = df.index.get_loc(idx)
        
        # Generate counterfactual
        cf, metrics = generate_counterfactual_for_row(
            exp, X, row_idx, row, feature_cols, feature_ranges, model
        )
        
        total_cfs += 1
        
        if cf is not None:
            # Translate to explanation
            explanation = translate_counterfactual(
                row, cf, feature_cols, row['risk_score'], metrics['new_risk_score']
            )
            
            if explanation:
                df.loc[idx, 'dice_explanation'] = explanation
                valid_cfs += 1
                proximities.append(metrics['proximity'])
                sparsities.append(metrics['sparsity'])
                
                if metrics['prediction_flipped']:
                    valid_cfs += 1  # Count as valid if prediction flipped
        else:
            print(f"  Failed to generate CF for row {i+1}")
    
    # Print quality metrics
    print("\n" + "="*60)
    print("COUNTERFACTUAL QUALITY METRICS")
    print("="*60)
    
    validity = (valid_cfs / total_cfs * 100) if total_cfs > 0 else 0
    avg_proximity = np.mean(proximities) if proximities else 0
    avg_sparsity = np.mean(sparsities) if sparsities else 0
    
    print(f"\nTotal counterfactuals attempted: {total_cfs}")
    print(f"Valid counterfactuals: {valid_cfs}")
    print(f"Validity (% that flip prediction): {validity:.2f}%")
    print(f"Average proximity (normalized distance): {avg_proximity:.4f}")
    print(f"Average sparsity (features changed): {avg_sparsity:.2f}")
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
    
    # Save final predictions
    output_file = OUTPUT_PATH / "final_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved final predictions: {output_file}")
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL EXPLANATIONS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
