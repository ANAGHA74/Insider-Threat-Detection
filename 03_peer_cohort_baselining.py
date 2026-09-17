"""
Peer-Cohort Dynamic Baselining Script
Computes rolling baselines per role group and calculates peer deviation scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
FEATURES_PATH = Path("data/cleaned/user_weekly_features.csv")
OUTPUT_PATH = Path("data/cleaned")

# Behavioral feature columns (exclude metadata and labels)
BEHAVIORAL_FEATURES = [
    'logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_count', 'email_attachment_count',
    'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count'
]

def load_features():
    """Load the weekly features."""
    print("Loading weekly features...")
    df = pd.read_csv(FEATURES_PATH)
    df['week'] = pd.to_datetime(df['week'])
    
    # Sort by user and week for rolling calculations
    df = df.sort_values(['user', 'week'])
    
    print(f"Loaded {len(df):,} user-weeks")
    print(f"Date range: {df['week'].min()} to {df['week'].max()}")
    return df

def create_hybrid_groups(df):
    """Create hybrid peer cohorts by consolidating small roles into broader categories."""
    print("\nCreating hybrid peer cohorts...")
    
    # Role consolidation mapping
    role_mapping = {
        # Keep large roles as individual groups
        'ProductionLineWorker': 'ProductionLineWorker',
        'Technician': 'Technician',
        'Salesman': 'Salesman',
        'SoftwareEngineer': 'SoftwareEngineer',
        'Scientist': 'Scientist',
        'AdministrativeAssistant': 'AdministrativeAssistant',
        'ITAdmin': 'ITAdmin',
        'ComputerScientist': 'ComputerScientist',
        'Mathematician': 'Mathematician',
        
        # Consolidate engineers
        'ElectricalEngineer': 'Engineering',
        'MechanicalEngineer': 'Engineering',
        'MaterialsEngineer': 'Engineering',
        'ChiefEngineer': 'Engineering',
        'HardwareEngineer': 'Engineering',
        'IndustrialEngineer': 'Engineering',
        'SystemsEngineer': 'Engineering',
        'TestEngineer': 'Engineering',
        'FieldServiceEngineer': 'Engineering',
        'Engineer': 'Engineering',
        
        # Consolidate management
        'Manager': 'Management',
        'Director': 'Management',
        'VicePresident': 'Management',
        'President': 'Management',
        'LabManager': 'Management',
        'AssemblySupervisor': 'Management',
        
        # Consolidate technical support
        'ComputerProgrammer': 'TechnicalSupport',
        'ComputerTrainer': 'TechnicalSupport',
        'TechnicalTrainer': 'TechnicalTrainer',
        
        # Consolidate admin/HR
        'HumanResourceSpecialist': 'AdminHR',
        'AdministrativeStaff': 'AdminHR',
        'InstructionalCoordinator': 'AdminHR',
        
        # Consolidate finance/operations
        'Accountant': 'FinanceOps',
        'FinancialAnalyst': 'FinanceOps',
        'PurchasingClerk': 'FinanceOps',
        
        # Consolidate specialized roles
        'Attorney': 'Specialized',
        'Statistician': 'Specialized',
        'SecurityGuard': 'Specialized',
        
        # Consolidate medical
        'Nurse': 'Medical',
        'NursePractitioner': 'Medical',
        'HealthSafetyEngineer': 'Medical',
    }
    
    # Apply mapping
    df['peer_cohort'] = df['role'].map(role_mapping)
    
    # Check for unmapped roles
    unmapped = df[df['peer_cohort'].isna()]['role'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped roles: {unmapped}")
        df['peer_cohort'] = df['peer_cohort'].fillna('Other')
    
    # Print group sizes
    cohort_counts = df['peer_cohort'].value_counts()
    print(f"\nHybrid cohort sizes:")
    print(cohort_counts)
    print(f"\nTotal cohorts: {len(cohort_counts)}")
    print(f"Cohorts with < 500 members: {(cohort_counts < 500).sum()}")
    
    return df, 'peer_cohort'

def compute_rolling_baselines(df, grouping_col):
    """Compute rolling mean and std for each group-week using only prior data."""
    print(f"\nComputing rolling baselines by {grouping_col}...")
    
    # Get all unique weeks in chronological order
    all_weeks = sorted(df['week'].unique())
    
    # Initialize columns for rolling statistics
    for feat in BEHAVIORAL_FEATURES:
        df[f'{feat}_rolling_mean'] = np.nan
        df[f'{feat}_rolling_std'] = np.nan
        df[f'{feat}_z_score'] = np.nan
    
    # Minimum weeks required for stable baseline
    MIN_WEEKS_FOR_BASELINE = 3
    
    # Process each group
    groups = df.groupby(grouping_col)
    total_groups = len(groups)
    
    for group_name, group_df in groups:
        if (group_idx := len(groups) - len(groups) + 1) % 10 == 0 or group_idx == 1:
            print(f"  Processing group {group_idx}/{total_groups}: {group_name}")
        
        # Get group's weeks in chronological order
        group_weeks = sorted(group_df['week'].unique())
        
        for week_idx, current_week in enumerate(group_weeks):
            # Get data from current week and all prior weeks (no lookahead)
            prior_weeks = [w for w in group_weeks if w <= current_week]
            
            # Get data for baseline computation
            baseline_data = group_df[group_df['week'].isin(prior_weeks)]
            
            # Skip if insufficient data
            if len(baseline_data) < MIN_WEEKS_FOR_BASELINE:
                continue
            
            # Compute rolling statistics for each feature
            for feat in BEHAVIORAL_FEATURES:
                if feat not in baseline_data.columns:
                    continue
                
                values = baseline_data[feat].values
                rolling_mean = np.mean(values)
                rolling_std = np.std(values, ddof=1)  # sample std
                
                # Avoid division by zero
                if rolling_std < 1e-6:
                    rolling_std = 1.0
                
                # Update rolling statistics for this group-week
                mask = (df[grouping_col] == group_name) & (df['week'] == current_week)
                df.loc[mask, f'{feat}_rolling_mean'] = rolling_mean
                df.loc[mask, f'{feat}_rolling_std'] = rolling_std
    
    print(f"Rolling baselines computed")
    return df

def compute_z_scores(df):
    """Compute z-scores for each feature against rolling baseline."""
    print("\nComputing z-scores...")
    
    for feat in BEHAVIORAL_FEATURES:
        mean_col = f'{feat}_rolling_mean'
        std_col = f'{feat}_rolling_std'
        z_col = f'{feat}_z_score'
        
        # Compute z-score where baseline exists
        valid_baseline = df[mean_col].notna() & df[std_col].notna()
        df.loc[valid_baseline, z_col] = (
            (df.loc[valid_baseline, feat] - df.loc[valid_baseline, mean_col]) / 
            df.loc[valid_baseline, std_col]
        )
    
    # Count how many rows have valid z-scores
    valid_z_rows = df[[f'{feat}_z_score' for feat in BEHAVIORAL_FEATURES]].notna().all(axis=1).sum()
    print(f"Valid z-scores computed for {valid_z_rows:,} rows")
    
    return df

def compute_peer_deviation_score(df):
    """Compute overall peer deviation score as mean of absolute z-scores."""
    print("\nComputing peer deviation scores...")
    
    # Get z-score columns that exist
    z_cols = [f'{feat}_z_score' for feat in BEHAVIORAL_FEATURES if f'{feat}_z_score' in df.columns]
    
    # Compute mean of absolute z-scores across all features
    df['peer_deviation_score'] = df[z_cols].abs().mean(axis=1)
    
    # Mark rows with insufficient baseline data
    df['has_sufficient_baseline'] = df[z_cols].notna().all(axis=1)
    
    print(f"Peer deviation scores computed")
    print(f"Rows with sufficient baseline: {df['has_sufficient_baseline'].sum():,}")
    print(f"Rows with insufficient baseline: {(~df['has_sufficient_baseline']).sum():,}")
    
    return df

def print_validation_report(df, grouping_col):
    """Print validation report."""
    print(f"\n{'='*60}")
    print("VALIDATION REPORT: Peer-Cohort Baselining")
    print(f"{'='*60}")
    
    # Group sizes
    print(f"\nGroup sizes by {grouping_col}:")
    group_sizes = df[grouping_col].value_counts()
    print(f"Total groups: {len(group_sizes)}")
    print(f"Mean group size: {group_sizes.mean():.1f}")
    print(f"Min group size: {group_sizes.min()}")
    print(f"Max group size: {group_sizes.max()}")
    print(f"\nAll groups by size:")
    print(group_sizes)
    
    # Example baseline shift over weeks (FIXED: show unique weeks)
    print(f"\nExample: Baseline shift for largest group")
    largest_group = group_sizes.index[0]
    group_data = df[df[grouping_col] == largest_group].sort_values('week')
    
    # Get unique weeks with their baseline values
    unique_weeks = group_data.drop_duplicates('week')[['week', 'logon_count_rolling_mean', 'logon_count_rolling_std']].head(5)
    
    if len(unique_weeks) >= 1:
        print(f"Group: {largest_group}")
        print(f"\nWeek progression for logon_count baseline:")
        for _, row in unique_weeks.iterrows():
            print(f"  Week {row['week'].strftime('%Y-%m-%d')}: "
                  f"mean={row['logon_count_rolling_mean']:.2f}, "
                  f"std={row['logon_count_rolling_std']:.2f}")
    
    # Overall distribution of peer_deviation_score
    print(f"\nPeer deviation score distribution:")
    valid_scores = df[df['has_sufficient_baseline']]['peer_deviation_score']
    print(f"Mean: {valid_scores.mean():.3f}")
    print(f"Median: {valid_scores.median():.3f}")
    print(f"Std: {valid_scores.std():.3f}")
    print(f"Min: {valid_scores.min():.3f}")
    print(f"Max: {valid_scores.max():.3f}")
    print(f"25th percentile: {valid_scores.quantile(0.25):.3f}")
    print(f"75th percentile: {valid_scores.quantile(0.75):.3f}")
    
    # Critical comparison: malicious vs normal
    print(f"\n{'='*60}")
    print("CRITICAL: Malicious vs Normal Peer Deviation Scores")
    print(f"{'='*60}")
    
    malicious_scores = df[df['is_malicious'] == 1]['peer_deviation_score']
    normal_scores = df[df['is_malicious'] == 0]['peer_deviation_score']
    
    print(f"\nMalicious rows (is_malicious=1):")
    print(f"  Count: {len(malicious_scores)}")
    print(f"  Mean peer deviation: {malicious_scores.mean():.3f}")
    print(f"  Median peer deviation: {malicious_scores.median():.3f}")
    
    print(f"\nNormal rows (is_malicious=0):")
    print(f"  Count: {len(normal_scores)}")
    print(f"  Mean peer deviation: {normal_scores.mean():.3f}")
    print(f"  Median peer deviation: {normal_scores.median():.3f}")
    
    # Statistical test
    if len(malicious_scores) > 0 and len(normal_scores) > 0:
        t_stat, p_value = stats.ttest_ind(
            malicious_scores.dropna(), 
            normal_scores.dropna(),
            equal_var=False
        )
        print(f"\nStatistical test (Welch's t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result: NOT significant (p >= 0.05)")
    
    # Sample rows
    print(f"\n{'='*60}")
    print("Sample rows (with peer scores):")
    print(f"{'='*60}")
    sample_cols = ['user', 'week', grouping_col, 'is_malicious', 'peer_deviation_score', 'has_sufficient_baseline']
    sample_cols += [f'{feat}_z_score' for feat in BEHAVIORAL_FEATURES[:3]]  # Show first 3 feature z-scores
    print(df[df['has_sufficient_baseline']][sample_cols].head(5))
    
    print(f"{'='*60}\n")

def main():
    """Main execution function."""
    print("="*60)
    print("PEER-COHORT DYNAMIC BASELINING")
    print("="*60)
    
    # Load features
    df = load_features()
    
    # Create hybrid peer cohorts
    df, grouping_col = create_hybrid_groups(df)
    
    # Compute rolling baselines
    df = compute_rolling_baselines(df, grouping_col)
    
    # Compute z-scores
    df = compute_z_scores(df)
    
    # Compute peer deviation score
    df = compute_peer_deviation_score(df)
    
    # Print validation report
    print_validation_report(df, grouping_col)
    
    # Save results
    print("Saving results...")
    output_file = OUTPUT_PATH / "user_weekly_features_with_peer_scores.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    print("\n" + "="*60)
    print("PEER-COHORT BASELINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
