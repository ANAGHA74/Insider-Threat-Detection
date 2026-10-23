# Risk Classification and Explanation Summary

## Overview
This document summarizes the risk classification results and counterfactual explanations generated for the insider threat detection system.

## Key Changes Made

### 1. Improved DICE Explanations
- **Problem**: Original DiCE counterfactuals were slow, often failed, and didn't provide baseline context
- **Solution**: Replaced DiCE with a baseline-aware explanation system that:
  - Identifies features where the user's behavior deviates significantly from their peer cohort baseline
  - Only suggests reducing behaviors that are above baseline (not below)
  - Tests different feature combinations to find one that actually lowers risk
  - Provides context about peer cohort/role baselines

### 2. Risk Classification Table
Created two output files:
- `risk_classification_table.csv`: Complete dataset with all 20,135 user-weeks
- `high_medium_risk_summary.csv`: Focused summary of 281 High/Medium risk cases

## Results Summary

### Risk Distribution
- **High Risk**: 36 user-weeks (0.18%)
- **Medium Risk**: 245 user-weeks (1.22%)
- **Low Risk**: 19,854 user-weeks (98.60%)

### False Positive Analysis
- **High Risk False Positives**: 12 (33% of High Risk cases are non-malicious)
- **Medium Risk False Positives**: 230 (94% of Medium Risk cases are non-malicious)
- **Malicious Users Not Flagged**: 27 (malicious users classified as Low Risk)

### Explanation Quality
- **Total High/Medium Risk Cases**: 281
- **Successful Explanations Generated**: 278 (98.93%)
- **Failed Explanations**: 3 (cases where baseline adjustment didn't lower risk)
- **Average Features Analyzed**: 1.07 per explanation
- **Average Z-Score Deviation**: 3.19 (indicating significant deviation from baseline)

## Example Explanations

### High Risk Example
**User**: EDB0714 (ProductionLineWorker)
**Risk Score**: 90.2
**Explanation**: "To lower risk score from 90.2 to 31.5, reduce email_attachment_count from 6 to 1 (peer baseline) and usb_events from 13 to 2 (peer baseline) to align with ProductionLineWorker group's typical behavior."

### Medium Risk Example
**User**: AVM0947 (ProductionLineWorker)
**Risk Score**: 5.4
**Explanation**: "To lower risk score from 5.4 to 0.1, reduce file_access_count from 118 to 3 (peer baseline) to align with ProductionLineWorker group's typical behavior."

## How Explanations Work

1. **Identify Deviations**: For each high/medium risk user, calculate z-scores for all mutable features
2. **Filter Above-Baseline**: Only consider features where current value > peer baseline
3. **Rank by Deviation**: Sort features by z-score (highest deviation first)
4. **Test Combinations**: Try reducing 1, 2, or 3 features to baseline
5. **Validate Risk Reduction**: Only accept combinations that actually lower the risk score
6. **Generate Explanation**: Create human-readable explanation with baseline context

## Files Generated

1. **data/cleaned/final_predictions.csv** - Complete dataset with risk scores and explanations
2. **data/cleaned/risk_classification_table.csv** - Full risk classification table
3. **data/cleaned/high_medium_risk_summary.csv** - Focused summary of High/Medium risk cases

## Limitations

1. **3 Cases Without Explanations**: Some cases couldn't be explained by simple baseline adjustments, suggesting their risk comes from complex feature interactions or immutable features
2. **High False Positive Rate in Medium Risk**: 94% of Medium Risk cases are non-malicious, suggesting the Medium threshold (5-20) may be too sensitive
3. **Some Malicious Users Missed**: 27 malicious users classified as Low Risk, indicating need for model improvement

## Recommendations

1. **Adjust Medium Risk Threshold**: Consider raising the lower bound from 5 to 10 to reduce false positives
2. **Add Immutable Feature Explanations**: Consider adding explanations for features like logon_count that are currently marked as immutable
3. **Investigate Unexplained Cases**: Review the 3 cases without explanations to understand their risk factors
4. **Ensemble Model Tuning**: Consider adjusting model weights to improve detection of malicious users while maintaining low false positive rate
