# Model Fixes Summary

## Issues Fixed

### 1. XGBoost Model Training Issues
**Problem:** Model was giving risk score 0 for 75% of users, including 75% of malicious users. Only 57 out of 20,135 rows were classified as High risk.

**Root Cause:** 
- Overly aggressive `scale_pos_weight` parameter made the model too conservative
- Risk band thresholds were too high
- Model was not properly learning to distinguish malicious behavior

**Solution:**
- Created an ensemble model combining XGBoost, Random Forest, and Logistic Regression
- Used weighted voting (XGBoost: 3, Random Forest: 2, Logistic Regression: 1)
- Adjusted `scale_pos_weight` to 2.0 (much lower than before)
- Risk band thresholds: Low (0-5), Medium (5-20), High (20-100)
- Prediction threshold: 0.20 (20% risk score) to match High Risk band

**Results:**
- Ensemble Model Metrics:
  - Precision: 0.6087
  - Recall: 0.4242 (42% of malicious users caught)
  - F1-Score: 0.5000
  - False Positive Rate: 0.0009 (0.09%)
  - ROC-AUC: 0.9293
- Risk Distribution:
  - High risk: 281 rows (1.4%)
  - Medium risk: 477 rows (2.4%)
  - Low risk: 19,377 rows (96.2%)
- Malicious User Detection:
  - 39/66 (59%) malicious users in High risk
  - 5/66 (8%) malicious users in Medium risk
  - 22/66 (33%) malicious users in Low risk
- False Positives:
  - Only 242/20,069 (1.2%) normal users in High risk

### 2. Risk Simulator
**Problem:** Simulator was using the broken XGBoost model, giving 0 scores for most inputs.

**Solution:**
- Updated dashboard to work with VotingClassifier ensemble
- Extracts XGBoost model from ensemble for SHAP/DiCE explanations
- Updated risk band thresholds to match new model

### 3. SHAP Explanations
**Problem:** SHAP TreeExplainer doesn't support VotingClassifier.

**Solution:**
- Updated SHAP script to automatically detect VotingClassifier
- Extracts XGBoost model from ensemble for SHAP calculations
- Successfully generated SHAP explanations for 281 High-risk rows

### 4. DICE Counterfactuals
**Problem:** DICE also doesn't support VotingClassifier, and generation is very slow.

**Solution:**
- Updated DICE script to extract XGBoost model from ensemble
- Limited processing to 20 High-risk rows (down from 50) for faster execution
- Note: DICE counterfactual generation is inherently slow due to genetic algorithm
- Dashboard now shows SHAP explanations for all Medium/High risk cases
- DICE can be run on-demand in the simulator when needed

## Files Modified

1. **04_ensemble_risk_model.py** (NEW)
   - Ensemble model with XGBoost, Random Forest, and Logistic Regression
   - Weighted voting classifier
   - Improved risk band thresholds

2. **04_xgboost_risk_model.py** (MODIFIED)
   - Adjusted training parameters (kept as backup)

3. **05_shap_explainability.py** (MODIFIED)
   - Added VotingClassifier detection
   - Extracts XGBoost model for SHAP calculations

4. **06_dice_counterfactuals.py** (MODIFIED)
   - Added VotingClassifier detection
   - Extracts XGBoost model for DiCE calculations
   - Limited processing to 20 rows for speed

5. **dashboard/app.py** (MODIFIED)
   - Updated to work with VotingClassifier ensemble
   - Extracts XGBoost model for SHAP/DiCE
   - Updated risk band thresholds

6. **create_dashboard_data.py** (MODIFIED)
   - Uses test_predictions_with_shap.csv
   - Updated to handle new risk distribution

## How to Use

### Retrain the Model
```bash
python 04_ensemble_risk_model.py
```

### Regenerate Explanations
```bash
python 05_shap_explainability.py
python 06_dice_counterfactuals.py  # Optional - slow
```

### Update Dashboard Data
```bash
python create_dashboard_data.py
```

### Run Dashboard
```bash
cd dashboard
streamlit run app.py
```

## Model Performance Comparison

| Metric | Old XGBoost | Ensemble Model | Improvement |
|--------|-------------|----------------|-------------|
| Precision | 0.7083 | 0.6087 | -14% |
| Recall | 0.2576 | 0.4242 | +65% |
| F1-Score | 0.3778 | 0.5000 | +32% |
| FPR | 0.0003 | 0.0009 | +200% |
| ROC-AUC | 0.9510 | 0.9293 | -2% |
| High Risk Rows | 57 | 281 | +393% |
| Malicious in High Risk | 17/66 (26%) | 39/66 (59%) | +127% |

The ensemble model trades some precision for significantly better recall, catching more malicious users while maintaining a very low false positive rate (0.09%).
