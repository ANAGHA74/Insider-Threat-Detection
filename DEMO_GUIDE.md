# Peer-Cohort Baselining: How It Works & Demo Guide

## **What is Peer-Cohort Baselining?**

### **The Problem with Traditional Approaches**
Traditional insider threat detection uses one-size-fits-all thresholds:
- "Flag anyone with >50 logons per day"
- "Alert if email attachments >10"

**This fails because:**
- IT staff naturally have more file access than HR
- Sales staff send more emails than engineers
- Different roles have different "normal" behavior patterns

### **Our Solution: Peer-Cohort Baselining**
Instead of universal thresholds, we compare each employee to **their own peer group**:
- Same role (e.g., all Salesmen)
- Same department
- Same time period (rolling weekly baseline)

**This works because:**
- A Salesman with 75 emails/week might be normal (if peers average 70)
- But an Engineer with 75 emails/week might be suspicious (if peers average 20)

---

## **How It Works - Step by Step**

### **Step 1: Group Employees into Peer Cohorts**
Employees are grouped by role and department:
- `Salesman` cohort: All employees with role="Salesman"
- `SoftwareEngineer` cohort: All employees with role="SoftwareEngineer"
- etc.

### **Step 2: Compute Rolling Baselines for Each Cohort**
For each cohort, we calculate weekly rolling statistics:
- `logon_count_rolling_mean`: Average logons for Salesmen this week
- `logon_count_rolling_std`: Standard deviation (how much variation is normal)
- Same for all features: off_hours_ratio, usb_events, email_count, etc.

**Example:**
```
Week of 2010-07-12, Salesman cohort:
- Logon Count: Mean=11.9, Std=3.6
- Off-Hours Ratio: Mean=0.45, Std=0.08
- Email Count: Mean=71.2, Std=11.5
```

### **Step 3: Calculate Z-Scores for Each Employee**
For each employee in that week, we calculate how far they deviate from their cohort:

```
z_score = (employee_value - cohort_mean) / cohort_std
```

**Example:**
- Employee BIH0745 (Salesman) has 12 logons
- Salesman cohort mean = 11.9, std = 3.6
- z_score = (12 - 11.9) / 3.6 = 0.03 (very close to normal)

### **Step 4: Compute Peer Deviation Score**
Average of all z-scores for that employee:
```
peer_deviation_score = mean(|z1|, |z2|, |z3|, ...)
```

This tells us: "How unusual is this employee compared to their peers?"

### **Step 5: Train Model on These Features**
The XGBoost model learns from:
- Raw features (actual values)
- Z-scores (deviation from peers)
- Peer deviation score (overall unusualness)

**Key insight:** The model learns that certain combinations of deviations are suspicious.

---

## **Demo Script for Judges**

### **Demo Setup**
1. Open the dashboard
2. Select **"Individual Analysis"** view
3. Choose a user (e.g., BIH0745)
4. Choose a week (e.g., 2010-07-12)

### **Demo Part 1: Show Peer-Cohort Comparison**
**Say:** "Let me show you how we compare this employee to their peer group."

**Show:**
- Look at the **Peer-Cohort Comparison** panel
- Point out: "This employee is a Salesman, so we compare them to other Salesmen"
- Show the bar chart: "You can see their behavior vs the cohort average"
- Explain: "The green bars show the cohort average, blue shows this employee"

### **Demo Part 2: Show Z-Score Analysis**
**Say:** "We calculate z-scores to measure how unusual each behavior is."

**Show:**
- Scroll to the **Z-Score Analysis** section (if visible)
- Explain: "Green means normal, yellow means moderately unusual, red means highly unusual"
- Point out: "This employee has some deviations but not extreme"

### **Demo Part 3: Show Risk Score**
**Say:** "The model combines all these factors to compute a risk score."

**Show:**
- Look at the **Risk Score** gauge
- Explain: "This employee has a 90.1 risk score, which is flagged as High"
- Explain: "This is because their combination of behaviors matches patterns we learned from historical malicious cases"

### **Demo Part 4: Show Explainability**
**Say:** "We can explain WHY the model flagged this employee."

**Show:**
- Look at **SHAP Explanation** panel
- Explain: "These are the top features that contributed to the high risk score"
- Explain: "For example, high off-hours ratio and email attachments are suspicious"

**Show:**
- Look at **DiCE Counterfactual** panel
- Explain: "This tells us what would lower the risk - e.g., if they reduced email attachments"

### **Demo Part 5: Show Ground Truth**
**Say:** "Let's verify if this was actually a malicious scenario."

**Show:**
- Click **"Reveal Ground Truth"**
- Explain: "This confirms it was a malicious scenario - our model was correct!"

### **Demo Part 6: Show a Normal Case**
**Say:** "Let me show you a normal case for comparison."

**Show:**
- Select a different user with low risk score
- Show that their behavior aligns with peer cohort
- Explain: "This employee's behavior is normal for their role, so low risk"

---

## **Key Talking Points for Judges**

### **1. Personalization**
"Our system personalizes risk assessment by comparing each employee to their peer group, not using one-size-fits-all thresholds."

### **2. Reduced False Positives**
"By using peer-cohort baselining, we reduced false positives by 99.6% - from 28.4% to 0.1%. This means we rarely accuse innocent employees."

### **3. Explainability**
"Our system is explainable - we can show exactly which behaviors contributed to the risk score and what would lower it (counterfactuals)."

### **4. Real-World Applicability**
"This approach works in real organizations because different roles have different normal patterns. Our system adapts to these differences automatically."

### **5. Continuous Learning**
"Our rolling baselines update weekly, so the system adapts to changing behavior patterns over time (e.g., during busy seasons)."

---

## **Live Simulator Demo (Optional)**

**Say:** "We also have a simulator that lets us test hypothetical scenarios."

**Show:**
- Switch to **"Live Risk Simulator"** view
- Select a peer cohort
- Click **"Load High-Risk Example"**
- Explain: "This loads values from an actual high-risk case"
- Click **"Calculate Risk"**
- Explain: "You can see it gives a high risk score"
- Adjust sliders and recalculate to show how changes affect risk

**Important Note:** Clarify that the simulator uses a representative baseline (middle of dataset) for simplicity, while the real Individual Analysis uses the exact week's baseline.

---

## **Summary**

**Peer-Cohort Baselining = Personalized Risk Assessment**

- **Groups** employees by role/department
- **Computes** rolling weekly baselines for each group
- **Compares** each employee to their cohort using z-scores
- **Trains** model on raw values + z-scores + deviation scores
- **Results:** 99.6% reduction in false positives, explainable predictions

**For Demo:** Focus on Individual Analysis view - it's the real use case with proper peer-cohort comparison.
