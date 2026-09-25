import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# Raw behavioral features
RAW_FEATURES = [
    'logon_count', 'off_hours_ratio', 'usb_events', 'file_access_count',
    'copy_to_removable_count', 'email_count', 'email_attachment_count',
    'avg_attachment_size', 'http_total_count', 'http_upload_count', 'http_download_count'
]

@st.cache_data
def load_dashboard_data():
    """Load dashboard development data with risk scores and explanations."""
    file_path = 'data/cleaned/final_predictions.csv'
    df = pd.read_csv(file_path, low_memory=False)
    df['week'] = pd.to_datetime(df['week'])
    return df

st.set_page_config(
    page_title="Insider Threat Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = load_dashboard_data()

st.sidebar.markdown("### Controls")
st.sidebar.markdown("---")

# Employee and week selector
st.sidebar.markdown("**Employee Selection**")
unique_users = sorted(df['user'].unique())
selected_user = st.sidebar.selectbox("Select Employee", unique_users, key="user_select")

st.sidebar.markdown("**Week Selection**")
# Get weeks for selected user
user_weeks = df[df['user'] == selected_user]['week'].sort_values().unique()
selected_week = st.sidebar.selectbox("Select Week", user_weeks, key="week_select")

# View selector
st.sidebar.markdown("---")
st.sidebar.markdown("**View Mode**")
view_mode = st.sidebar.radio(
    "Select View",
    ["Individual Analysis", "Live Risk Simulator", "Summary & Evaluation"],
    key="view_mode"
)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Custom CSS for styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

/* Apply fonts */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Mono', monospace !important;
}

.stMetricLabel {
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stMarkdownContainer"] > p {
    font-family: 'Inter', sans-serif !important;
}

label {
    font-family: 'Inter', sans-serif !important;
}

/* Color scheme */
.stApp {
    background-color: #0B0F17;
}

[data-testid="stSidebar"] {
    background-color: #131A26;
}

[data-testid="stBlockContainer"] {
    background-color: #131A26;
    border-radius: 8px;
    padding: 0.5rem;
}

[data-testid="stMetricValue"] {
    color: #E4E9F2;
}

[data-testid="stMetricLabel"] {
    color: #6B7A94;
}

h1, h2, h3, h4, h5, h6 {
    color: #E4E9F2;
}

div[data-testid="stMarkdownContainer"] > p {
    color: #E4E9F2;
}

label {
    color: #6B7A94;
}

[data-testid="stSelectbox"] label {
    color: #6B7A94;
}

/* Dossier header styling */
.employee-dossier {
    background-color: #131A26;
    border: 1px solid rgba(107, 122, 148, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

.employee-dossier-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #E4E9F2;
    margin-bottom: 0.25rem;
}

.employee-dossier-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #6B7A94;
}

/* Ground truth box styling */
.ground-truth-sealed {
    background-color: transparent;
    border: 2px dashed rgba(107, 122, 148, 0.5);
    border-radius: 8px;
    padding: 0.75rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #6B7A94;
    transition: all 0.3s ease;
}

.ground-truth-revealed {
    border-radius: 8px;
    padding: 0.75rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
}

.ground-truth-malicious {
    background-color: rgba(248, 113, 113, 0.2);
    border: 1px solid #F87171;
    color: #F87171;
}

.ground-truth-normal {
    background-color: rgba(74, 222, 128, 0.2);
    border: 1px solid #4ADE80;
    color: #4ADE80;
}

/* Reduce vertical spacing */
[data-testid="stMarkdownContainer"] {
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
}

h3 {
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}

h1, h2 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
    gap: 0.5rem !important;
}

[data-testid="stColumn"] {
    padding: 0.25rem !important;
}

/* Reduce top padding of main content */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0.5rem !important;
}

[data-testid="stHeader"] {
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Insider Threat Detection Dashboard")

# Get selected row
selected_row = df[(df['user'] == selected_user) & (df['week'] == selected_week)].iloc[0]

if view_mode == "Individual Analysis":
   
    # Employee Information - Dossier-style header
    st.markdown(f"""
    <div class="employee-dossier">
        <div class="employee-dossier-main">
            {selected_user} • {selected_row['role']}
        </div>
        <div class="employee-dossier-sub">
            {selected_row['department']} • {selected_week.strftime("%Y-%m-%d")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Score Display with Alert
    risk_score = selected_row['risk_score']
    risk_band = selected_row['risk_band']
    
    # Alert Panel
    st.markdown("### Risk Assessment")
    
    if risk_score > 70:
        st.error(f"FLAGGED - High Risk Score: {risk_score:.1f}")
    elif risk_score > 30:
        st.warning(f"Medium Risk Score: {risk_score:.1f}")
    else:
        st.success(f"Low Risk Score: {risk_score:.1f}")
    
    # Risk Score Gauge and Metrics - Three column layout
    st.markdown("### Risk Score & Behavioral Metrics")
    
    col1, col2, col3 = st.columns([0.35, 0.40, 0.25])
    
    with col1:
        st.subheader("Risk Score")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 18, 'color': '#E4E9F2'}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#6B7A94"},
                'bar': {'color': "#E4E9F2", 'thickness': 0.3},
                'bgcolor': "#131A26",
                'borderwidth': 1,
                'bordercolor': "#6B7A94",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(74, 222, 128, 0.3)'},
                    {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#F87171", 'width': 2},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            font={'color': '#E4E9F2'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Behavioral Metrics")
        # Compact grid of metrics
        metric_cols = st.columns(3)
        metrics = [
            ('Logon Count', selected_row['logon_count']),
            ('Off-Hours Ratio', f"{selected_row['off_hours_ratio']:.2f}"),
            ('USB Events', f"{selected_row['usb_events']:.0f}"),
            ('File Access Count', f"{selected_row['file_access_count']:.0f}"),
            ('Email Count', f"{selected_row['email_count']:.0f}"),
            ('HTTP Total Count', f"{selected_row['http_total_count']:.0f}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            col_idx = i % 3
            with metric_cols[col_idx]:
                st.metric(label, value)
    
    with col3:
        st.subheader("Peer-Cohort Comparison")
        st.metric("Peer Group", selected_row['peer_cohort'])
        st.metric("Deviation Score", f"{selected_row['peer_deviation_score']:.4f}")
        
        # Show baseline statistics
        st.markdown("**Cohort Baseline (This Week)**")
        key_features = ['off_hours_ratio', 'usb_events', 'file_access_count']
        
        # Get cohort statistics for the same week
        cohort_data = df[
            (df['peer_cohort'] == selected_row['peer_cohort']) & 
            (df['week'] == selected_row['week'])
        ]
        
        if len(cohort_data) > 0:
            cohort_stats = cohort_data[key_features].agg(['mean', 'std']).round(2)
            
            # Display baseline stats in a compact format
            for feat in key_features:
                if feat in cohort_stats.columns:
                    mean_val = cohort_stats[feat]['mean']
                    std_val = cohort_stats[feat]['std']
                    feat_name = feat.replace('_', ' ').title()
                    st.markdown(f"**{feat_name}**: μ={mean_val}, σ={std_val}")
        
        # Peer cohort comparison chart
        st.markdown("**Behavioral Comparison**")
        
        # Get cohort averages for the same week
        cohort_avg = cohort_data[key_features].mean()
        
        # Prepare comparison data
        comparison_data = []
        for feat in key_features:
            if feat in selected_row.index and feat in cohort_avg.index:
                comparison_data.append({
                    'Feature': feat.replace('_', ' ').title(),
                    'User Value': selected_row[feat],
                    'Cohort Average': cohort_avg[feat]
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df_melted = comparison_df.melt(id_vars=['Feature'], var_name='Type', value_name='Value')
            
            fig = px.bar(
                comparison_df_melted,
                x='Feature',
                y='Value',
                color='Type',
                barmode='group',
                color_discrete_map={'User Value': '#3b82f6', 'Cohort Average': '#10b981'},
                labels={'Value': 'Value', 'Feature': 'Feature'}
            )
            fig.update_layout(height=250, showlegend=True, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Unable to generate peer cohort comparison")
    
    st.markdown("")
    
    # SHAP Explanation Panel - Only show for Medium/High risk
    if risk_score > 30:
        st.markdown("### Why is this risk score high?")
        if pd.notna(selected_row['shap_top_features']):
            try:
                shap_features = json.loads(selected_row['shap_top_features'])
                if shap_features:
                    # Handle both list of lists and list of dicts formats
                    if isinstance(shap_features[0], dict):
                        feature_names = [f['feature'] for f in shap_features]
                        importance_values = [f['value'] for f in shap_features]
                    else:
                        # Fallback for list of lists format
                        feature_names = [f[0] for f in shap_features]
                        importance_values = [f[1] for f in shap_features]
                    
                    # Simplify feature names for judges
                    feature_name_map = {
                        'off_hours_ratio': 'After-hours activity',
                        'off_hours_ratio_z_score': 'Unusual after-hours activity',
                        'email_attachment_count': 'Email attachments sent',
                        'email_attachment_count_z_score': 'Unusual email attachments',
                        'usb_events': 'USB drive usage',
                        'usb_events_z_score': 'Unusual USB drive usage',
                        'file_access_count': 'File access',
                        'file_access_count_z_score': 'Unusual file access',
                        'copy_to_removable_count': 'Files copied to USB',
                        'copy_to_removable_count_z_score': 'Unusual USB file copying',
                        'http_total_count': 'Web activity',
                        'http_total_count_z_score': 'Unusual web activity',
                        'http_upload_count': 'File uploads',
                        'http_upload_count_z_score': 'Unusual file uploads',
                        'peer_deviation_score': 'Overall deviation from peers'
                    }
                    
                    simplified_names = [feature_name_map.get(name, name.replace('_', ' ').title()) for name in feature_names]
                    
                    # Use absolute values for the chart
                    abs_importance = [abs(v) for v in importance_values]
                    
                    # Create a simple bar chart with clear labels
                    fig = px.bar(
                        x=abs_importance,
                        y=simplified_names,
                        orientation='h',
                        labels={'x': 'How much this contributed to risk', 'y': 'Behavior'},
                        color=abs_importance,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=250,
                        showlegend=False,
                        xaxis_title="Contribution to Risk Score",
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add specific explanations with actual values
                    st.markdown("**Top Risk Factors:**")
                    for i, (name, value) in enumerate(zip(simplified_names[:3], importance_values[:3])):
                        # Get actual feature value if available
                        orig_name = feature_names[i]
                        if orig_name in selected_row.index:
                            actual_val = selected_row[orig_name]
                            if isinstance(actual_val, (int, float)):
                                val_str = f"{actual_val:.2f}" if isinstance(actual_val, float) else str(int(actual_val))
                            else:
                                val_str = str(actual_val)
                            
                            # Check if it's a z-score
                            if 'z_score' in orig_name:
                                explanation = f"• **{name}**: Value is {val_str} (deviates significantly from peer group average)"
                            else:
                                explanation = f"• **{name}**: Value is {val_str}"
                        else:
                            explanation = f"• **{name}**: Contributed significantly to risk"
                        
                        st.markdown(explanation)
                else:
                    st.info("No SHAP features available for this case")
            except Exception as e:
                st.error(f"Error parsing SHAP data: {str(e)}")
        else:
            st.info("No SHAP explanation data available for this case")
    
    st.markdown("")
    
    # DiCE Explanation Panel - Only show for Medium/High risk
    if risk_score > 30:
        st.markdown("### What would lower the risk?")
        if pd.notna(selected_row['dice_explanation']):
            st.info(selected_row['dice_explanation'])
            st.markdown("**Interpretation:** If this employee reduced the behaviors mentioned above, their risk score would decrease significantly.")
        else:
            st.info("No DiCE explanation data available for this case")
    
    st.markdown("")
    
    # Ground Truth Reveal Button
    st.markdown("### Ground Truth")
    if 'reveal_ground_truth' not in st.session_state:
        st.session_state.reveal_ground_truth = False
    
    if not st.session_state.reveal_ground_truth:
        # Show sealed box with button
        st.markdown("""
        <div class="ground-truth-sealed">
            Ground truth — sealed
        </div>
        """, unsafe_allow_html=True)
        if st.button("Reveal Ground Truth"):
            st.session_state.reveal_ground_truth = True
            st.rerun()
    else:
        # Show revealed box
        if selected_row['is_malicious'] == 1:
            st.markdown("""
            <div class="ground-truth-revealed ground-truth-malicious">
                CONFIRMED MALICIOUS SCENARIO
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ground-truth-revealed ground-truth-normal">
                CONFIRMED NORMAL ACTIVITY
            </div>
            """, unsafe_allow_html=True)

elif view_mode == "Live Risk Simulator":
    st.markdown("# Live Risk Simulator")
    
    # Load model for predictions
    try:
        import joblib
        import shap
        import dice_ml
        
        model = joblib.load('models/xgb_risk_model.pkl')
        
        # Get feature ranges from data
        feature_ranges = {}
        for feat in RAW_FEATURES:
            if feat in df.columns:
                feature_ranges[feat] = {
                    'min': df[feat].min(),
                    'max': df[feat].max(),
                    'median': df[feat].median()
                }
        
        # Peer cohort selection
        st.markdown("### Peer Cohort Selection")
        unique_cohorts = sorted(df['peer_cohort'].unique())
        selected_cohort = st.selectbox("Select Role/Peer Cohort", unique_cohorts, key="sim_cohort")
        
        # Behavioral feature inputs
        st.markdown("### Behavioral Feature Inputs")
        st.markdown("Adjust the sliders to simulate different behavioral patterns:")
        
        # Check if we should load high risk example
        if 'load_high_risk_example' not in st.session_state:
            st.session_state.load_high_risk_example = False
        
        # Get high risk example values if flag is set
        high_risk_values = {}
        if st.session_state.load_high_risk_example:
            high_risk_case = df[df['risk_band'] == 'High'].iloc[0]
            for feat in RAW_FEATURES:
                if feat in high_risk_case.index:
                    high_risk_values[feat] = high_risk_case[feat]
            st.session_state.load_high_risk_example = False
        
        # Create input fields for each feature
        input_values = {}
        cols = st.columns(3)
        for i, feat in enumerate(RAW_FEATURES):
            col_idx = i % 3
            with cols[col_idx]:
                if feat in feature_ranges:
                    # Use high risk values if available, otherwise use median
                    if feat in high_risk_values:
                        default_val = high_risk_values[feat]
                    else:
                        default_val = feature_ranges[feat]['median']
                    
                    min_val = feature_ranges[feat]['min']
                    max_val = feature_ranges[feat]['max']
                    
                    # Handle case where min==max (add small buffer)
                    if min_val == max_val:
                        min_val = max(0, min_val - 1)
                        max_val = max_val + 1
                    
                    if 'ratio' in feat or 'avg' in feat:
                        # Use number input for decimal values
                        input_val = st.number_input(
                            feat.replace('_', ' ').title(),
                            value=float(default_val),
                            min_value=float(min_val),
                            max_value=float(max_val),
                            step=0.01,
                            key=f"sim_input_{feat}"
                        )
                    else:
                        # Use slider for integer values
                        input_val = st.slider(
                            feat.replace('_', ' ').title(),
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(default_val),
                            key=f"sim_input_{feat}"
                        )
                    input_values[feat] = input_val
        
        # Calculate Risk button
        st.markdown("---")
        if 'sim_calculate_risk' not in st.session_state:
            st.session_state.sim_calculate_risk = False
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Calculate Risk", key="btn_calculate_risk"):
                st.session_state.sim_calculate_risk = True
        with col_btn2:
            if st.button("Load High-Risk Example", key="btn_load_high_risk"):
                # Set flag to load high risk example on next rerun
                st.session_state.load_high_risk_example = True
                st.session_state.sim_calculate_risk = False
                st.rerun()
        
        if st.session_state.sim_calculate_risk:
            with st.spinner("Calculating risk score..."):
                # Get baseline from the middle of the dataset (representative of training distribution)
                # Note: This is a simplified approach for the simulator. In real Individual Analysis,
                # we use the baseline from the exact same week as the selected user.
                cohort_data = df[df['peer_cohort'] == selected_cohort].sort_values('week')
                mid_idx = len(cohort_data) // 2
                baseline_week = cohort_data['week'].iloc[mid_idx]
                baseline = cohort_data[cohort_data['week'] == baseline_week].iloc[0]
                
                st.info(f"Using baseline from week {baseline_week.strftime('%Y-%m-%d')} (representative of training distribution). For real-time analysis, use Individual Analysis view which uses the exact week's baseline.")
                
                # Compute z-scores
                z_scores = {}
                for feat in RAW_FEATURES:
                    if feat in input_values and f'{feat}_rolling_mean' in baseline.index and f'{feat}_rolling_std' in baseline.index:
                        rolling_mean = baseline[f'{feat}_rolling_mean']
                        rolling_std = baseline[f'{feat}_rolling_std']
                        if rolling_std > 0:
                            z_score = (input_values[feat] - rolling_mean) / rolling_std
                            # Cap z-scores to reasonable range (-5 to +5) to stay within training distribution
                            z_scores[f'{feat}_z_score'] = max(-5, min(5, z_score))
                        else:
                            z_scores[f'{feat}_z_score'] = 0.0
                
                # Compute peer deviation score
                peer_deviation_score = np.mean([abs(v) for v in z_scores.values()])
                
                # Construct feature vector in model's expected order
                feature_vector = {}
                for feat in RAW_FEATURES:
                    feature_vector[feat] = input_values[feat]
                feature_vector.update(z_scores)
                feature_vector['peer_deviation_score'] = peer_deviation_score
                
                # Get model's expected feature order
                model_features = model.get_booster().feature_names
                
                # Create input in correct order
                input_row = pd.DataFrame([feature_vector])[model_features]
                
                # Get prediction
                risk_proba_raw = model.predict_proba(input_row)[0][1]
                risk_proba = risk_proba_raw
                risk_score = risk_proba * 100
                
                # Determine risk band
                if risk_score > 70:
                    risk_band = 'High'
                elif risk_score > 30:
                    risk_band = 'Medium'
                else:
                    risk_band = 'Low'
                
                # Display results
                st.markdown("### Risk Assessment Results")
                
                # Alert panel
                if risk_score > 70:
                    st.error(f"FLAGGED - High Risk Score: {risk_score:.1f}")
                elif risk_score > 30:
                    st.warning(f"Medium Risk Score: {risk_score:.1f}")
                else:
                    st.success(f"Low Risk Score: {risk_score:.1f}")
                
                # Risk score gauge
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Risk Score")
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Score", 'font': {'size': 18, 'color': '#E4E9F2'}},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#6B7A94"},
                            'bar': {'color': "#E4E9F2", 'thickness': 0.3},
                            'bgcolor': "#131A26",
                            'borderwidth': 1,
                            'bordercolor': "#6B7A94",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(74, 222, 128, 0.3)'},
                                {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "#F87171", 'width': 2},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(
                        height=250,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=20, b=10),
                        font={'color': '#E4E9F2'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Input Summary")
                    st.metric("Peer Cohort", selected_cohort)
                    st.metric("Peer Deviation Score", f"{peer_deviation_score:.4f}")
                    st.metric("Risk Band", risk_band)
                
                st.markdown("---")
                
                # Show z-scores comparison
                st.markdown("### Z-Score Analysis")
                st.info("Z-scores show how much each feature deviates from the peer cohort baseline (|z| > 2 indicates significant deviation)")
                
                z_score_data = []
                for feat in RAW_FEATURES:
                    if f'{feat}_z_score' in z_scores:
                        z_score_data.append({
                            'Feature': feat.replace('_', ' ').title(),
                            'Input Value': input_values[feat],
                            'Z-Score': z_scores[f'{feat}_z_score']
                        })
                
                if z_score_data:
                    z_df = pd.DataFrame(z_score_data)
                    # Highlight significant deviations
                    def highlight_z_score(val):
                        if abs(val) > 2:
                            return 'background-color: #fee2e2'  # Red for high deviation
                        elif abs(val) > 1:
                            return 'background-color: #fef3c7'  # Yellow for moderate deviation
                        else:
                            return 'background-color: #d1fae5'  # Green for normal
                    
                    z_df_styled = z_df.style.map(highlight_z_score, subset=['Z-Score'])
                    st.dataframe(z_df_styled, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Reset calculate state so user can calculate again
                st.session_state.sim_calculate_risk = False
                
                # SHAP explanation
                st.markdown("### SHAP Explanation")
                if risk_score > 30:
                    st.subheader("Why is this risk score elevated?")
                else:
                    st.subheader("Feature importance for this prediction")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_row)
                    
                    # Get top features
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
                    
                    feature_names = [model_features[i] for i in top_indices]
                    importance_values = [mean_abs_shap[i] for i in top_indices]
                    
                    fig = px.bar(
                        x=importance_values,
                        y=feature_names,
                        orientation='h',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=importance_values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {str(e)}")
                
                # DiCE counterfactual button
                st.markdown("---")
                st.markdown("### DiCE Counterfactual")
                if st.button("Generate Counterfactual", key="generate_cf"):
                    with st.spinner("Generating counterfactual..."):
                        try:
                            # Prepare DiCE data - use a sample of the training data
                            dice_sample = df[model_features + ['is_malicious']].sample(min(1000, len(df)), random_state=42)
                            
                            d = dice_ml.Data(
                                dataframe=dice_sample,
                                continuous_features=model_features,
                                outcome_name='is_malicious'
                            )
                            m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
                            exp = dice_ml.Dice(d, m, method='genetic')
                            
                            # Generate counterfactual
                            cf_exp = exp.generate_counterfactuals(
                                input_row,
                                total_CFs=1,
                                desired_class="opposite",
                                features_to_vary=['off_hours_ratio', 'usb_events', 'file_access_count', 'copy_to_removable_count', 'email_attachment_count', 'avg_attachment_size', 'http_upload_count', 'http_download_count'],
                                permitted_range={feat: [feature_ranges[feat]['min'], feature_ranges[feat]['max']] for feat in RAW_FEATURES if feat in feature_ranges},
                                verbose=False
                            )
                            
                            cfs = cf_exp.cf_examples_list[0].final_cfs_df
                            if cfs is not None and len(cfs) > 0:
                                cf = cfs.iloc[0]
                                new_risk_proba = model.predict_proba(cf[model_features].values.reshape(1, -1))[0][1]
                                new_risk_score = new_risk_proba * 100
                                
                                # Generate explanation
                                changes = []
                                for feat in ['off_hours_ratio', 'usb_events', 'file_access_count', 'copy_to_removable_count', 'email_attachment_count', 'avg_attachment_size', 'http_upload_count', 'http_download_count']:
                                    if feat in input_values and feat in cf.index:
                                        original_val = input_values[feat]
                                        cf_val = cf[feat]
                                        if abs(original_val - cf_val) > 1e-6:
                                            if 'ratio' in feat or 'avg' in feat:
                                                changes.append(f"{feat} from {original_val:.2f} to {cf_val:.2f}")
                                            else:
                                                changes.append(f"{feat} from {int(original_val)} to {int(cf_val)}")
                                
                                if changes:
                                    if len(changes) == 1:
                                        explanation = f"Reducing {changes[0]} would lower this user's risk score from {risk_score:.1f} to {new_risk_score:.1f}."
                                    elif len(changes) == 2:
                                        explanation = f"Reducing {changes[0]} and {changes[1]} would lower this user's risk score from {risk_score:.1f} to {new_risk_score:.1f}."
                                    else:
                                        changes_str = ", ".join(changes[:-1]) + ", and " + changes[-1]
                                        explanation = f"Reducing {changes_str} would lower this user's risk score from {risk_score:.1f} to {new_risk_score:.1f}."
                                    st.info(explanation)
                                else:
                                    st.info("No significant changes needed to lower risk")
                            else:
                                st.info("Could not generate counterfactual - try different input values")
                        except Exception as e:
                            st.error(f"Error generating counterfactual: {str(e)}")
                            st.info("DiCE counterfactual generation can be slow or fail for certain input patterns. Try adjusting the input values.")
    
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.info("Please ensure models/xgb_risk_model.pkl exists and data is available.")


elif view_mode == "Summary & Evaluation":
    st.markdown("# Summary & Evaluation Metrics")
    
    # Load real evaluation results
    try:
        results_df = pd.read_csv('outputs/results_summary.csv')
        
        # Get full model metrics
        full_model = results_df[results_df['Model'] == 'Full Peer-Cohort Model'].iloc[0]
        baseline_model = results_df[results_df['Model'] == 'Static Baseline'].iloc[0]
        
        # Real metrics
        st.markdown("### Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision", f"{full_model['Precision']:.4f}")
        with col2:
            st.metric("Recall", f"{full_model['Recall']:.4f}")
        with col3:
            st.metric("F1 Score", f"{full_model['F1-Score']:.4f}")
        with col4:
            st.metric("FPR", f"{full_model['False Positive Rate']:.4f}")
        
        st.markdown("---")
        
        # FPR Comparison Chart
        st.subheader("False Positive Rate Comparison")
        methods = ['Static Baseline', 'Full Peer-Cohort Model']
        fpr_rates = [baseline_model['False Positive Rate'], full_model['False Positive Rate']]
        
        fig = px.bar(
            x=methods,
            y=fpr_rates,
            labels={'x': 'Method', 'y': 'FPR'},
            color=methods,
            color_discrete_map={'Static Baseline': '#ef4444', 'Full Peer-Cohort Model': '#10b981'},
            text=fpr_rates,
            text_auto='.4f'
        )
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
        fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics table
        st.markdown("---")
        st.subheader("Detailed Results Comparison")
        st.dataframe(
            results_df[['Model', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate', 'ROC-AUC', 'PR-AUC']],
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")
        st.info("Please ensure outputs/results_summary.csv exists from the evaluation script.")

st.markdown("---")
st.caption("Insider Threat Detection Dashboard - Development Version")
