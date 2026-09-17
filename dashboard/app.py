import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time

# =============================================================================
# DATA LOADING FUNCTION (Single point of entry - easy to swap file path later)
# =============================================================================
@st.cache_data
def load_dashboard_data():
    """Load dashboard development data with risk scores and explanations."""
    file_path = 'data/cleaned/final_predictions.csv'
    df = pd.read_csv(file_path, low_memory=False)
    df['week'] = pd.to_datetime(df['week'])
    return df

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        text-align: left;
        padding: 20px 24px;
        background: white;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #3b82f6;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 24px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Insider Threat Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD DATA
# =============================================================================
df = load_dashboard_data()

# =============================================================================
# SIDEBAR - CONTROLS
# =============================================================================
st.sidebar.markdown("### Controls")
st.sidebar.markdown("---")

# Employee and week selector
st.sidebar.markdown("**Employee Selection**")
unique_users = sorted(df['user'].unique())
selected_user = st.sidebar.selectbox("Select Employee", unique_users, key="user_select")

# Reset auto-replay when user changes
if 'last_selected_user' not in st.session_state:
    st.session_state.last_selected_user = selected_user
elif st.session_state.last_selected_user != selected_user:
    st.session_state.last_selected_user = selected_user
    st.session_state.auto_replay_week_index = 0
    st.session_state.last_replay_time = time.time()

st.sidebar.markdown("**Week Selection**")
# Get weeks for selected user
user_weeks = df[df['user'] == selected_user]['week'].sort_values().unique()

# Replay mechanism
st.sidebar.markdown("---")
st.sidebar.markdown("**Replay Mode**")
auto_replay = st.sidebar.checkbox("Enable Auto-Replay", key="auto_replay")
replay_speed = st.sidebar.slider("Speed (seconds)", 1, 5, 2, key="replay_speed")

# Initialize session state for replay
if 'auto_replay_week_index' not in st.session_state:
    st.session_state.auto_replay_week_index = 0
if 'last_replay_time' not in st.session_state:
    st.session_state.last_replay_time = time.time()

if auto_replay:
    # Show current week as display (not selectable)
    current_week = user_weeks[st.session_state.auto_replay_week_index]
    st.sidebar.metric("Current Week", current_week.strftime("%Y-%m-%d"))
    
    # Auto-advance based on time
    if time.time() - st.session_state.last_replay_time >= replay_speed:
        if st.session_state.auto_replay_week_index < len(user_weeks) - 1:
            st.session_state.auto_replay_week_index += 1
            st.session_state.last_replay_time = time.time()
            st.rerun()
    
    selected_week = user_weeks[st.session_state.auto_replay_week_index]
else:
    # Manual week selection
    selected_week = st.sidebar.selectbox("Select Week", user_weeks, key="week_select")
    # Reset auto-replay index when switching to manual mode
    st.session_state.auto_replay_week_index = list(user_weeks).index(selected_week)

# View selector
st.sidebar.markdown("---")
st.sidebar.markdown("**View Mode**")
view_mode = st.sidebar.radio(
    "Select View",
    ["Individual Analysis", "Risk Overview", "Summary & Evaluation"],
    key="view_mode"
)

# =============================================================================
# MAIN CONTENT
# =============================================================================
st.markdown("# Insider Threat Detection Dashboard")
st.markdown("---")

# Get selected row
selected_row = df[(df['user'] == selected_user) & (df['week'] == selected_week)].iloc[0]

if view_mode == "Individual Analysis":
    # =============================================================================
    # INDIVIDUAL ANALYSIS VIEW
    # =============================================================================
    
    # Header with employee info
    st.markdown("### Employee Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Employee", selected_user)
    with col2:
        st.metric("Week", selected_week.strftime("%Y-%m-%d"))
    with col3:
        st.metric("Role", selected_row['role'])
    with col4:
        st.metric("Department", selected_row['department'])
    
    st.markdown("---")
    
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
    
    # Risk Score Gauge and Metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Score")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 18}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#6b7280"},
                'bar': {'color': "#2563eb", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 30], 'color': '#d1fae5'},
                    {'range': [30, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 2},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Behavioral Metrics")
        metrics_data = {
            'Logon Count': selected_row['logon_count'],
            'Off-Hours Ratio': f"{selected_row['off_hours_ratio']:.2f}",
            'USB Events': f"{selected_row['usb_events']:.0f}",
            'File Access Count': f"{selected_row['file_access_count']:.0f}",
            'Email Count': f"{selected_row['email_count']:.0f}",
            'HTTP Total Count': f"{selected_row['http_total_count']:.0f}"
        }
        
        for metric, value in metrics_data.items():
            st.metric(metric, value)
    
    st.markdown("---")
    
    # Peer-Cohort Comparison Panel
    st.markdown("### Peer-Cohort Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Peer Group", selected_row['peer_cohort'])
    with col2:
        st.metric("Deviation from Peer Baseline", f"{selected_row['peer_deviation_score']:.4f}")
    
    # Peer cohort comparison chart
    st.subheader("Behavioral Comparison with Peer Cohort")
    key_features = ['off_hours_ratio', 'usb_events', 'file_access_count']
    
    # Get cohort averages for the same week
    cohort_avg = df[
        (df['peer_cohort'] == selected_row['peer_cohort']) & 
        (df['week'] == selected_row['week'])
    ][key_features].mean()
    
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
        fig.update_layout(height=300, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Unable to generate peer cohort comparison")
    
    st.markdown("---")
    
    # SHAP Explanation Panel
    st.markdown("### SHAP Explanation")
    if pd.notna(selected_row['shap_top_features']):
        st.subheader("Why is this risk score high?")
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
            else:
                st.info("No SHAP features available for this case")
        except Exception as e:
            st.error(f"Error parsing SHAP data: {str(e)}")
    else:
        st.info("No SHAP explanation data available for this case")
    
    st.markdown("---")
    
    # DiCE Explanation Panel
    st.markdown("### DiCE Counterfactual")
    if pd.notna(selected_row['dice_explanation']):
        st.subheader("What would lower the risk?")
        st.info(selected_row['dice_explanation'])
    else:
        st.info("No DiCE explanation data available for this case")
    
    st.markdown("---")
    
    # Ground Truth Reveal Button
    st.markdown("### Ground Truth")
    if 'reveal_ground_truth' not in st.session_state:
        st.session_state.reveal_ground_truth = False
    
    if st.button("Reveal Ground Truth"):
        st.session_state.reveal_ground_truth = True
    
    if st.session_state.reveal_ground_truth:
        if selected_row['is_malicious'] == 1:
            st.error("Confirmed Malicious Scenario")
        else:
            st.success("Confirmed Normal Activity")

elif view_mode == "Risk Overview":
    # =============================================================================
    # RISK OVERVIEW VIEW
    # =============================================================================
    
    st.markdown("# Risk Distribution Overview")
    
    # Risk Band Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Band Distribution")
        risk_counts = df['risk_band'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.4,
            color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, 
            x='risk_score', 
            nbins=50,
            color='risk_band',
            color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'},
            marginal='box'
        )
        fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # High Risk Cases Table
    st.markdown("### High Risk Cases")
    high_risk_df = df[df['risk_band'] == 'High'].sort_values('risk_score', ascending=False).head(20)
    st.dataframe(
        high_risk_df[['user', 'week', 'risk_score', 'role', 'department']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'user': st.column_config.TextColumn('Employee', width='medium'),
            'week': st.column_config.DateColumn('Week', width='medium'),
            'risk_score': st.column_config.NumberColumn('Risk Score', format='%.1f'),
            'role': st.column_config.TextColumn('Role', width='medium'),
            'department': st.column_config.TextColumn('Department', width='medium')
        }
    )


elif view_mode == "Summary & Evaluation":
    # =============================================================================
    # SUMMARY & EVALUATION VIEW (Real metrics from evaluation)
    # =============================================================================
    
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

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Insider Threat Detection Dashboard - Development Version")
