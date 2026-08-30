import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# =============================================================================
# DATA LOADING FUNCTION (Single point of entry - easy to swap file path later)
# =============================================================================
@st.cache_data
def load_dashboard_data():
    """Load dashboard development data with risk scores and explanations."""
    file_path = 'data/cleaned/dashboard_dev_data.csv'
    df = pd.read_csv(file_path)
    df['week'] = pd.to_datetime(df['week'])
    return df

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        padding: 20px;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #1e293b;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        padding: 16px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid #334155;
    }
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
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin: 0;
        border: 1px solid #e2e8f0;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.12);
    }
    .info-box {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin: 0;
        border: 1px solid #e2e8f0;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 24px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    [data-testid="stMarkdownContainer"] > p {
        color: #334155;
    }
    h1, h2, h3 {
        color: #0f172a !important;
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
st.sidebar.markdown('<div class="sidebar-header">Controls</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

# Employee and week selector
st.sidebar.markdown("**Employee Selection**")
unique_users = sorted(df['user'].unique())
selected_user = st.sidebar.selectbox("Select Employee", unique_users, key="user_select")

st.sidebar.markdown("**Week Selection**")
# Get weeks for selected user
user_weeks = df[df['user'] == selected_user]['week'].sort_values().unique()
selected_week = st.sidebar.selectbox("Select Week", user_weeks, key="week_select")

# Replay mechanism
st.sidebar.markdown("---")
st.sidebar.markdown("**Replay Mode**")
auto_replay = st.sidebar.checkbox("Enable Auto-Replay", key="auto_replay")
replay_speed = st.sidebar.slider("Speed (seconds)", 1, 5, 2, key="replay_speed")

if auto_replay:
    current_week_index = list(user_weeks).index(selected_week)
    if current_week_index < len(user_weeks) - 1:
        selected_week = user_weeks[current_week_index + 1]
        st.rerun()

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
st.markdown('<div class="main-header">Insider Threat Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Get selected row
selected_row = df[(df['user'] == selected_user) & (df['week'] == selected_week)].iloc[0]

if view_mode == "Individual Analysis":
    # =============================================================================
    # INDIVIDUAL ANALYSIS VIEW
    # =============================================================================
    
    # Header with employee info in styled containers
    st.markdown('<div class="section-header">Employee Information</div>', unsafe_allow_html=True)
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
    
    # Alert Panel with styled container
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
    
    if risk_score > 70:
        st.error(f"FLAGGED - High Risk Score: {risk_score:.1f}")
    elif risk_score > 30:
        st.warning(f"Medium Risk Score: {risk_score:.1f}")
    else:
        st.success(f"Low Risk Score: {risk_score:.1f}")
    
    # Risk Score Gauge and Metrics
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SHAP Explanation Panel
    st.markdown('<div class="section-header">SHAP Explanation</div>', unsafe_allow_html=True)
    if pd.notna(selected_row['shap_top_features']):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Why is this risk score high?")
        try:
            shap_features = json.loads(selected_row['shap_top_features'])
            if shap_features:
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
        except:
            st.info("SHAP explanation data not available")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No SHAP explanation data available for this case")
    
    st.markdown("---")
    
    # DiCE Explanation Panel
    st.markdown('<div class="section-header">DiCE Counterfactual</div>', unsafe_allow_html=True)
    if pd.notna(selected_row['dice_explanation']):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("What would lower the risk?")
        st.info(selected_row['dice_explanation'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No DiCE explanation data available for this case")

elif view_mode == "Risk Overview":
    # =============================================================================
    # RISK OVERVIEW VIEW
    # =============================================================================
    
    st.markdown('<div class="main-header">Risk Distribution Overview</div>', unsafe_allow_html=True)
    
    # Risk Band Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # High Risk Cases Table
    st.markdown('<div class="section-header">High Risk Cases</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

elif view_mode == "Summary & Evaluation":
    # =============================================================================
    # SUMMARY & EVALUATION VIEW (Placeholders for real metrics)
    # =============================================================================
    
    st.markdown('<div class="main-header">Summary & Evaluation Metrics</div>', unsafe_allow_html=True)
    
    # Placeholder metrics in styled containers
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", "0.85", "↑ 5%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", "0.78", "↑ 3%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1 Score", "0.81", "↑ 4%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("FPR", "0.12", "↓ 8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Placeholder charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Model Performance Over Time")
        # Placeholder data
        weeks = df['week'].dt.strftime('%Y-%m').unique()[:12]
        precision = np.random.uniform(0.75, 0.90, 12)
        recall = np.random.uniform(0.70, 0.85, 12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=precision, name='Precision', mode='lines+markers', line=dict(color='#2563eb', width=2)))
        fig.add_trace(go.Scatter(x=weeks, y=recall, name='Recall', mode='lines+markers', line=dict(color='#f59e0b', width=2)))
        fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("False Positive Rate Comparison")
        # Placeholder comparison
        methods = ['Fixed Threshold', 'Peer-Cohort Baselining']
        fpr_rates = [0.25, 0.12]
        
        fig = px.bar(
            x=methods,
            y=fpr_rates,
            labels={'x': 'Method', 'y': 'FPR'},
            color=methods,
            color_discrete_map={'Fixed Threshold': '#ef4444', 'Peer-Cohort Baselining': '#10b981'},
            text=fpr_rates,
            text_auto='.2f'
        )
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
        fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("Note: Evaluation metrics are placeholders and will be replaced with real model performance data once the XGBoost model is trained and evaluated.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Insider Threat Detection Dashboard - Development Version")
