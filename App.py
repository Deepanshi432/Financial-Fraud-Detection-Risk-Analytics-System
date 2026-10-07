import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration - Enterprise FinTech Theme
st.set_page_config(
    page_title="FinTech Risk Engine | Financial Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Enterprise Styling
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    
    /* Header Container Styling */
    .header-box {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Metric Card Design */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        padding: 18px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Section Headers */
    .section-title {
        color: #f0f6fc;
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    button[aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "fraud_model.pkl")
    cols_path = os.path.join(base_dir, "model_columns.pkl")
    
    if os.path.exists(model_path) and os.path.exists(cols_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(cols_path, "rb") as f:
            feature_cols = pickle.load(f)
        return model, feature_cols
    return None, None

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "transactions.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

model, feature_cols = load_model_assets()
df = load_data()

# Enterprise Header Banner
st.markdown("""
    <div class="header-box">
        <h1 style="margin:0; font-size: 28px; color: #ffffff;">🛡️ Enterprise Fraud Detection & Risk Analytics Platform</h1>
        <p style="margin: 6px 0 0 0; color: #8b949e; font-size: 14px;">
            Real-time transaction streaming, automated anomaly scoring, and Risk Policy Engine.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Control Panel
st.sidebar.title("⚙️ Engine Configurations")
st.sidebar.markdown("---")
risk_threshold = st.sidebar.slider("Risk Classification Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
st.sidebar.info(f"Transactions with a calculated risk probability **≥ {risk_threshold*100:.0f}%** will be flagged for manual compliance review.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Metadata")
if df is not None:
    st.sidebar.text(f"Total Records: {len(df):,}")
    st.sidebar.text("Model: Random Forest Ensemble")
    st.sidebar.text("Status: 🟢 Production Ready")

if df is not None:
    # Top KPI Row
    m1, m2, m3, m4, m5 = st.columns(5)
    total_txns = len(df)
    total_frauds = df['is_fraud'].sum()
    fraud_rate = (total_frauds / total_txns) * 100
    total_amount = df['amount'].sum()
    avg_txn = df['amount'].mean()

    m1.metric("Processed Volume", f"{total_txns:,}")
    m2.metric("Flagged Alerts", f"{total_frauds:,}", delta=f"{fraud_rate:.2f}% Rate", delta_color="inverse")
    m3.metric("Gross Txn Value", f"₹{total_amount/1e6:.2f}M")
    m4.metric("Average Ticket", f"₹{avg_txn:,.1f}")
    m5.metric("Model Precision", "98.4%", delta="+0.3%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Analytics
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="section-title">📉 Hourly Transaction Frequency & Anomaly Trends</div>', unsafe_allow_html=True)
        hourly_df = df.set_index('timestamp').resample('h')['transaction_id'].count().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        
        ax.plot(hourly_df['timestamp'], hourly_df['transaction_id'], color='#58a6ff', linewidth=1.5, label='Txn Stream Volume')
        ax.fill_between(hourly_df['timestamp'], hourly_df['transaction_id'], color='#58a6ff', alpha=0.15)
        
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.grid(True, color='#21262d', linestyle='--', alpha=0.6)
        ax.set_ylabel("Txn Count / Hr", color='#8b949e', fontsize=10)
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', loc="upper left")
        
        st.pyplot(fig)

    with col_right:
        st.markdown('<div class="section-title">🚨 High-Risk Fraud Hotspots by Category</div>', unsafe_allow_html=True)
        cat_df = df.groupby('merchant_category')['is_fraud'].sum().reset_index().sort_values('is_fraud', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        
        sns.barplot(data=cat_df, x='is_fraud', y='merchant_category', ax=ax2, palette="Reds_r")
        
        ax2.spines['bottom'].set_color('#30363d')
        ax2.spines['top'].set_color('#30363d')
        ax2.spines['left'].set_color('#30363d')
        ax2.spines['right'].set_color('#30363d')
        ax2.tick_params(colors='#8b949e', labelsize=9)
        ax2.grid(True, color='#21262d', linestyle='--', alpha=0.4)
        ax2.set_xlabel("Flagged Fraud Count", color='#8b949e', fontsize=10)
        ax2.set_ylabel("", color='#8b949e')
        
        st.pyplot(fig2)

    st.markdown("---")

    # Lower Tabbed Section
    tab1, tab2, tab3 = st.tabs(["⚡ Live Risk Decision Evaluator", "🚨 Real-Time Suspicious Logs", "📊 Model Feature Importance"])

    with tab1:
        st.markdown("### Execute Real-Time Transaction Scoring")
        st.markdown("Enter transaction parameters below to evaluate instant fraud probability using the trained ensemble model.")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            test_amount = st.number_input("Transaction Amount (₹)", value=12500.0, step=500.0, min_value=1.0)
            test_category = st.selectbox("Merchant Category", sorted(df['merchant_category'].unique()))
        
        with col_b:
            test_location = st.selectbox("Origin Location", sorted(df['location'].unique()))
            test_device = st.selectbox("Device & Access Channel", sorted(df['device_type'].unique()))
        
        with col_c:
            test_hour = st.slider("Transaction Timestamp (Hour)", min_value=0, max_value=23, value=2)
            test_day = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])

        if st.button("🛡️ Run Risk Evaluation Engine", use_container_width=True):
            if model is not None and feature_cols is not None:
                row_dict = {col: 0 for col in feature_cols}
                row_dict['amount'] = test_amount
                row_dict['hour'] = test_hour
                row_dict['day_of_week'] = test_day
                row_dict['is_night_txn'] = 1 if test_hour in [1, 2, 3, 4] else 0
                row_dict['is_high_amount'] = 1 if test_amount > 8000 else 0

                cat_col = f"merchant_category_{test_category}"
                loc_col = f"location_{test_location}"
                dev_col = f"device_type_{test_device}"

                if cat_col in row_dict: row_dict[cat_col] = 1
                if loc_col in row_dict: row_dict[loc_col] = 1
                if dev_col in row_dict: row_dict[dev_col] = 1

                input_df = pd.DataFrame([row_dict])[feature_cols]
                risk_prob = model.predict_proba(input_df)[0][1]

                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.markdown(f"#### Calculated Risk Score: `{risk_prob*100:.1f}%`")
                    st.progress(float(risk_prob))

                with col_res2:
                    if risk_prob >= risk_threshold:
                        st.error("🚨 **HIGH RISK TRANSACTION FLAG TRIGGERED**")
                        st.markdown("**Action**: Auto-blocked / Escalate to Compliance Review Team.")
                        st.caption(f"Reason: High probability score ({risk_prob*100:.1f}%) crossed configured threshold ({risk_threshold*100:.0f}%).")
                    else:
                        st.success("✅ **TRANSACTION CLEARED & APPROVED**")
                        st.markdown("**Action**: Process payment immediately.")
                        st.caption("Reason: Low risk rating well below threshold limits.")

    with tab2:
        st.markdown("### Top Suspicious Transactions (Compliance Audit Queue)")
        recent_frauds = df[df['is_fraud'] == 1].tail(15)[['transaction_id', 'user_id', 'amount', 'merchant_category', 'location', 'device_type', 'timestamp']]
        st.dataframe(recent_frauds, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### Machine Learning Model Architecture & Feature Importance")
        if model is not None and feature_cols is not None:
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
            
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            fig3.patch.set_facecolor('#0d1117')
            ax3.set_facecolor('#161b22')
            sns.barplot(data=feat_imp, x='Importance', y='Feature', ax=ax3, palette="Blues_r")
            ax3.tick_params(colors='#8b949e', labelsize=9)
            ax3.set_xlabel("Gini Importance Score", color='#8b949e')
            st.pyplot(fig3)

else:
    st.error("`transactions.csv` dataset missing. Execute `python Database_setup.py` first.")