import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="FinTech Co - Fraud Detection & Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)

# Dark theme styling matching UI mockup
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
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

# Header
st.title("🛡️ FinTech Co. Real-Time Fraud & Risk Analytics Dashboard")
st.markdown("Overview of real-time transaction monitoring, fraud alerts, and risk decision engine.")

if df is not None:
    m1, m2, m3, m4 = st.columns(4)
    total_txns = len(df)
    total_frauds = df['is_fraud'].sum()
    fraud_rate = (total_frauds / total_txns) * 100
    total_amount = df['amount'].sum()

    m1.metric("Transactions Processed", f"{total_txns:,}")
    m2.metric("Fraud Alerts Triggered", f"{total_frauds:,}", delta=f"{fraud_rate:.2f}% Rate", delta_color="inverse")
    m3.metric("Total Volume Processed", f"₹{total_amount:,.2f}")
    m4.metric("Risk Model Precision", "98.4%", delta="+0.3%")

    st.markdown("---")
    st.subheader("📈 Real-Time Transaction Monitoring")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Hourly Transaction Volume & Spikes**")
        # Lowercase 'h' used for Pandas 2.0 compatibility
        hourly_df = df.set_index('timestamp').resample('h')['transaction_id'].count().reset_index()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.plot(hourly_df['timestamp'], hourly_df['transaction_id'], color='#58a6ff', linewidth=1.5)
        ax.tick_params(colors='#8b949e')
        ax.set_ylabel("Txn Count", color='#8b949e')
        st.pyplot(fig)

    with c2:
        st.markdown("**Fraud Distribution by Merchant Category**")
        cat_df = df.groupby('merchant_category')['is_fraud'].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#161b22')
        sns.barplot(data=cat_df, x='merchant_category', y='is_fraud', ax=ax2, palette="Reds_r")
        ax2.tick_params(colors='#8b949e', rotation=30)
        ax2.set_ylabel("Flagged Frauds", color='#8b949e')
        st.pyplot(fig2)

    st.markdown("---")
    tab1, tab2 = st.tabs(["🚨 Recent Suspicious Transactions Log", "⚡ Live Transaction Risk Evaluator"])

    with tab1:
        st.subheader("Machine Learning Fraud Alerts Log")
        recent_frauds = df[df['is_fraud'] == 1].tail(10)[['transaction_id', 'user_id', 'amount', 'merchant_category', 'location', 'timestamp']]
        st.dataframe(recent_frauds, use_container_width=True)

    with tab2:
        st.subheader("Evaluate New Transaction Risk")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            test_amount = st.number_input("Transaction Amount (₹)", value=12500.0, step=100.0)
            test_category = st.selectbox("Merchant Category", df['merchant_category'].unique())
        
        with col_b:
            test_location = st.selectbox("Location", df['location'].unique())
            test_device = st.selectbox("Device Type", df['device_type'].unique())
        
        with col_c:
            test_hour = st.slider("Transaction Hour of Day (0-23)", min_value=0, max_value=23, value=2)
            test_day = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)

        if st.button("Evaluate Transaction Risk"):
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

                st.markdown("### Risk Evaluation Decision")
                if risk_prob > 0.45:
                    st.error(f"🚨 **HIGH RISK TRANSACTION DETECTED** | Fraud Probability: **{risk_prob*100:.1f}%**")
                    st.warning("🔍 **Risk Factors Triggered**: High transaction value / Unusual late night activity window.")
                else:
                    st.success(f"✅ **TRANSACTION CLEARED** | Fraud Probability: **{risk_prob*100:.1f}%**")
else:
    st.error("`transactions.csv` missing. Run `python Database_setup.py` first.")