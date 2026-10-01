import sys
import os
from streamlit.runtime import exists

# Auto-launch Streamlit if executed directly with `python app.py` or Run button in IBM Bob
if not exists():
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Setup import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import FraudDetector

st.set_page_config(
    page_title="IBM BOB - Banking Fraud Detection System",
    page_icon="???",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f62fe 0%, #001d6c 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .badge-critical {
        background-color: #da1e28;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: #ff832b;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: #24a148;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f4f6f8;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load detector
@st.cache_resource
def get_detector():
    return FraudDetector()

detector = get_detector()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.2rem;">??? IBM BOB | Banking Fraud Detection System</h1>
    <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
        Enterprise Financial Risk Engine & Real-Time Anomaly Detection ? Inspired by Bank of Baroda & IBM Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=110)
    st.title("System Control")
    st.markdown(f"**Champion Model:** `{detector.model_name}`")
    st.markdown(f"**Classification Threshold:** `{detector.threshold:.4f}`")
    st.markdown("**Target Class Imbalance:** ~1.5% Fraud Rate")
    st.markdown("---")
    st.markdown("### ?? Running in IBM Bob")
    st.code("bobide .", language="bash")
    st.caption("You can open and develop this project directly inside IBM Bob IDE via `bobide`.")

tab1, tab2, tab3 = st.tabs([
    "?? Real-Time Transaction Screener",
    "?? Batch Transaction Auditor",
    "?? Model Benchmarks & Explainability"
])

# ----------------- TAB 1: Real-Time Screener -----------------
with tab1:
    st.subheader("Live Transaction Risk Evaluation")
    st.write("Input financial transaction attributes below or load sample presets to test the fraud model.")
    
    col_pre1, col_pre2, _ = st.columns([1.5, 1.5, 4])
    if col_pre1.button("? Preset: Suspicious Liquidation"):
        st.session_state.tx_type = "TRANSFER"
        st.session_state.amount = 450000.0
        st.session_state.old_orig = 450000.0
        st.session_state.new_orig = 0.0
        st.session_state.old_dest = 0.0
        st.session_state.new_dest = 450000.0
        st.session_state.hour = 3
        st.session_state.device_risk = 92.0
        st.session_state.intl = True

    if col_pre2.button("? Preset: Normal Merchant Payment"):
        st.session_state.tx_type = "PAYMENT"
        st.session_state.amount = 1200.0
        st.session_state.old_orig = 25000.0
        st.session_state.new_orig = 23800.0
        st.session_state.old_dest = 50000.0
        st.session_state.new_dest = 51200.0
        st.session_state.hour = 14
        st.session_state.device_risk = 15.0
        st.session_state.intl = False

    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("### Transaction Parameters")
        c1, c2 = st.columns(2)
        with c1:
            tx_type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                index=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"].index(
                    st.session_state.get("tx_type", "TRANSFER")
                )
            )
            amount = st.number_input(
                "Transaction Amount (?)",
                min_value=1.0, max_value=10000000.0,
                value=float(st.session_state.get("amount", 25000.0)),
                step=500.0
            )
            old_orig = st.number_input(
                "Sender Starting Balance (?)",
                min_value=0.0, max_value=50000000.0,
                value=float(st.session_state.get("old_orig", 30000.0)),
                step=1000.0
            )
            new_orig = st.number_input(
                "Sender Ending Balance (?)",
                min_value=0.0, max_value=50000000.0,
                value=float(st.session_state.get("new_orig", 5000.0)),
                step=1000.0
            )
        with c2:
            hour = st.slider(
                "Transaction Time (Hour of Day: 0-23)",
                min_value=0, max_value=23,
                value=int(st.session_state.get("hour", 14))
            )
            old_dest = st.number_input(
                "Recipient Starting Balance (?)",
                min_value=0.0, max_value=50000000.0,
                value=float(st.session_state.get("old_dest", 10000.0)),
                step=1000.0
            )
            new_dest = st.number_input(
                "Recipient Ending Balance (?)",
                min_value=0.0, max_value=50000000.0,
                value=float(st.session_state.get("new_dest", 35000.0)),
                step=1000.0
            )
            device_risk = st.slider(
                "Device/Network Risk Score (0-100)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state.get("device_risk", 20.0))
            )
            intl = st.checkbox("International / Foreign Routing", value=bool(st.session_state.get("intl", False)))

    with col2:
        st.markdown("### Risk Assessment Result")
        tx_data = {
            "step": hour,
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "device_risk_score": device_risk,
            "is_international": 1 if intl else 0
        }
        
        result = detector.score_transaction(tx_data)
        prob = result["fraud_probability"]
        is_fraud = result["is_fraud"]
        tier = result["risk_tier"]
        
        # Display Card
        if is_fraud:
            st.error(f"?? **ALERT: TRANSACTION FLAGGED AS FRAUDULENT**")
        else:
            st.success(f"? **TRANSACTION CLEARED: APPROVED**")
            
        m1, m2 = st.columns(2)
        m1.metric("Fraud Probability", f"{prob * 100:.1f}%")
        m2.metric("Decision Status", "DECLINE" if is_fraud else "APPROVE")
        
        st.progress(prob)
        
        st.markdown(f"**Risk Tier:** `{tier}`")
        
        st.markdown("#### Identified Risk Indicators:")
        for factor in result["risk_factors"]:
            st.markdown(f"- {factor}")

# ----------------- TAB 2: Batch Auditor -----------------
with tab2:
    st.subheader("Batch Transaction Risk Screening")
    st.write("Upload a transaction ledger CSV or audit our prepared benchmark test set.")
    
    test_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "test_raw.csv")
    
    col_b1, col_b2 = st.columns([2, 1])
    uploaded_file = col_b1.file_uploader("Upload Transactions CSV", type=["csv"])
    load_sample = col_b2.button("?? Load 200 Test Split Transactions")
    
    df_to_audit = None
    if uploaded_file is not None:
        df_to_audit = pd.read_csv(uploaded_file)
    elif load_sample and os.path.exists(test_csv_path):
        df_to_audit = pd.read_csv(test_csv_path).head(200)
    elif os.path.exists(test_csv_path):
        df_to_audit = pd.read_csv(test_csv_path).head(50)
        
    if df_to_audit is not None:
        with st.spinner("Screening batch transactions through ML pipeline..."):
            audited_df = detector.score_batch(df_to_audit)
            
        total_tx = len(audited_df)
        flagged_tx = int(audited_df["is_fraud_predicted"].sum())
        total_amt = audited_df["amount"].sum()
        flagged_amt = audited_df[audited_df["is_fraud_predicted"] == 1]["amount"].sum()
        
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Transactions", f"{total_tx:,}")
        b2.metric("Flagged Frauds", f"{flagged_tx:,}")
        b3.metric("Fraud Rate", f"{(flagged_tx / total_tx) * 100:.2f}%")
        b4.metric("Flagged Volume", f"?{flagged_amt:,.2f}")
        
        st.dataframe(
            audited_df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", "device_risk_score", "fraud_probability", "risk_tier", "is_fraud_predicted"]],
            use_container_width=True
        )
        
        csv_bytes = audited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "?? Download Audited Report (CSV)",
            data=csv_bytes,
            file_name="audited_fraud_report.csv",
            mime="text/csv"
        )

# ----------------- TAB 3: Model Benchmarks & Explainability -----------------
with tab3:
    st.subheader("Model Evaluation & Performance Diagnostics")
    
    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        st.markdown("### Candidate Model Comparison (Validation Split)")
        val_metrics = meta.get("validation_metrics", {})
        table_rows = []
        for m_name, m_vals in val_metrics.items():
            table_rows.append({
                "Model Architecture": m_name,
                "PR-AUC": m_vals["pr_auc"],
                "ROC-AUC": m_vals["roc_auc"],
                "Precision": m_vals["precision"],
                "Recall": m_vals["recall"],
                "F1-Score": m_vals["f1"],
                "Threshold": m_vals["threshold"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        
    st.markdown("### Diagnostic Plots (Unseen Test Set)")
    p_col1, p_col2 = st.columns(2)
    
    cm_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "confusion_matrix.png")
    curves_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "evaluation_curves.png")
    feat_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "feature_importance.png")
    
    if os.path.exists(cm_img):
        p_col1.image(cm_img, caption="Confusion Matrix on 4,500 Test Transactions", use_container_width=True)
    if os.path.exists(curves_img):
        p_col2.image(curves_img, caption="ROC and Precision-Recall Curves", use_container_width=True)
    if os.path.exists(feat_img):
        st.image(feat_img, caption="Key Fraud Risk Features & Weights", use_container_width=True)
