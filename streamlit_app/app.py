import streamlit as st
import numpy as np
import joblib
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Churn Prediction System", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("Enter customer details to predict churn risk + recommendations")

# ----------------------------
# SAFE PATH FIX (IMPORTANT)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ----------------------------
# Load models
# ----------------------------
try:
    model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ----------------------------
# Recommendation logic
# ----------------------------
def get_recommendation(churn_prob, cluster):

    if churn_prob > 0.75:
        return "High Risk: Offer discount + priority support"

    elif churn_prob > 0.4:
        return "Medium Risk: Send engagement emails"

    else:
        return "Stable Customer → Loyalty program"

# ----------------------------
# INPUT (10 FEATURES)
# ----------------------------
st.subheader("📥 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    AccountWeeks = st.number_input("Account Weeks", min_value=0)
    ContractRenewal = st.number_input("Contract Renewal (0/1)", 0, 1)
    DataPlan = st.number_input("Data Plan (0/1)", 0, 1)
    DataUsage = st.number_input("Data Usage", min_value=0.0)
    CustServCalls = st.number_input("Customer Service Calls", min_value=0)

with col2:
    DayMins = st.number_input("Day Minutes", min_value=0.0)
    DayCalls = st.number_input("Day Calls", min_value=0)
    MonthlyCharge = st.number_input("Monthly Charge", min_value=0.0)
    OverageFee = st.number_input("Overage Fee", min_value=0.0)
    RoamMins = st.number_input("Roam Minutes", min_value=0.0)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("🚀 Predict Churn"):

    input_data = np.array([
        AccountWeeks,
        ContractRenewal,
        DataPlan,
        DataUsage,
        CustServCalls,
        DayMins,
        DayCalls,
        MonthlyCharge,
        OverageFee,
        RoamMins
    ]).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    churn_prob = model.predict_proba(scaled_data)[0][1]
    cluster = kmeans.predict(scaled_data)[0]

    recommendation = get_recommendation(churn_prob, cluster)

    st.subheader("📊 Results")

    st.metric("Churn Probability", f"{churn_prob:.2f}")
    st.metric("Cluster", int(cluster))

    if churn_prob > 0.75:
        st.error("High Risk")
    elif churn_prob > 0.4:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")

    st.info(recommendation)