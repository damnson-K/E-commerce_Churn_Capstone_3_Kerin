# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go


# Set Page Title and Layout
st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛍️", layout="wide")

# Apply Custom CSS for Styling
st.markdown("""
    <style>
        /* Background gradient for the main title */
        .title-text {
            background: linear-gradient(to right, #4b6cb7, #182848);
            -webkit-background-clip: text;
            color: transparent;
            font-size: 32px;
            font-weight: bold;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1e1e2f !important;
        }

        /* Table styling */
        div[data-testid="stDataFrame"] {
            width: 100% !important;
        }
        table {
            font-size: 16px !important;
            text-align: left !important;
        }
        thead th {
            background-color: #4b6cb7 !important;
            color: white !important;
        }
        tbody tr {
            background-color: #2a2a40 !important;
            color: white !important;
        }
        tbody tr:nth-child(even) {
            background-color: #3a3a55 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #4b6cb7;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #182848;
        }

        /* Footer */
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Add Header Image
st.image("churn.png", use_container_width=True)

# Title
st.markdown("<h1 class='title-text'>📊 E-Commerce Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.header("📊 E-Commerce Customer Churn Prediction")
st.markdown("Use this tool to predict whether a customer is likely to churn or stay loyal.")

# Sidebar for user input
st.sidebar.header("🔍 Enter Customer Details")

def get_user_input():
    tenure = st.sidebar.slider("📅 Tenure (Months)", 0, 61, 12)
    warehouse_to_home = st.sidebar.slider("📍 Distance from Warehouse (km)", 5, 127, 30)
    num_devices = st.sidebar.slider("📱 Devices Registered", 1, 6, 2)
    satisfaction_score = st.sidebar.slider("😊 Satisfaction Score", 1, 5, 3)
    num_address = st.sidebar.slider("🏠 Number of Addresses", 1, 21, 5)
    complain = st.sidebar.radio("❗ Complaints Filed?", ["No (0)", "Yes (1)"])
    days_since_last_order = st.sidebar.slider("📆 Days Since Last Order", 0, 46, 10)
    cashback_amount = st.sidebar.number_input("💰 Cashback Amount (USD)", 0.0, 324.99, 50.0)
    
    prefered_order_cat = st.sidebar.radio("🛒 Preferred Order Category", ["Mobile Phone", "Grocery", "Fashion", "Laptop & Accessory", "Others"])
    marital_status = st.sidebar.radio("💍 Marital Status", ["Divorced", "Married", "Single"])

    user_data = pd.DataFrame({
        "Tenure": [tenure],
        "WarehouseToHome": [warehouse_to_home],
        "NumberOfDeviceRegistered": [num_devices],
        "SatisfactionScore": [satisfaction_score],
        "NumberOfAddress": [num_address],
        "Complain": [0 if complain == "No (0)" else 1],
        "DaySinceLastOrder": [days_since_last_order],
        "CashbackAmount": [cashback_amount],
        "PreferedOrderCat": [prefered_order_cat],
        "MaritalStatus": [marital_status]
    })
    
    return user_data

# Collect customer data
customer_data = get_user_input()

# Rename columns for better readability
customer_data_renamed = customer_data.rename(columns={
    "Tenure": "Tenure (Months)",
    "WarehouseToHome": "Distance to Warehouse (km)",
    "NumberOfDeviceRegistered": "Devices Registered",
    "SatisfactionScore": "Satisfaction (1-5)",
    "NumberOfAddress": "Number of Addresses",
    "Complain": "Filed Complaint (0 = No, 1 = Yes)",
    "DaySinceLastOrder": "Days Since Last Order",
    "CashbackAmount": "Cashback Received (USD)",
    "PreferedOrderCat": "Preferred Order Category",
    "MaritalStatus": "Marital Status"
})

# Layout with two columns
col1, col2 = st.columns(2)

# Left Column - Display Customer Data
with col1:
    st.subheader("📌 Customer Details")
    st.dataframe(customer_data_renamed.T, width=700, height=400)
    st.caption("This table provides a breakdown of customer attributes used for prediction.")

# Load trained model
with open("finalmodel_lgbm.sav", "rb") as f:
    model = pickle.load(f)

# Predict churn
prediction = model.predict(customer_data)[0]
probability = model.predict_proba(customer_data)[0][1]  # Probability of churn

# Gauge Chart for Churn Probability
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={"text": "Churn Probability (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#950606" if probability > 0.5 else "#06402B"},
        "steps": [
            {"range": [0, 50], "color": "#ACE1AF"},
            {"range": [50, 100], "color": "#B22222"},
        ],
    }
))

# Right Column - Display Prediction
with col2:
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is **likely to churn**.")
        churn_color = "red"
    else:
        st.success("✅ This customer is **not likely to churn**.")
        churn_color = "green"
    
    st.plotly_chart(fig, use_container_width=True)

# Download Button
csv = customer_data.to_csv(index=False)
st.download_button(label="📥 Download Data", data=csv, file_name="customer_data.csv", mime="text/csv")

# Footer with Credit
st.markdown(
    "<div class='footer'>Created by <b>Kerin</b> | <a href='https://www.linkedin.com/in/kerin-m' target='_blank'>LinkedIn</a></div>", 
    unsafe_allow_html=True
)
