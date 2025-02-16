# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib  # Import joblib for alternative model loading
import plotly.graph_objects as go
import os  # Added to check for model file existence

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# List files in the directory to check if the model file is present
print("Files in Directory:", os.listdir())

# Set Page Title and Layout
st.set_page_config(page_title="E-Commerce Churn Predictor", page_icon="🛍️", layout="wide")

# Apply Custom CSS for Styling
st.markdown("""
    <style>
        .title-text {
            background: linear-gradient(to right, #4b6cb7, #182848);
            -webkit-background-clip: text;
            color: transparent;
            font-size: 32px;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e2f !important;
        }
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
st.image("churn.png", use_container_width=True)  # Updated to use_container_width

# Title
st.markdown("<h1 class='title-text'>📊 E-Commerce Customer Churn Prediction</h1>", unsafe_allow_html=True)

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

# Check if model file exists before loading
model_path = "finalmodel_lgbm.sav"

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.warning("⚠️ Pickle failed to load. Trying joblib...")
        try:
            model = joblib.load(model_path)  # Attempt to load with joblib
        except Exception as e:
            st.error(f"❌ Failed to load model. Error: {e}")
            st.stop()  # Stop execution if model loading fails

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
        else:
            st.success("✅ This customer is **not likely to churn**.")
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("🚨 Model file not found! Please check if `finalmodel_lgbm.sav` exists in the directory.")
    st.stop()  # Stop execution if the model file is missing

# Footer with Credit
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            left: 10px;  /* Move to the left */
            font-size: 14px;
            color: #666;
        }
        .footer a {
            color: #666;
            text-decoration: none;
        }
        .footer a:hover {
            color: #4b6cb7; /* Hover effect */
            text-decoration: underline;
        }
    </style>
    <div class='footer'>
        Created by <b>Kerin</b> | 
        <a href='https://www.linkedin.com/in/kerin-m' target='_blank'>LinkedIn</a>
    </div>
""", unsafe_allow_html=True)

