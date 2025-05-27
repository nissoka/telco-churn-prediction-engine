import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
import sagemaker # Import sagemaker here
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer # This will be used to send data to the endpoint
from sagemaker.deserializers import CSVDeserializer # This will be used to receive data from the endpoint

# --- AWS SageMaker Endpoint Configuration ---
# IMPORTANT: This endpoint name must match your deployed SageMaker endpoint.
# If you re-deploy your model in SageMaker, this name will change, and you'll need to update it here.
SAGEMAKER_ENDPOINT_NAME = "telco-churn-xgb-training-2025-05-27-03-39-12-964" # <--- YOUR ENDPOINT NAME

# Initialize SageMaker predictor (cached for performance)
@st.cache_resource
def get_sagemaker_predictor(endpoint_name):
    """
    Initializes and returns a SageMaker Predictor object configured for CSV input/output.
    """
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker.Session(), # <--- CRITICAL FIX: Use sagemaker.Session()
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer()
    )
    return predictor

# --- Feature List (Must match the order of your model's expected features) ---
# This list is CRITICAL for sending data in the correct order to the model.
# This order was determined during the preprocessing steps in the Jupyter notebook (e.g., from X_train.columns.tolist()).
feature_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes',
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year',
    'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]


# --- Streamlit UI ---
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("📞 Telco Customer Churn Prediction Engine")
st.markdown("""
This application predicts whether a customer is likely to churn based on their service details.
Powered by an XGBoost model deployed on Amazon SageMaker.
""")

st.header("Customer Details")

# Input fields for numerical features and binary categorical features (handled as 0/1)
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"], index=0) # Default to No
senior_citizen_val = 1 if senior_citizen == "Yes" else 0

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=0.01)

st.subheader("Demographic & Service Information")

gender_male = st.selectbox("Gender", ["Female", "Male"], index=0) # Default to Female
gender_male_val = 1 if gender_male == "Male" else 0

partner = st.selectbox("Partner?", ["No", "Yes"], index=0) # Default to No
partner_val = 1 if partner == "Yes" else 0

dependents = st.selectbox("Dependents?", ["No", "Yes"], index=0) # Default to No
dependents_val = 1 if dependents == "Yes" else 0

phone_service = st.selectbox("Phone Service?", ["No", "Yes"], index=1) # Default to Yes
phone_service_val = 1 if phone_service == "Yes" else 0

multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes"], index=0) # Default to No
multiple_lines_val = 1 if multiple_lines == "Yes" else 0

internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"], index=0) # Default to DSL
internet_service_fiber_optic_val = 1 if internet_service == "Fiber optic" else 0
internet_service_no_val = 1 if internet_service == "No" else 0

online_security = st.selectbox("Online Security?", ["No", "Yes"], index=0) # Default to No
online_security_val = 1 if online_security == "Yes" else 0

online_backup = st.selectbox("Online Backup?", ["No", "Yes"], index=0) # Default to No
online_backup_val = 1 if online_backup == "Yes" else 0

device_protection = st.selectbox("Device Protection?", ["No", "Yes"], index=0) # Default to No
device_protection_val = 1 if device_protection == "Yes" else 0

tech_support = st.selectbox("Tech Support?", ["No", "Yes"], index=0) # Default to No
tech_support_val = 1 if tech_support == "Yes" else 0

streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes"], index=0) # Default to No
streaming_tv_val = 1 if streaming_tv == "Yes" else 0

streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes"], index=0) # Default to No
streaming_movies_val = 1 if streaming_movies == "Yes" else 0

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0) # Default to Month-to-month
contract_one_year_val = 1 if contract == "One year" else 0
contract_two_year_val = 1 if contract == "Two year" else 0

paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"], index=1) # Default to Yes
paperless_billing_val = 1 if paperless_billing == "Yes" else 0

payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0) # Default to Electronic check
payment_method_credit_card_val = 1 if payment_method == "Credit card (automatic)" else 0
payment_method_electronic_check_val = 1 if payment_method == "Electronic check" else 0
payment_method_mailed_check_val = 1 if payment_method == "Mailed check" else 0


if st.button("Predict Churn"):
    # Create input features array in the correct order as expected by the model
    # This order must strictly match the order of columns in X_train used during model training.
    input_data = np.array([
        senior_citizen_val,
        tenure,
        monthly_charges,
        total_charges,
        gender_male_val,
        partner_val,
        dependents_val,
        phone_service_val,
        multiple_lines_val,
        internet_service_fiber_optic_val,
        internet_service_no_val,
        online_security_val,
        online_backup_val,
        device_protection_val,
        tech_support_val,
        streaming_tv_val,
        streaming_movies_val,
        contract_one_year_val,
        contract_two_year_val,
        paperless_billing_val,
        payment_method_credit_card_val,
        payment_method_electronic_check_val,
        payment_method_mailed_check_val
    ]).astype(float) # Ensure all inputs are float for the model

    # Verify input data shape and order before sending to endpoint
    if len(input_data) != len(feature_columns):
        st.error(f"Input data mismatch! Expected {len(feature_columns)} features, but got {len(input_data)}.")
        st.stop()

    # Connect to SageMaker endpoint and make prediction
    try:
        predictor = get_sagemaker_predictor(SAGEMAKER_ENDPOINT_NAME)
        # Send data to endpoint as a single row CSV. Predictor expects a list of lists.
        prediction_proba_raw = predictor.predict([input_data.tolist()])
        # The deserializer returns a list of lists, e.g., [['0.75']]. Extract the float.
        prediction_proba = float(prediction_proba_raw[0][0])

        st.subheader("Prediction Result:")
        st.write(f"**Churn Probability:** `{prediction_proba:.4f}`")

        # Determine binary prediction based on threshold
        if prediction_proba >= 0.5:
            st.error("This customer is **likely to churn!** 😬")
        else:
            st.success("This customer is **unlikely to churn.** 🎉")

        st.markdown("---")
        st.info("Note: A threshold of 0.5 is used for binary classification. You might adjust this based on business needs.")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.warning("Please ensure your SageMaker endpoint is deployed and the name in `app.py` is correct.")
        st.info(f"The configured endpoint name is: `{SAGEMAKER_ENDPOINT_NAME}`")
        st.info("If the endpoint was deleted, you will need to re-deploy it from your SageMaker notebook (run Cell 9 then Cell 10) and update this `app.py` file with the new endpoint name.")
        st.code("""
        # In your SageMaker notebook, re-run Cell 9 (training) if needed, then Cell 10 (deployment).
        # Make sure to copy the new endpoint name from Cell 10's output and update app.py.
        """)

st.sidebar.header("About")
st.sidebar.info("This is a Telco Customer Churn Prediction Engine built using Streamlit, powered by AWS SageMaker.")

st.sidebar.header("Feature Order (for debugging/reference)")
st.sidebar.code("\n".join(feature_columns))

