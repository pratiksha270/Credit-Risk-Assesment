import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model and encoders
rf = joblib.load("rf_model.joblib")  # Make sure this file is saved from Colab
le_dict = joblib.load("label_encoders.joblib")  # Dictionary of LabelEncoders

# Define Bayes' update function
def bayesian_update(prior, p_evidence_given_default, p_evidence):
    if p_evidence == 0:
        return prior
    return (p_evidence_given_default * prior) / p_evidence

# Pre-calculated from your training data
P_EVIDENCE_GIVEN_DEFAULT = 0.21  # Example: P(missed EMI | default)
P_EVIDENCE = 0.19  # Example: P(missed EMI overall)

st.title("Credit Risk Assessment App")
st.write("Enter loan applicant information to assess default risk:")

# User inputs
loan_amnt = st.number_input("Loan Amount ($)", 500, 50000, 15000)
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 18.0)
annual_inc = st.number_input("Annual Income ($)", 10000, 200000, 45000)
dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 20.0)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", 0, 10, 0)
grade = st.selectbox("Grade", options=le_dict['grade'].classes_)
emp_length = st.selectbox("Employment Length", options=le_dict['emp_length'].classes_)
home_ownership = st.selectbox("Home Ownership", options=le_dict['home_ownership'].classes_)
purpose = st.selectbox("Purpose of Loan", options=le_dict['purpose'].classes_)
open_acc = st.number_input("Open Credit Accounts", 0, 50, 6)
revol_util = st.slider("Revolving Credit Utilization (%)", 0.0, 100.0, 40.0)

# Prediction trigger
if st.button("Predict Risk"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'grade': le_dict['grade'].transform([grade])[0],
        'emp_length': le_dict['emp_length'].transform([emp_length])[0],
        'home_ownership': le_dict['home_ownership'].transform([home_ownership])[0],
        'purpose': le_dict['purpose'].transform([purpose])[0],
        'open_acc': open_acc,
        'revol_util': revol_util
    }])

    # Predict with RandomForest
    prior_risk = rf.predict_proba(input_data)[0][1]
    missed_emi = delinq_2yrs > 0
    updated_risk = bayesian_update(prior_risk, P_EVIDENCE_GIVEN_DEFAULT, P_EVIDENCE) if missed_emi else prior_risk

    # Output results
    st.success(f"Predicted Prior Risk of Default: {prior_risk:.2%}")
    if missed_emi:
        st.warning(f"Updated Risk After Missed EMI: {updated_risk:.2%}")
    else:
        st.info("No missed EMI history — risk unchanged.")
