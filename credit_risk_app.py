import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and encoder
model = joblib.load("rf_model.joblib")
encoders = joblib.load("label_encoders.joblib")

# Sample applicant profiles
sample_profiles = {
    "Low Risk Applicant": {
        "term": "36 months",
        "int_rate": 8.5,
        "grade": "A",
        "purpose": "credit_card",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 85000,
        "dti": 10.0,
        "delinq_2yrs": 0,
        "open_acc": 8,
        "revol_util": 20.5,
        "total_acc": 25,
        "missed_emi": "No",
    },
    "Moderate Risk Applicant": {
        "term": "36 months",
        "int_rate": 14.2,
        "grade": "C",
        "purpose": "debt_consolidation",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 48000,
        "dti": 17.0,
        "delinq_2yrs": 1,
        "open_acc": 6,
        "revol_util": 45.0,
        "total_acc": 20,
        "missed_emi": "No",
    },
    "High Risk Applicant": {
        "term": "60 months",
        "int_rate": 22.5,
        "grade": "E",
        "purpose": "small_business",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "annual_inc": 30000,
        "dti": 28.0,
        "delinq_2yrs": 3,
        "open_acc": 4,
        "revol_util": 80.0,
        "total_acc": 15,
        "missed_emi": "Yes",
    }
}

# Sidebar: Pre-filled options
st.sidebar.title("Sample Profiles")
selected_profile = st.sidebar.selectbox("Choose a sample applicant:", [""] + list(sample_profiles.keys()))
user_input = sample_profiles.get(selected_profile, {})

# App title
st.title("Credit Risk Assessment")

# UI inputs
term = st.selectbox("Loan Term", ["36 months", "60 months"], index=["36 months", "60 months"].index(user_input.get("term", "36 months")))
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, float(user_input.get("int_rate", 10.0)))
grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(user_input.get("grade", "B")))
purpose = st.selectbox("Loan Purpose", list(encoders['purpose'].classes_), index=encoders['purpose'].classes_.tolist().index(user_input.get("purpose", "debt_consolidation")))
emp_length = st.selectbox("Employment Length", list(encoders['emp_length'].classes_), index=encoders['emp_length'].classes_.tolist().index(user_input.get("emp_length", "10+ years")))
home_ownership = st.selectbox("Home Ownership", list(encoders['home_ownership'].classes_), index=encoders['home_ownership'].classes_.tolist().index(user_input.get("home_ownership", "RENT")))
annual_inc = st.number_input("Annual Income ($)", value=float(user_input.get("annual_inc", 50000)))
dti = st.slider("Debt vs Income (%)", 0.0, 50.0, float(user_input.get("dti", 20.0)))
delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", 0, 10, int(user_input.get("delinq_2yrs", 0)))
open_acc = st.number_input("Open Credit Lines", 0, 50, int(user_input.get("open_acc", 5)))
revol_util = st.slider("Credit Utilization (%)", 0.0, 150.0, float(user_input.get("revol_util", 30.0)))
total_acc = st.number_input("Total Credit Accounts", 0, 100, int(user_input.get("total_acc", 15)))
missed_emi = st.radio("Has the borrower missed an EMI?", ["Yes", "No"], index=["Yes", "No"].index(user_input.get("missed_emi", "No")))

# Prediction
if st.button("Predict Credit Risk"):
    input_dict = {
        "term": encoders["term"].transform([term])[0],
        "int_rate": int_rate,
        "grade": encoders["grade"].transform([grade])[0],
        "purpose": encoders["purpose"].transform([purpose])[0],
        "emp_length": encoders["emp_length"].transform([emp_length])[0],
        "home_ownership": encoders["home_ownership"].transform([home_ownership])[0],
        "annual_inc": annual_inc,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc,
        "revol_util": revol_util,
        "total_acc": total_acc,
    }

    X = pd.DataFrame([input_dict])
    prior_risk = model.predict_proba(X)[0][1]

    # Bayes adjustment
    emi_factor = 2.2 if missed_emi == "Yes" else 0.7
    adjusted_risk = (prior_risk * emi_factor) / ((prior_risk * emi_factor) + (1 - prior_risk))

    # Verdict
    if adjusted_risk < 0.4:
        verdict = "Low Risk"
    elif adjusted_risk < 0.7:
        verdict = "Moderate Risk"
    else:
        verdict = "High Risk"

    # Display
    st.subheader("Prediction Results")
    st.write(f"Prior Risk (Model): {round(prior_risk, 3)}")
    st.write(f"Adjusted Risk (Bayes' Theorem): {round(adjusted_risk, 3)}")
    st.success(verdict)
