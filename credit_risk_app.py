import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("rf_model.joblib")
encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment")

st.markdown("""
This app predicts the risk of loan default using a machine learning model (Random Forest) and adjusts it using Bayes' Theorem if the borrower has missed an EMI.
""")

with st.expander("What do the fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: Amount requested by the applicant
    - **Loan Term**: Loan repayment duration (36 or 60 months)
    - **Interest Rate (%)**: Interest on loan annually
    - **Installment ($)**: Monthly repayment amount
    - **Credit Grade**: Grade based on creditworthiness
    - **Employment Duration**: Years of employment
    - **Home Ownership Status**: RENT/OWN/MORTGAGE
    - **Annual Income ($)**: Total annual income declared
    - **Loan Purpose**: Reason for loan
    - **Debt vs Income (%)**: Ratio of debt to income
    - **Delinquencies in Past 2 Years**: Count of past defaults
    - **Open Credit Lines**: Active lines of credit
    - **Credit Utilization (%)**: % of revolving credit used
    - **Total Credit Accounts**: Total number of credit accounts
    """)

# Pre-filled scenarios
def get_sample_input(scenario):
    if scenario == "Low Risk Applicant":
        return {
            "loan_amnt": 5000,
            "term": " 36 months",
            "int_rate": 7.0,
            "installment": 150,
            "grade": "A",
            "emp_length": "10+ years",
            "home_ownership": "OWN",
            "annual_inc": 85000,
            "purpose": "credit_card",
            "dti": 10.0,
            "delinq_2yrs": 0,
            "open_acc": 12,
            "revol_util": 25.0,
            "total_acc": 30
        }
    elif scenario == "Moderate Risk Applicant":
        return {
            "loan_amnt": 15000,
            "term": " 36 months",
            "int_rate": 14.5,
            "installment": 450,
            "grade": "C",
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "annual_inc": 65000,
            "purpose": "debt_consolidation",
            "dti": 25.0,
            "delinq_2yrs": 1,
            "open_acc": 8,
            "revol_util": 45.0,
            "total_acc": 22
        }
    elif scenario == "High Risk Applicant":
        return {
            "loan_amnt": 25000,
            "term": " 60 months",
            "int_rate": 24.0,
            "installment": 850,
            "grade": "G",
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "annual_inc": 30000,
            "purpose": "small_business",
            "dti": 45.0,
            "delinq_2yrs": 3,
            "open_acc": 4,
            "revol_util": 110.0,
            "total_acc": 10
        }
    return {}

st.header("Loan Application Form")

scenario = st.selectbox("Choose Sample Scenario (Optional)", ["None", "Low Risk Applicant", "Moderate Risk Applicant", "High Risk Applicant"])
user_input = get_sample_input(scenario) if scenario != "None" else {}

loan_amnt = st.number_input("Loan Amount ($)", 500, 50000, value=user_input.get("loan_amnt", 10000), step=500)
term_options = [' 36 months', ' 60 months']
term_display = {" 36 months": "36 months", " 60 months": "60 months"}
term_selected = st.selectbox("Loan Term", [term_display[k] for k in term_options],
                             index=term_options.index(user_input.get("term", " 36 months")))
term = [k for k, v in term_display.items() if v == term_selected][0]

int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, value=user_input.get("int_rate", 10.0), step=0.1)
installment = st.number_input("Installment ($)", 50, 2000, value=user_input.get("installment", 250))
grade = st.selectbox("Credit Grade", list(encoders["grade"].classes_),
                     index=list(encoders["grade"].classes_).index(user_input.get("grade", "B")))
emp_length = st.selectbox("Employment Duration", list(encoders["emp_length"].classes_),
                          index=list(encoders["emp_length"].classes_).index(user_input.get("emp_length", "5 years")))
home_ownership = st.selectbox("Home Ownership Status", list(encoders["home_ownership"].classes_),
                               index=list(encoders["home_ownership"].classes_).index(user_input.get("home_ownership", "RENT")))
annual_inc = st.number_input("Annual Income ($)", 10000, 500000, value=user_input.get("annual_inc", 50000), step=1000)
purpose = st.selectbox("Loan Purpose", list(encoders["purpose"].classes_),
                       index=list(encoders["purpose"].classes_).index(user_input.get("purpose", "credit_card")))
dti = st.slider("Debt vs Income (%)", 0.0, 50.0, value=user_input.get("dti", 20.0), step=0.1)
delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", 0, 10, value=user_input.get("delinq_2yrs", 0))
open_acc = st.number_input("Open Credit Lines", 0, 50, value=user_input.get("open_acc", 6))
revol_util = st.slider("Credit Utilization (%)", 0.0, 150.0, value=user_input.get("revol_util", 30.0), step=0.1)
total_acc = st.number_input("Total Credit Accounts", 1, 100, value=user_input.get("total_acc", 20))

missed_emi = st.radio("Has the borrower missed an EMI?", ["Yes", "No"])

if st.button("Predict Risk"):
    try:
        data = {
            "loan_amnt": loan_amnt,
            "term": encoders["term"].transform([term])[0],
            "int_rate": int_rate,
            "installment": installment,
            "grade": encoders["grade"].transform([grade])[0],
            "emp_length": encoders["emp_length"].transform([emp_length])[0],
            "home_ownership": encoders["home_ownership"].transform([home_ownership])[0],
            "annual_inc": annual_inc,
            "purpose": encoders["purpose"].transform([purpose])[0],
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "open_acc": open_acc,
            "revol_util": revol_util,
            "total_acc": total_acc
        }
        X = pd.DataFrame([data])
        prior_risk = model.predict_proba(X)[0][1]

        P_prior = prior_risk
        P_miss_given_default = 0.85
        P_miss_given_no_default = 0.25

        if missed_emi == "Yes":
            numerator = P_miss_given_default * P_prior
            denominator = (P_miss_given_default * P_prior) + (P_miss_given_no_default * (1 - P_prior))
            updated_risk = numerator / denominator
        else:
            updated_risk = prior_risk

        if updated_risk < 0.4:
            verdict = "Low Risk"
        elif updated_risk < 0.7:
            verdict = "Moderate Risk"
        else:
            verdict = "High Risk"

        st.subheader("Prediction Result")
        st.write(f"Prior Risk (ML Model): {prior_risk:.3f}")
        st.write(f"Adjusted Risk (Bayes' Theorem): {updated_risk:.3f}")
        st.success(f"Final Verdict: {verdict}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
