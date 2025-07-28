import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment App")

st.markdown("""
This app predicts the risk of loan default using a machine learning model (Random Forest) and adjusts it using Bayes' Theorem if the borrower misses an EMI (Equated Monthly Installment).
""")

with st.expander("What do these fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: Principal amount requested.
    - **Loan Term**: Duration in months (36 or 60).
    - **Interest Rate (%)**: Annual interest rate of the loan.
    - **Monthly EMI ($)**: Expected monthly payment.
    - **Credit Grade**: Loan grade assigned (A to G).
    - **Employment Duration**: How long the borrower has been employed.
    - **Home Ownership Status**: Rent, Own, Mortgage, etc.
    - **Annual Income ($)**: Yearly income declared.
    - **Loan Purpose**: Reason for loan.
    - **Debt vs Income (%)**: Ratio of existing debt to income.
    - **Missed Payments (past 2 yrs)**: Defaults in recent 2 years.
    - **Open Credit Lines**: Number of active credit lines.
    - **Credit Utilization (%)**: Revolving credit use.
    - **Total Credit Accounts**: Lifetime credit accounts.
    """)

# Pre-filled scenarios
st.sidebar.header("Sample Profiles")
scenarios = {
    "Select a Profile": {},
    "Low Risk Applicant": {
        "loan_amnt": 5000,
        "term": " 36 months",
        "int_rate": 8.5,
        "installment": 160,
        "grade": "A",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 85000,
        "purpose": "credit_card",
        "dti": 10.0,
        "delinq_2yrs": 0,
        "open_acc": 8,
        "revol_util": 20.5,
        "total_acc": 25
    },
    "Moderate Risk Applicant": {
        "loan_amnt": 15000,
        "term": " 36 months",
        "int_rate": 15.2,
        "installment": 480,
        "grade": "C",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 50000,
        "purpose": "debt_consolidation",
        "dti": 20.0,
        "delinq_2yrs": 1,
        "open_acc": 6,
        "revol_util": 55.0,
        "total_acc": 18
    },
    "High Risk Applicant": {
        "loan_amnt": 25000,
        "term": " 60 months",
        "int_rate": 26.9,
        "installment": 760,
        "grade": "G",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "annual_inc": 25000,
        "purpose": "small_business",
        "dti": 42.0,
        "delinq_2yrs": 3,
        "open_acc": 4,
        "revol_util": 100.0,
        "total_acc": 12
    }
}
selected_profile = st.sidebar.selectbox("Choose a sample applicant:", list(scenarios.keys()))
def_profile = scenarios[selected_profile] if selected_profile != "Select a Profile" else {}

# Form
st.header("Loan Application Form")

loan_amnt = st.number_input("Loan Amount ($)", value=def_profile.get("loan_amnt", 10000), step=500)
term = st.selectbox("Loan Term", options=[" 36 months", " 60 months"], index=[" 36 months", " 60 months"].index(def_profile.get("term", " 36 months")))
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, value=def_profile.get("int_rate", 10.0))
installment = st.number_input("Monthly EMI ($)", value=def_profile.get("installment", 300))
grade = st.selectbox("Credit Grade", options=['A','B','C','D','E','F','G'], index=['A','B','C','D','E','F','G'].index(def_profile.get("grade", "B")))
emp_length = st.selectbox("Employment Duration", options=["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"], index=["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"].index(def_profile.get("emp_length", "5 years")))
home_ownership = st.selectbox("Home Ownership Status", options=['RENT','OWN','MORTGAGE','OTHER'], index=['RENT','OWN','MORTGAGE','OTHER'].index(def_profile.get("home_ownership", "RENT")))
annual_inc = st.number_input("Annual Income ($)", value=def_profile.get("annual_inc", 60000), step=1000)
purpose = st.selectbox("Loan Purpose", options=[
    'credit_card', 'car', 'small_business', 'wedding', 'debt_consolidation',
    'home_improvement', 'major_purchase', 'medical', 'vacation', 'house', 'moving'
], index=[
    'credit_card', 'car', 'small_business', 'wedding', 'debt_consolidation',
    'home_improvement', 'major_purchase', 'medical', 'vacation', 'house', 'moving'
].index(def_profile.get("purpose", 'credit_card')))
dti = st.slider("Debt vs Income (%)", 0.0, 50.0, value=def_profile.get("dti", 15.0))
delinq_2yrs = st.number_input("Missed Payments (past 2 yrs)", min_value=0, max_value=10, value=def_profile.get("delinq_2yrs", 0))
open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=def_profile.get("open_acc", 6))
revol_util = st.slider("Credit Utilization (%)", 0.0, 150.0, value=def_profile.get("revol_util", 45.0))
total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100, value=def_profile.get("total_acc", 20))

missed_emi = st.radio("Has the borrower missed an EMI?", options=["Yes", "No"])

if st.button("Predict Credit Risk"):
    input_dict = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'purpose': purpose,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'open_acc': open_acc,
        'revol_util': revol_util,
        'total_acc': total_acc
    }
    df_input = pd.DataFrame([input_dict])

    for col in ['term','grade','emp_length','home_ownership','purpose']:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    try:
        prior_risk = rf.predict_proba(df_input)[0][1]
        P_prior = prior_risk
        P_miss_given_default = 0.85
        P_miss_given_no_default = 0.25

        if missed_emi == "Yes":
            numerator = P_miss_given_default * P_prior
            denominator = (P_miss_given_default * P_prior) + (P_miss_given_no_default * (1 - P_prior))
            updated_risk = numerator / denominator
        else:
            updated_risk = prior_risk

        st.subheader("Prediction Results")
        st.write(f"Prior Risk (Model): {prior_risk:.3f}")
        st.write(f"Adjusted Risk (Bayes' Theorem): {updated_risk:.3f}")

        if updated_risk <= 0.3:
            st.success("Low Risk")
        elif updated_risk <= 0.6:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
