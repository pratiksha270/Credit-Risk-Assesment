import streamlit as st
import joblib
import numpy as np
import pandas as pd

rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment App")

st.markdown("""
This app predicts the **risk of loan default** using a machine learning model (Random Forest) and adjusts it using **Bayes' Theorem** if a borrower has missed an EMI (Equated Monthly Installment).

### How it works
- **Prior Risk**: Estimated using your loan and credit information.
- **Updated Risk**: Adjusted risk after missed EMI.
""")

with st.expander("What do the input fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: The total amount the borrower wants to borrow.
    - **Term**: Loan repayment period (e.g., 36 or 60 months).
    - **Interest Rate (%)**: Yearly interest percentage on the loan.
    - **Installment ($)**: Monthly repayment amount.
    - **Grade**: Assigned loan grade based on creditworthiness.
    - **Employment Length**: Years of employment history.
    - **Home Ownership**: Whether the applicant rents, owns, or mortgages a home.
    - **Annual Income ($)**: Declared yearly income.
    - **Purpose of Loan**: Reason for taking the loan.
    - **Debt vs Income (%)**: Proportion of debt to income.
    - **Missed Payments (Last 2 Years)**: Count of missed credit payments.
    - **Open Credit Lines**: Number of open credit accounts.
    - **Credit Utilization (%)**: Used credit relative to total credit limit.
    - **Total Credit Lines**: Total credit accounts (past + present).
    """)

st.header("Loan Application Form")

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
term = st.selectbox("Term", options=[" 36 months", " 60 months"])
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1)
installment = st.number_input("Installment ($)", min_value=50, max_value=2000)
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'])
emp_length = st.selectbox("Employment Length", options=["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"])
home_ownership = st.selectbox("Home Ownership", options=['MORTGAGE','RENT','OWN','OTHER'])
annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000)
purpose = st.selectbox("Purpose of Loan", options=[
    'car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
    'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding'
])
dti = st.slider("Debt vs Income (%)", min_value=0.0, max_value=50.0, step=0.1)
delinq_2yrs = st.number_input("Missed Payments (Last 2 Years)", min_value=0, max_value=10)
open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50)
revol_util = st.slider("Credit Utilization (%)", min_value=0.0, max_value=150.0, step=0.1)
total_acc = st.number_input("Total Credit Lines", min_value=1, max_value=100)

missed_emi = st.radio("Has the borrower missed an EMI?", options=["Yes", "No"])

# Scenario examples
with st.expander("Try a Sample Scenario"):
    sample = st.selectbox("Choose Example", options=["None", "Low Risk Applicant", "Moderate Risk Applicant", "High Risk Applicant"])
    if sample != "None":
        if sample == "Low Risk Applicant":
            loan_amnt = 8000
            term = " 36 months"
            int_rate = 7.0
            installment = 250
            grade = 'A'
            emp_length = "5 years"
            home_ownership = 'OWN'
            annual_inc = 90000
            purpose = 'credit_card'
            dti = 10.0
            delinq_2yrs = 0
            open_acc = 6
            revol_util = 20.0
            total_acc = 15
            missed_emi = "No"
        elif sample == "Moderate Risk Applicant":
            loan_amnt = 15000
            term = " 60 months"
            int_rate = 14.5
            installment = 400
            grade = 'C'
            emp_length = "3 years"
            home_ownership = 'RENT'
            annual_inc = 50000
            purpose = 'debt_consolidation'
            dti = 22.5
            delinq_2yrs = 1
            open_acc = 7
            revol_util = 60.0
            total_acc = 22
            missed_emi = "No"
        else:
            loan_amnt = 25000
            term = " 60 months"
            int_rate = 22.5
            installment = 750
            grade = 'F'
            emp_length = "< 1 year"
            home_ownership = 'RENT'
            annual_inc = 30000
            purpose = 'small_business'
            dti = 40.0
            delinq_2yrs = 3
            open_acc = 2
            revol_util = 130.0
            total_acc = 10
            missed_emi = "Yes"

if st.button("Predict Risk"):
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

        # Add verdict based on risk
        if updated_risk <= 0.33:
            verdict = "Low Risk"
        elif updated_risk <= 0.66:
            verdict = "Moderate Risk"
        else:
            verdict = "High Risk"

        st.subheader("Prediction Results")
        st.success(f"Prior Risk (Random Forest): {prior_risk:.3f}")
        st.info(f"Updated Risk (Bayes' Theorem): {updated_risk:.3f}")
        st.markdown(f"### Verdict: **{verdict}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
