import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and label encoders
rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment", layout="centered")

st.title("Credit Risk Assessment App")
st.markdown("""
This app predicts the risk of loan default using a machine learning model and adjusts the risk using Bayes' Theorem if a borrower misses an EMI.

- **Prior Risk**: Based on initial loan details.
- **Adjusted Risk**: Updated risk if EMI is missed.
""")

with st.expander("What do the fields mean?"):
    st.markdown("""
    - **Loan Amount**: Principal amount requested by borrower.
    - **Term**: Duration of the loan (36 or 60 months).
    - **Interest Rate**: Annual interest rate on the loan.
    - **Installment**: Monthly payment amount.
    - **Grade**: LendingClub assigned credit grade.
    - **Employment Length**: Years of employment.
    - **Home Ownership**: Borrower’s home ownership status.
    - **Annual Income**: Reported yearly income.
    - **Purpose**: Reason for the loan.
    - **DTI**: Debt-to-income ratio.
    - **Delinquencies**: Past delinquencies in 2 years.
    - **Open Credit Lines**: Active credit lines.
    - **Revolving Utilization**: Utilized revolving credit.
    - **Total Accounts**: Total number of credit lines.
    """)

st.header("Loan Application Form")

# Input fields
loan_amnt = st.number_input("Loan Amount ($)", value=15000, step=500)
term = st.selectbox("Term", options=[' 36 months', ' 60 months'])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, step=0.1)
installment = st.number_input("Installment ($)", value=450)
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'])
emp_length = st.selectbox("Employment Length", options=[
    '1 year', '2 years', '3 years', '4 years', '5 years', '6 years',
    '7 years', '8 years', '9 years', '10+ years', '< 1 year'
])
home_ownership = st.selectbox("Home Ownership", options=[
    'MORTGAGE', 'RENT', 'OWN', 'OTHER', 'ANY', 'NONE'
])
annual_inc = st.number_input("Annual Income ($)", value=65000, step=1000)
purpose = st.selectbox("Purpose", options=[
    'credit_card', 'car', 'small_business', 'wedding', 'debt_consolidation',
    'home_improvement', 'major_purchase', 'medical', 'vacation', 'house',
    'moving', 'educational', 'renewable_energy', 'other'
])
dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", min_value=0, max_value=10)
open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50)
revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, step=0.1)
total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100)
missed_emi = st.radio("Has the borrower missed an EMI?", options=["Yes", "No"])

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

    input_df = pd.DataFrame([input_dict])

    # Encode categorical fields
    for col in ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']:
        encoder = label_encoders[col]
        input_df[col] = encoder.transform(input_df[col])

    try:
        prior_risk = rf.predict_proba(input_df)[0][1]
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
        st.success(f"Prior Risk (Before EMI): {prior_risk:.3f}")
        st.info(f"Adjusted Risk (After EMI Status): {updated_risk:.3f}")

        verdict = (
            "⚠️ High Risk - Likely to default." if updated_risk > 0.5 else
            "✅ Low Risk - Unlikely to default."
        )
        st.warning(verdict)

        # CSV Export
        result_data = input_dict.copy()
        result_data["prior_risk"] = prior_risk
        result_data["adjusted_risk"] = updated_risk
        result_data["emi_missed"] = missed_emi
        df_result = pd.DataFrame([result_data])
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name=f"credit_risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
