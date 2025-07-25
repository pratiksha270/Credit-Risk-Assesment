import streamlit as st
import joblib
import numpy as np
import pandas as pd

rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Predictor")
st.title("💳 Credit Risk Predictor")

with st.expander("ℹ️ What do these fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: Requested principal amount.
    - **Term**: Loan duration (36 or 60 months).
    - **Interest Rate (%)**: Annual interest rate.
    - **Installment ($)**: Monthly payment.
    - **Grade**: Credit grade assigned by lender.
    - **Employment Length**: Years employed.
    - **Home Ownership**: Homeownership status.
    - **Annual Income ($)**: Stated annual income.
    - **Purpose of Loan**: Why the loan is needed.
    - **Debt-to-Income Ratio**: Ratio of debts to income.
    - **Delinquencies (past 2 yrs)**: Count of past due payments.
    - **Open Credit Lines**: Number of active credit lines.
    - **Revolving Utilization (%)**: Utilized revolving credit.
    - **Total Credit Accounts**: Total credit accounts held.
    """)

st.header("📋 Enter Applicant Info")

loan_amnt = st.number_input("Loan Amount ($)", 500, 50000, step=500)
term = st.selectbox("Term", ["36 months", "60 months"])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, step=0.1)
installment = st.number_input("Installment ($)", 50, 2000)
grade = st.selectbox("Grade", ['A','B','C','D','E','F','G'])
emp_length = st.selectbox("Employment Length", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"])
home_ownership = st.selectbox("Home Ownership", ['MORTGAGE','RENT','OWN','OTHER'])
annual_inc = st.number_input("Annual Income ($)", 10000, 500000, step=1000)
purpose = st.selectbox("Purpose of Loan", [
    'credit_card', 'car', 'small_business', 'wedding', 'debt_consolidation',
    'home_improvement', 'major_purchase', 'medical', 'vacation', 'house', 'moving'
])
dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", 0, 10)
open_acc = st.number_input("Open Credit Lines", 0, 50)
revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, step=0.1)
total_acc = st.number_input("Total Credit Accounts", 1, 100)
missed_emi = st.radio("Has borrower missed an EMI?", ["Yes", "No"])

if st.button("📊 Predict Risk"):
    try:
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
        for col in ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']:
            df_input[col] = label_encoders[col].transform(df_input[col])

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

        st.subheader("📈 Prediction")
        st.success(f"🔵 Prior Risk: {prior_risk:.3f}")
        st.info(f"🟠 Updated Risk (Bayesian): {updated_risk:.3f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
