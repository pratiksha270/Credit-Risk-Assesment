import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment App")

st.markdown("""
This app predicts the risk of loan default using a machine learning model (Random Forest) and adjusts it using Bayes' Theorem if a borrower has missed an EMI (Equated Monthly Installment).

### How it works
- **Prior Risk**: Estimated risk before borrower behavior is considered
- **Updated Risk**: Risk adjusted using Bayes' Theorem based on EMI behavior
""")

with st.expander("What do the fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: Requested loan principal
    - **Term**: Loan repayment period (e.g., 36 or 60 months)
    - **Interest Rate (%)**: Annual interest rate on loan
    - **Installment ($)**: Monthly payment amount
    - **Grade**: Credit quality grade (A to G)
    - **Employment Length**: Years of employment
    - **Home Ownership**: Applicant’s ownership status
    - **Annual Income ($)**: Yearly income
    - **Purpose of Loan**: Loan category (e.g., car, medical, credit card)
    - **Debt-to-Income Ratio**: Debt burden percentage
    - **Delinquencies (past 2 yrs)**: Missed payments
    - **Open Credit Lines**: Current active credit lines
    - **Revolving Utilization (%)**: Credit usage rate
    - **Total Credit Accounts**: Count of all credit accounts
    """)

st.header("Loan Application Form")

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
term = st.selectbox("Term", options=[" 36 months", " 60 months"])
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1)
installment = st.number_input("Installment ($)", min_value=50, max_value=2000)
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'])
emp_length = st.selectbox("Employment Length", options=['1 year', '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '< 1 year'])
home_ownership = st.selectbox("Home Ownership", options=['ANY', 'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT'])
annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000)
purpose = st.selectbox("Purpose of Loan", options=[
    'car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
    'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',
    'small_business', 'vacation', 'wedding'
])
dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", min_value=0, max_value=10)
open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50)
revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=150.0, step=0.1)
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

    df_input = pd.DataFrame([input_dict])

    try:
        for col in ['term','grade','emp_length','home_ownership','purpose']:
            le = label_encoders[col]
            df_input[col] = le.transform(df_input[col])

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
        st.success(f"Prior Risk (Random Forest): {prior_risk:.3f}")
        st.info(f"Updated Risk (Bayes' Theorem): {updated_risk:.3f}")

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(df_input)

        st.subheader("Feature Impact Explanation")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[1], df_input, plot_type="bar", show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
