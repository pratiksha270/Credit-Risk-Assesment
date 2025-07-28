import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
import os

# Load model and encoders
rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment App")

st.markdown("""
This app predicts the risk of loan default using a machine learning model (Random Forest) and adjusts it using Bayes' Theorem if a borrower has missed an EMI.

How it works:
- Prior Risk: Risk estimated based on loan details using ML model.
- Updated Risk: Adjusted risk after missed EMI using Bayes' Theorem.
""")

with st.expander("What do the fields mean?"):
    st.markdown("""
    - Loan Amount: The principal amount requested.
    - Term: Duration of the loan (36 or 60 months).
    - Interest Rate: Annual interest rate on the loan.
    - Installment: Monthly payment amount.
    - Grade: Credit grade assigned (A to G).
    - Employment Length: Number of years employed.
    - Home Ownership: Whether the applicant owns, rents, etc.
    - Annual Income: Yearly income.
    - Purpose: Why the applicant is taking the loan.
    - DTI: Debt-to-Income ratio.
    - Delinquencies: Number of times borrower was late.
    - Open Credit Lines: Active lines of credit.
    - Revolving Utilization: Credit card utilization.
    - Total Credit Accounts: Total credit accounts held.
    """)

st.header("Loan Application Form")

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500, value=15000)
term = st.selectbox("Term", options=[' 36 months', ' 60 months'], index=0)
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1, value=13.5)
installment = st.number_input("Installment ($)", min_value=50, max_value=2000, value=450)
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'], index=1)
emp_length = st.selectbox("Employment Length", options=['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years'], index=5)
home_ownership = st.selectbox("Home Ownership", options=['RENT','OWN','MORTGAGE','OTHER','NONE','ANY'], index=0)
annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000, value=65000)
purpose = st.selectbox("Purpose of Loan", options=[
    'debt_consolidation','credit_card','home_improvement','small_business','car','major_purchase','house',
    'medical','moving','vacation','wedding','other','educational','renewable_energy'
], index=0)
dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, step=0.1, value=15.0)
delinq_2yrs = st.number_input("Delinquencies (past 2 yrs)", min_value=0, max_value=10, value=0)
open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=6)
revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=150.0, step=0.1, value=40.0)
total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100, value=22)
missed_emi = st.radio("Has the borrower missed an EMI?", options=["Yes", "No"], index=1)

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
        st.write(f"Prior Risk (Random Forest): {prior_risk:.3f}")
        st.write(f"Updated Risk (Bayes' Theorem): {updated_risk:.3f}")

        # SHAP explanation
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(df_input)

        st.subheader("Top Risk Factors (SHAP)")
        shap_df = pd.DataFrame({
            'Feature': df_input.columns,
            'SHAP Value': shap_values[1][0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)
        st.dataframe(shap_df.head(5))

        # Export logic
        export_df = df_input.copy()
        export_df['prior_risk'] = prior_risk
        export_df['updated_risk'] = updated_risk
        export_df['missed_emi'] = missed_emi
        for i in range(len(shap_df)):
            export_df[f"shap_{shap_df.iloc[i, 0]}"] = shap_df.iloc[i, 1]

        if not os.path.exists("predictions.csv"):
            export_df.to_csv("predictions.csv", index=False)
        else:
            export_df.to_csv("predictions.csv", mode='a', header=False, index=False)

        st.download_button(
            label="Download Prediction Record",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name='credit_risk_prediction.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
