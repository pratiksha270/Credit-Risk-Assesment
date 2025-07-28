import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime

# Load model and encoders
rf = joblib.load("rf_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment App")

st.markdown("""
This application helps assess the **risk of loan default** using machine learning (Random Forest Classifier). The risk is further adjusted based on missed EMI using **Bayes' Theorem**.

### How it works:
- **Prior Risk**: Risk predicted using loan application details.
- **Updated Risk**: Risk updated after missed EMI using Bayes' Theorem.
""")

with st.expander("What do these fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: The principal amount requested by the borrower.
    - **Term**: Duration of the loan (36 or 60 months).
    - **Interest Rate (%)**: Annual rate of interest applied on the loan.
    - **Installment ($)**: Monthly EMI calculated.
    - **Grade**: LendingClub-assigned credit grade (A to G).
    - **Employment Length**: How long the borrower has been employed.
    - **Home Ownership**: Whether borrower rents, owns, etc.
    - **Annual Income ($)**: Borrower's declared yearly income.
    - **Purpose**: The stated reason for taking the loan.
    - **Debt-to-Income Ratio**: Borrower’s debt compared to income.
    - **Delinquencies (past 2 yrs)**: Missed payments in last 2 years.
    - **Open Credit Lines**: Current active lines of credit.
    - **Revolving Utilization (%)**: How much of revolving credit is used.
    - **Total Credit Accounts**: Total accounts borrower has had.
    """)

st.header("Loan Application Form")

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
term = st.selectbox("Term", options=[' 36 months', ' 60 months'])
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1)
installment = st.number_input("Installment ($)", min_value=50, max_value=2000)
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'])
emp_length = st.selectbox("Employment Length", options=['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years'])
home_ownership = st.selectbox("Home Ownership", options=['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE', 'ANY'])
annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, step=1000)
purpose = st.selectbox("Purpose", options=[
    'credit_card', 'car', 'small_business', 'wedding', 'debt_consolidation',
    'home_improvement', 'major_purchase', 'medical', 'vacation', 'house',
    'moving', 'educational', 'renewable_energy', 'other'
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
        st.success(f"Prior Risk (Random Forest): {prior_risk:.3f}")
        st.info(f"Updated Risk (Bayes' Theorem): {updated_risk:.3f}")

        # Export option
        export_df = df_input.copy()
        export_df['Prior Risk'] = prior_risk
        export_df['Updated Risk'] = updated_risk
        export_filename = f"risk_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("Download Prediction as CSV", export_df.to_csv(index=False), file_name=export_filename, mime='text/csv')

    except Exception as e:
        st.error(f"Prediction failed: {e}")
