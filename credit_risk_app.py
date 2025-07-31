import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("rf_model.joblib")
encoders = joblib.load("label_encoders.joblib")

st.set_page_config(page_title="Credit Risk Assessment App")
st.title("Credit Risk Assessment")

st.markdown("""
This application predicts the probability of a loan default using a machine learning model trained on LendingClub data. It adjusts the prediction using Bayes' Theorem if an EMI has been missed.
""")

with st.expander("What do the input fields mean?"):
    st.markdown("""
    - **Loan Amount ($)**: Requested loan principal.
    - **Term**: Duration of loan (36 or 60 months).
    - **Interest Rate (%)**: Annual interest rate applied.
    - **Installment ($)**: Fixed monthly payment.
    - **Grade**: Credit grade assigned by lender.
    - **Employment Length**: Years of employment.
    - **Home Ownership**: Applicant's home ownership status.
    - **Annual Income ($)**: Total yearly income.
    - **Purpose**: Reason for loan.
    - **Debt vs Income (%)**: Debt-to-Income ratio.
    - **Delinquencies in Past 2 Years**: Number of missed payments.
    - **Open Credit Lines**: Number of open accounts.
    - **Credit Utilization (%)**: Percentage of credit limit being used.
    - **Total Credit Accounts**: Total credit accounts.
    """)

# Sample Scenarios
sample_option = st.selectbox("Choose a sample scenario (optional):", ["None", "Low Risk Applicant", "Moderate Risk Applicant", "High Risk Applicant"])

if sample_option == "Low Risk Applicant":
    inputs = {
        'loan_amnt': 8000,
        'term': ' 36 months',
        'int_rate': 7.5,
        'installment': 250,
        'grade': 'A',
        'emp_length': '10+ years',
        'home_ownership': 'OWN',
        'annual_inc': 85000,
        'purpose': 'credit_card',
        'dti': 8.0,
        'delinq_2yrs': 0,
        'open_acc': 12,
        'revol_util': 25.0,
        'total_acc': 28,
        'missed_emi': "No"
    }
elif sample_option == "Moderate Risk Applicant":
    inputs = {
        'loan_amnt': 10000,
        'term': ' 36 months',
        'int_rate': 13.5,
        'installment': 350,
        'grade': 'C',
        'emp_length': '3 years',
        'home_ownership': 'RENT',
        'annual_inc': 45000,
        'purpose': 'home_improvement',
        'dti': 22.0,
        'delinq_2yrs': 0,
        'open_acc': 5,
        'revol_util': 40.0,
        'total_acc': 18,
        'missed_emi': "No"
    }

elif sample_option == "High Risk Applicant":
    inputs = {
        'loan_amnt': 25000,
        'term': ' 60 months',
        'int_rate': 21.0,
        'installment': 700,
        'grade': 'F',
        'emp_length': '< 1 year',
        'home_ownership': 'RENT',
        'annual_inc': 30000,
        'purpose': 'small_business',
        'dti': 35.0,
        'delinq_2yrs': 3,
        'open_acc': 3,
        'revol_util': 95.0,
        'total_acc': 10,
        'missed_emi': "Yes"
    }
else:
    inputs = {}

# UI for Inputs
loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500, value=inputs.get('loan_amnt', 10000))
term = st.selectbox("Term", options=[' 36 months', ' 60 months'], index=[' 36 months', ' 60 months'].index(inputs.get('term', ' 36 months')))
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, value=inputs.get('int_rate', 12.0))
installment = st.number_input("Installment ($)", min_value=50, max_value=2000, value=inputs.get('installment', 300))
grade = st.selectbox("Grade", options=['A','B','C','D','E','F','G'], index=['A','B','C','D','E','F','G'].index(inputs.get('grade', 'B')))
emp_length = st.selectbox("Employment Length", options=['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years'], index=['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years'].index(inputs.get('emp_length', '5 years')))
home_ownership = st.selectbox("Home Ownership", options=['MORTGAGE','RENT','OWN','OTHER'], index=['MORTGAGE','RENT','OWN','OTHER'].index(inputs.get('home_ownership', 'RENT')))
annual_inc = st.number_input("Annual Income ($)", 10000, 500000, value=inputs.get('annual_inc', 50000), step=1000)
purpose = st.selectbox("Purpose", options=encoders['purpose'].classes_.tolist(), index=encoders['purpose'].classes_.tolist().index(inputs.get('purpose', 'credit_card')))
dti = st.slider("Debt vs Income (%)", 0.0, 50.0, value=inputs.get('dti', 10.0))
delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", 0, 10, value=inputs.get('delinq_2yrs', 0))
open_acc = st.number_input("Open Credit Lines", 0, 50, value=inputs.get('open_acc', 5))
revol_util = st.slider("Credit Utilization (%)", 0.0, 150.0, value=inputs.get('revol_util', 40.0))
total_acc = st.number_input("Total Credit Accounts", 1, 100, value=inputs.get('total_acc', 20))
missed_emi = st.radio("Has the borrower missed an EMI?", ["Yes", "No"], index=["Yes", "No"].index(inputs.get('missed_emi', "No")))

if st.button("Predict Risk"):
    try:
        df_input = pd.DataFrame([{
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
        }])

        prior = model.predict_proba(df_input)[0][1]
        P_miss_given_default = 0.85
        P_miss_given_no_default = 0.25

        if missed_emi == "Yes":
            updated = (P_miss_given_default * prior) / ((P_miss_given_default * prior) + (P_miss_given_no_default * (1 - prior)))
        else:
            updated = prior

        if updated < 0.4:
            verdict = "Low Risk"
        elif updated < 0.7:
            verdict = "Moderate Risk"
        else:
            verdict = "High Risk"

        st.subheader("Prediction Results")
        st.write(f"Model Estimated Risk (Prior): {prior:.3f}")
        st.write(f"Adjusted Risk after EMI Check: {updated:.3f}")
        st.success(f"Verdict: {verdict}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
