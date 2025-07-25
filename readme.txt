Credit Risk Assessment using Machine Learning and Bayesian Updating
# Credit Risk Assessment using Machine Learning and Bayesian Updating

### 🔗 Live Demo

[![Streamlit App](https://img.shields.io/badge/🚀_Try_App_on-Streamlit-orange?logo=streamlit&logoColor=white)](https://cfd9dvmd8wjccygatqevhr.streamlit.app/)

👉 Click the badge above to **interact with the Credit Risk Assessment App** and predict the risk of loan default using real-time inputs.

This project is a Credit Risk Assessment system that predicts the probability of a loan applicant defaulting using:
Random Forest Classifier
Bayesian Updating for risk refinement
Interactive Streamlit Web App for user-friendly prediction

Problem Statement
In the financial sector, it's critical to assess whether a borrower will default on a loan. Lenders often rely on historical data to make decisions. This project uses statistical and machine learning techniques to predict default risk and provides an interactive tool to assess it in real-time.

Key Features
Machine Learning Model trained on LendingClub dataset (2007–2018)
Monthly Time-Series Analysis using ARIMA to observe default trends
Bayesian Risk Updating — adjusts risk based on missed EMI feedback
Streamlit Web App with user input form to predict:
🔵 Prior Risk (model-based)
🟠 Updated Risk (Bayesian-adjusted)
Deployed on Streamlit Cloud with modern UI

Dataset
Source: LendingClub Accepted Loans 2007–2018Q4 (Kaggle)
Data Size: ~2.2 million records
Target Variable: loan_status → Converted to binary (Fully Paid = 0, Charged Off = 1)

Model Overview
Model Used: RandomForestClassifier

Selected Features:
Loan amount, term, interest rate, grade, employment length, home ownership.
Annual income, loan purpose, DTI, delinquencies, open accounts, credit utilization.

Risk Update:
Prior risk: Direct output from Random Forest
Updated risk: Applies Bayes' Theorem using prior + EMI status

Deploying on Streamlit Cloud
Push the following files to your GitHub repository:

credit_risk_app.py (main app)
rf_model.joblib (trained model)
label_encoders.joblib (encoders)
README.md

Go to streamlit.io/cloud and click “New App”.
Connect your GitHub account and select the repo.
Set credit_risk_app.py as the main file.
Click Deploy — your app will be live!

Repository Structure
File	Description
credit_risk_app.py	Streamlit frontend + prediction logic
rf_model.joblib	Trained Random Forest model
label_encoders.joblib	Saved encoders for categorical features
CreditRiskAss.ipynb	Notebook with data preprocessing and training
README.md	Project documentation

Future Improvements
Add bulk CSV upload support
Use SHAP for model explainability
Integrate ARIMA risk forecasting into the app UI

LinkedIn
GitHub

