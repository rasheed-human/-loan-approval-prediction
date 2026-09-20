import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    applicant_id = st.number_input(
        "Applicant ID",
        min_value=1,
        value=1
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=0
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=50000.0
    )

with col2:
    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=100000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=120
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Car", "Education", "Home", "Personal"]
    )

with col3:
    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Government", "MNC", "Private", "Unemployed"]
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.30
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

st.divider()

if st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
):

    education_encoded = (
        0 if education_level == "Graduate" else 1
    )

    input_data = pd.DataFrame({

        "Applicant_ID": [applicant_id],
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Existing_Loans": [existing_loans],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Education_Level": [education_encoded],

        "Employment_Status_Salaried": [
            1 if employment_status == "Salaried" else 0
        ],

        "Employment_Status_Self-employed": [
            1 if employment_status == "Self-employed" else 0
        ],

        "Employment_Status_Unemployed": [
            1 if employment_status == "Unemployed" else 0
        ],

        "Marital_Status_Single": [
            1 if marital_status == "Single" else 0
        ],

        "Loan_Purpose_Car": [
            1 if loan_purpose == "Car" else 0
        ],

        "Loan_Purpose_Education": [
            1 if loan_purpose == "Education" else 0
        ],

        "Loan_Purpose_Home": [
            1 if loan_purpose == "Home" else 0
        ],

        "Loan_Purpose_Personal": [
            1 if loan_purpose == "Personal" else 0
        ],

        "Property_Area_Semiurban": [
            1 if property_area == "Semiurban" else 0
        ],

        "Property_Area_Urban": [
            1 if property_area == "Urban" else 0
        ],

        "Gender_Male": [
            1 if gender == "Male" else 0
        ],

        "Employer_Category_Government": [
            1 if employer_category == "Government" else 0
        ],

        "Employer_Category_MNC": [
            1 if employer_category == "MNC" else 0
        ],

        "Employer_Category_Private": [
            1 if employer_category == "Private" else 0
        ],

        "Employer_Category_Unemployed": [
            1 if employer_category == "Unemployed" else 0
        ],

        "DTI_Ratio_sq": [
            dti_ratio ** 2
        ],

        "Credit_Score_sq": [
            credit_score ** 2
        ],

        "Applicant_Income_log": [
            np.log1p(applicant_income)
        ]
    })

    feature_order = [
        "Applicant_ID",
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Education_Level",
        "Employment_Status_Salaried",
        "Employment_Status_Self-employed",
        "Employment_Status_Unemployed",
        "Marital_Status_Single",
        "Loan_Purpose_Car",
        "Loan_Purpose_Education",
        "Loan_Purpose_Home",
        "Loan_Purpose_Personal",
        "Property_Area_Semiurban",
        "Property_Area_Urban",
        "Gender_Male",
        "Employer_Category_Government",
        "Employer_Category_MNC",
        "Employer_Category_Private",
        "Employer_Category_Unemployed",
        "DTI_Ratio_sq",
        "Credit_Score_sq",
        "Applicant_Income_log"
    ]

    input_data = input_data[feature_order]

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)

    probability = model.predict_proba(scaled_input)

    approved_probability = probability[0][1]
    not_approved_probability = probability[0][0]

    st.divider()

    if prediction[0] == 1:
        st.success("## ✅ Loan Approved")
    else:
        st.error("## ❌ Loan Not Approved")

    st.subheader("Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Not Approved",
            f"{not_approved_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Approved",
            f"{approved_probability * 100:.2f}%"
        )

    st.progress(float(approved_probability))

    st.caption(
        "This prediction is generated by a machine learning model "
        "and should not be considered a financial decision."
    )