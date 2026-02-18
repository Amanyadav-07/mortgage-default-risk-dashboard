import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Mortgage Default Risk Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# PREMIUM FINTECH THEME
# ---------------------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #eef2f3, #dbe9f4);
    }

    h1 {
        text-align: center;
        font-weight: 800;
        color: #1f2c56;
    }

    h3 {
        color: #2f3e5c;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    }

    section[data-testid="stSidebar"] {
        background-color: #1f2c56;
        color: white;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: white !important;
    }

    .risk-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
        text-align: center;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("mortgage_default_model.pkl")

model = load_model()

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------
st.markdown("<h1>🏦 Mortgage Default Risk Scoring Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Real-Time Credit Risk Assessment Engine</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------
st.sidebar.header("Borrower Information")

age = st.sidebar.slider("Age", 18, 70, 35)
income = st.sidebar.number_input("Annual Income ($)", value=80000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", value=200000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
months_employed = st.sidebar.slider("Months Employed", 0, 480, 60)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 15, 5)
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 7.5)
loan_term = st.sidebar.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360])
dti_ratio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.35)

education = st.sidebar.selectbox("Education", ["High School","Bachelor","Master","PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status",["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Existing Mortgage?", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose",["Home Purchase", "Refinance", "Investment"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer?", ["Yes", "No"])

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
loan_to_income = loan_amount / income if income > 0 else 0
employment_ratio = months_employed / age if age > 0 else 0
credit_utilization_proxy = loan_amount / (num_credit_lines + 1)

input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "LoanToIncome": loan_to_income,
    "EmploymentRatio": employment_ratio,
    "CreditUtilizationProxy": credit_utilization_proxy,
    "Education": education,
    "EmploymentType": employment_type,
    "MaritalStatus": marital_status,
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose,
    "HasCoSigner": has_cosigner
}])

# ---------------------------------------------------
# GAUGE FUNCTION
# ---------------------------------------------------
def plot_gauge(probability):
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.barh(0, 1, color='#e0e0e0')

    if probability < 0.2:
        color = "#2ecc71"
    elif probability < 0.4:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    ax.barh(0, probability, color=color)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])

    ax.text(probability, 0, f"{probability:.2%}",
            ha='center', va='center',
            color='white',
            fontweight='bold',
            fontsize=14)

    ax.set_title("Default Risk Gauge", fontsize=12)

    return fig

# ---------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------
if st.button("🔍 Assess Default Risk"):

    probability = model.predict_proba(input_data)[0][1]

    if probability < 0.2:
        risk_level = "🟢 Low Risk"
        color = "#2ecc71"
    elif probability < 0.4:
        risk_level = "🟡 Moderate Risk"
        color = "#f39c12"
    else:
        risk_level = "🔴 High Risk"
        color = "#e74c3c"

    st.markdown("## 📊 Risk Assessment Result")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Default Probability", f"{probability:.2%}")
        st.markdown(
            f"<div class='risk-card'><h3 style='color:{color};'>{risk_level}</h3></div>",
            unsafe_allow_html=True
        )

    with col2:
        fig = plot_gauge(probability)
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("📈 Key Financial Ratios")

    ratio_df = pd.DataFrame({
        "Metric": ["Loan-to-Income", "DTI Ratio", "Employment Stability"],
        "Value": [loan_to_income, dti_ratio, employment_ratio]
    })

    st.dataframe(ratio_df, use_container_width=True)

    st.markdown("---")

    st.subheader("💡 Decision Recommendation")

    if probability < 0.2:
        st.success("Loan can be approved under standard underwriting guidelines.")
    elif probability < 0.4:
        st.warning("Proceed with caution. Consider additional verification or adjusted pricing.")
    else:
        st.error("High default risk detected. Recommend rejection or strong compensating factors.")

    st.markdown("---")
    st.markdown("<center>© 2026 Mortgage Risk Analytics | Built with Streamlit</center>", unsafe_allow_html=True)