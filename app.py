import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from pathlib import Path
from startup import run as startup

startup()

# Page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

DB_PATH = DB_PATH = Path(__file__).parent / "data" / "churn.db"

# Load model and data
@st.cache_resource
def load_model():
    with open("models/xgb_churn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM customers", engine)
    df_raw = pd.read_sql("SELECT * FROM raw_customers", engine)
    return df, df_raw

model = load_model()
df, df_raw = load_data()

# Sidebar navigation
st.sidebar.title("📡 Churn Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA Dashboard", "🔮 Predict Churn"]
)

# PAGE 1 - Overview
if page == "🏠 Overview":
    st.title("Telco Customer Churn Analysis")
    st.markdown("""
    This app presents an end-to-end data pipeline for predicting customer churn
    at a telecom company. Built with Python, SQLite, XGBoost, and Streamlit.

    **Navigate using the sidebar:**
    - 📊 **EDA Dashboard** - explore the data visually
    - 🔮 **Predict Churn** - enter a customer profile and get a churn prediction
    """)

    st.markdown("---")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df_raw):,}")
    churn_rate = df_raw['Churn'].mean()
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Avg Monthly Charges", f"${df_raw['MonthlyCharges'].mean():.2f}")
    col4.metric("Avg Tenure", f"{df_raw['tenure'].mean():.0f} months")

    st.markdown("---")
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "XGBoost")
    col2.metric("ROC-AUC", "0.836")
    col3.metric("Recall", "65.2%")
    col4.metric("Precision", "54.8%")

# PAGE 2 - EDA Dashboard
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    # Churn overview
    st.subheader("Churn Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        churn_counts = df_raw["Churn"].value_counts()
        ax.pie(churn_counts, labels=["No Churn", "Churn"],
               autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"])
        ax.set_title("Churn Rate")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df_raw, x="Contract", hue="Churn", ax=ax)
        ax.set_title("Churn by Contract Type")
        ax.set_xlabel("Contract Type")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Customer profile charts
    st.subheader("Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df_raw, x="tenure", hue="Churn", bins=30, ax=ax)
        ax.set_title("Tenure Distribution")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_raw, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        st.pyplot(fig)
        plt.close()

    with col3:
        fig, ax = plt.subplots()
        sns.countplot(data=df_raw, x="InternetService", hue="Churn", ax=ax)
        ax.set_title("Churn by Internet Service")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Feature importance
    st.subheader("Top 15 Feature Importances - XGBoost")
    importance = pd.Series(
        model.feature_importances_,
        index=df.drop(columns=["Churn"]).columns
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    plt.close()


# PAGE 3 - Predict Churn
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter a customer profile below to get a churn risk prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("Account Info")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * tenure), step=1.0)

    with col3:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    st.markdown("---")

    if st.button("🔮 Predict", use_container_width=True):
        # Build input matching transformed feature columns
        def yn(val): return 1 if val == "Yes" else 0
        def match(val, target): return 1 if val == target else 0

        input_data = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": yn(senior),
            "Partner": yn(partner),
            "Dependents": yn(dependents),
            "tenure": tenure,
            "PhoneService": yn(phone),
            "MultipleLines": yn(multiple_lines),
            "PaperlessBilling": yn(paperless),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "InternetService_Fiber optic": match(internet, "Fiber optic"),
            "InternetService_No": match(internet, "No"),
            "OnlineSecurity_No internet service": 1 if internet == "No" else 0,
            "OnlineSecurity_Yes": yn(online_security) if internet != "No" else 0,
            "OnlineBackup_No internet service": 1 if internet == "No" else 0,
            "OnlineBackup_Yes": yn(online_backup) if internet != "No" else 0,
            "DeviceProtection_No internet service": 1 if internet == "No" else 0,
            "DeviceProtection_Yes": yn(device_protection) if internet != "No" else 0,
            "TechSupport_No internet service": 1 if internet == "No" else 0,
            "TechSupport_Yes": yn(tech_support) if internet != "No" else 0,
            "StreamingTV_No internet service": 1 if internet == "No" else 0,
            "StreamingTV_Yes": yn(streaming_tv) if internet != "No" else 0,
            "StreamingMovies_No internet service": 1 if internet == "No" else 0,
            "StreamingMovies_Yes": yn(streaming_movies) if internet != "No" else 0,
            "Contract_One year": match(contract, "One year"),
            "Contract_Two year": match(contract, "Two year"),
            "PaymentMethod_Credit card (automatic)": match(payment, "Credit card (automatic)"),
            "PaymentMethod_Electronic check": match(payment, "Electronic check"),
            "PaymentMethod_Mailed check": match(payment, "Mailed check"),
            "charges_per_tenure": monthly_charges if tenure == 0 else total_charges / tenure,
            "is_long_term": 1 if tenure >= 24 else 0,
            "is_high_spender": 1 if monthly_charges > df_raw["MonthlyCharges"].median() else 0,
        }

        input_df = pd.DataFrame([input_data])
        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("### Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(f"⚠️ High Churn Risk - {probability:.1%} probability")
            else:
                st.success(f"✅ Low Churn Risk - {probability:.1%} probability")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(["Stay", "Churn"],
                    [1 - probability, probability],
                    color=["#4C72B0", "#DD8452"])
            ax.set_xlim(0, 1)
            ax.set_title("Churn Probability")
            st.pyplot(fig)
            plt.close()