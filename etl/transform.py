import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is read as string due to some blank values — convert to float."""
    logging.info("Fixing TotalCharges column...")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_count = df["TotalCharges"].isnull().sum()
    logging.info(f"Found {blank_count} blank TotalCharges — filling with 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def encode_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn column from Yes/No to 1/0."""
    logging.info("Encoding Churn column to 1/0...")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No columns to 1/0."""
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines"
    ]
    logging.info(f"Encoding binary columns: {binary_cols}")
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-value categorical columns."""
    cat_cols = [
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod"
    ]
    logging.info(f"One-hot encoding categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender as 1/0."""
    logging.info("Encoding gender column...")
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features useful for churn prediction."""
    logging.info("Adding engineered features...")

    # Average monthly spend relative to tenure
    df["charges_per_tenure"] = np.where(
        df["tenure"] == 0,
        df["MonthlyCharges"],
        df["TotalCharges"] / df["tenure"]
    )

    # Flag long-term customers
    df["is_long_term"] = (df["tenure"] >= 24).astype(int)

    # High spender flag
    df["is_high_spender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not useful for modeling."""
    logging.info("Dropping customerID column...")
    df = df.drop(columns=["customerID"])
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full transformation pipeline."""
    logging.info("Starting transformation pipeline...")
    df = fix_total_charges(df)
    df = encode_churn(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = encode_gender(df)
    df = add_features(df)
    df = drop_unnecessary_columns(df)
    logging.info(f"Transformation complete. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    from ingest import load_raw_data
    df = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_transformed = transform(df)
    print("\n--- Transformed Data Sample ---")
    print(df_transformed.head())
    print("\n--- Columns ---")
    print(list(df_transformed.columns))