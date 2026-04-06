import pandas as pd
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from the given filepath."""
    logging.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logging.info(f"Columns: {list(df.columns)}")
    
    return df

def preview_data(df: pd.DataFrame) -> None:
    """Print a quick preview of the dataframe."""
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Basic Stats ---")
    print(df.describe())

if __name__ == "__main__":
    filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_raw_data(filepath)
    preview_data(df)