import pandas as pd
from sqlalchemy import create_engine
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_engine(db_path: str = "data/churn.db"):
    """Create a SQLite database engine."""
    logging.info(f"Connecting to SQLite database at {db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    return engine

def load_raw(df: pd.DataFrame, engine) -> None:
    """Load raw data into a staging table."""
    logging.info("Loading raw data into 'raw_customers' table...")
    df.to_sql("raw_customers", engine, if_exists="replace", index=False)
    logging.info(f"Loaded {len(df)} rows into 'raw_customers'")

def load_transformed(df: pd.DataFrame, engine) -> None:
    """Load transformed data into the main table."""
    logging.info("Loading transformed data into 'customers' table...")
    df.to_sql("customers", engine, if_exists="replace", index=False)
    logging.info(f"Loaded {len(df)} rows into 'customers'")

if __name__ == "__main__":
    from ingest import load_raw_data
    from transform import transform

    # Run the full ETL pipeline
    df_raw = load_raw_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_transformed = transform(df_raw)

    # Load both into the database
    engine = get_engine()
    load_raw(df_raw, engine)
    load_transformed(df_transformed, engine)

    logging.info("ETL pipeline complete!")
    logging.info("Database saved to data/churn.db")