from etl.ingest import load_raw_data
from etl.transform import transform
from etl.load import get_engine, load_raw, load_transformed
from models.train import load_data, prepare_features, split_and_balance, train_xgboost, save_model
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run():
    csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    db_path = Path("data/churn.db")
    model_path = Path("models/xgb_churn_model.pkl")

    # Only run if db or model don't exist
    if not db_path.exists():
        logging.info("Database not found — running ETL pipeline...")
        df_raw = load_raw_data(str(csv_path))
        df_transformed = transform(df_raw)
        engine = get_engine(str(db_path))
        load_raw(df_raw, engine)
        load_transformed(df_transformed, engine)
        logging.info("ETL complete.")

    if not model_path.exists():
        logging.info("Model not found — training XGBoost...")
        df = load_data()
        X, y = prepare_features(df)
        X_train, X_test, y_train, y_test = split_and_balance(X, y)
        model = train_xgboost(X_train, y_train)
        save_model(model, "xgb_churn_model.pkl")
        logging.info("Model training complete.")

if __name__ == "__main__":
    run()