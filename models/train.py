import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = Path(__file__).parent.parent / "data" / "churn.db"

def load_data():
    """Load transformed data from the database."""
    logging.info("Loading data from database...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM customers", engine)
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def prepare_features(df: pd.DataFrame):
    """Split data into features and target, handle any remaining nulls."""
    logging.info("Preparing features...")

    # Fill any remaining NaN values
    df = df.fillna(0)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    logging.info(f"Features: {X.shape[1]} columns")
    logging.info(f"Churn rate: {y.mean():.2%}")
    return X, y

def split_and_balance(X, y):
    """Train/test split then apply SMOTE to training set only."""
    logging.info("Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logging.info("Applying SMOTE to balance training set...")

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    logging.info(f"Balanced train size: {len(X_train_bal)}")
    logging.info(f"Balanced churn rate: {y_train_bal.mean():.2%}")

    return X_train_bal, X_test, y_train_bal, y_test

def evaluate_model(model, X_test, y_test, model_name: str):
    """Print evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Results ---")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"models/{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.show()
    logging.info(f"Confusion matrix saved for {model_name}")

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression baseline model."""
    logging.info("Training Logistic Regression baseline...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost model."""
    logging.info("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filename: str):
    """Save model to disk using pickle."""
    filepath = f"models/{filename}"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {filepath}")

def plot_feature_importance(model, feature_names):
    """Plot top 15 most important features from XGBoost."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title("Top 15 Feature Importances — XGBoost")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.show()
    logging.info("Feature importance plot saved")

if __name__ == "__main__":
    # Load and prepare
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(X, y)

    # Train both models
    lr_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate both
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Save feature importance plot
    plot_feature_importance(xgb_model, X.columns)

    # Save XGBoost model for Streamlit app
    save_model(xgb_model, "xgb_churn_model.pkl")
    save_model(lr_model, "lr_churn_model.pkl")

    logging.info("Training complete!")