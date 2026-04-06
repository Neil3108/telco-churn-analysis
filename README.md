# Telco Customer Churn Analysis
An end-to-end data pipeline and machine learning project to predict customer churn 
for a telecom company. Built with Python, SQLite, XGBoost, and Streamlit.

🔗 **[Live Demo](YOUR_STREAMLIT_URL_HERE)** ← we'll fill this in after deployment

---

## Project Overview
This project simulates a real-world CRM analytics pipeline:
- Raw customer data is ingested from CSV
- Cleaned, encoded, and feature-engineered via a modular ETL pipeline
- Stored in a SQLite database (upgradeable to PostgreSQL)
- Analyzed with exploratory visualizations
- Used to train a churn prediction model (XGBoost, ROC-AUC: 0.836)
- Served via an interactive Streamlit web app

## Key Findings
- **26.5%** of customers churned
- Contract type is the strongest churn predictor — month-to-month customers churn at nearly 1:1
- New customers (< 10 months tenure) are highest risk
- Fiber optic internet customers churn significantly more than DSL customers
- Bundled services (online security, tech support) correlate with retention

## Tech Stack
| Layer | Tools |
|---|---|
| Data Ingestion | Python, pandas |
| Transformation | pandas, numpy |
| Storage | SQLite, SQLAlchemy |
| Analysis | matplotlib, seaborn, Jupyter |
| Modeling | scikit-learn, XGBoost, imbalanced-learn |
| App | Streamlit |
| Version Control | Git, GitHub |

## Project Structure
churn-etl-project/
├── data/                          # Raw CSV dataset
├── etl/
│   ├── ingest.py                  # Extract — load raw CSV
│   ├── transform.py               # Transform — clean and encode
│   └── load.py                    # Load — write to SQLite
├── analysis/
│   └── eda.ipynb                  # Exploratory data analysis notebook
├── models/
│   └── train.py                   # Model training and evaluation
├── app.py                         # Streamlit web app
├── startup.py                     # Auto-runs ETL + training if needed
├── requirements.txt
└── README.md

## Setup Instructions
1. Clone the repo
```bash
   git clone https://github.com/Neil3108/telco-churn-analysis.git
   cd telco-churn-analysis
```

2. Create and activate virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Download the dataset from Kaggle
   - [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Place the CSV in the `data/` folder

5. Run the app
```bash
   streamlit run app.py
```
   The ETL pipeline and model training will run automatically on first launch.

## Model Performance
| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.828 | 0.567 | 0.671 | 0.614 |
| XGBoost | 0.836 | 0.548 | 0.652 | 0.596 |

## Dataset
IBM Telco Customer Churn dataset — 7,043 customers, 21 features.
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).