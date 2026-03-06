from __future__ import annotations
import pandas as pd
from src.logger import setup_logger

logger = setup_logger()


def preprocess_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the loan dataset.
    """

    logger.info("Starting preprocessing")

    df = df.copy()

    # Drop ID column
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # Fill categorical missing values with mode
    cat_mode_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
    ]

    for col in cat_mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Numeric imputation
    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    if "Loan_Amount_Term" in df.columns:
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # Encode target
    if "Loan_Approved" in df.columns:
        df["Loan_Approved"] = df["Loan_Approved"].map({"Y": 1, "N": 0})

    # One-hot encode categorical variables
    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logger.info("Preprocessing finished")

    return df