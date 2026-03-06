from __future__ import annotations

import pandas as pd

from src.config import DATA_PATH
from src.logger import setup_logger

logger = setup_logger()


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """
    Load the loan eligibility dataset and perform basic validation.
    """
    try:
        logger.info(f"Loading dataset from: {path}")

        if not path.exists():
            logger.error(f"Dataset not found at: {path}")
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        if df.empty:
            logger.error("Dataset loaded but is empty.")
            raise ValueError("Dataset is empty.")

        required_cols = {"Loan_Approved"}
        missing = required_cols - set(df.columns)

        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception(f"Failed to load dataset: {e}")
        raise