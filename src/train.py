from __future__ import annotations

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import MODEL_DIR, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_data
from src.features import preprocess_loan_data
from src.logger import setup_logger

logger = setup_logger()


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def train_and_save():

    logger.info("Starting training pipeline")

    # Load dataset
    df = load_data()

    # Preprocess
    df = preprocess_loan_data(df)

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():

        logger.info(f"Training {name}")

        model.fit(X_train_scaled, y_train)

        metrics = evaluate_model(model, X_test_scaled, y_test)

        results[name] = metrics
        trained_models[name] = model

    # Select best model using F1-score
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_model = trained_models[best_model_name]

    best_metrics = results[best_model_name]

    logger.info(f"Best model: {best_model_name}")
    logger.info(best_metrics)

    MODEL_DIR.mkdir(exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "metrics": best_metrics,
        "all_results": results,
    }

    joblib.dump(bundle, MODEL_PATH)

    logger.info(f"Model saved to {MODEL_PATH}")

    return bundle


if __name__ == "__main__":
    result = train_and_save()
    print("Training finished")
    print(result["model_name"])