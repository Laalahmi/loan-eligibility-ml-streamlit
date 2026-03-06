# Loan Eligibility Prediction – ML Deployment with Streamlit

## Overview

This project implements a machine learning pipeline to predict whether a loan application is likely to be approved based on applicant demographic, financial, and property information.

The original solution was first developed in a Jupyter Notebook and later modularized into reusable Python modules. The final model was deployed as an interactive web application using Streamlit.

This project demonstrates important machine learning engineering concepts including:

- code modularization
- preprocessing and feature engineering
- training and comparing multiple classification models
- model evaluation
- logging and error handling
- model deployment with Streamlit

This project was developed as part of:

**CST2216 – Modularizing and Deploying ML Code**  
**Algonquin College**

---

## Live Application

You can access the deployed application here:

https://loan-eligibility-ml-app-nmfpuyjtkmmbav4ifgkv4h.streamlit.app/

---

## Problem Statement

The objective of this project is to predict whether a loan application will be approved using applicant information such as income, education, credit history, employment status, and property area.

This is a **binary classification** problem where:

- `1` = Loan Approved
- `0` = Loan Not Approved

---

## Dataset Features

The dataset includes the following input features:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

Target variable:

- Loan_Approved

---

## Preprocessing and Feature Engineering

The preprocessing pipeline includes:

- dropping `Loan_ID`
- handling missing values
- filling categorical/discrete features with mode
- filling `LoanAmount` with median
- converting `Loan_Approved` from `Y/N` to `1/0`
- one-hot encoding categorical variables

Categorical variables encoded with dummies:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- Property_Area

Numeric variables kept as numeric:

- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History

---

## Models Compared

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

The final deployed model was selected automatically based on the **highest F1-score**.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics are more informative than accuracy alone for binary classification tasks.

---

## Project Structure

```text
loan-eligibility-ml-streamlit/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── algonquin_logo.png
│
├── data/
│   └── loan_eligibility.csv
│
├── models/
│   └── loan_eligibility_model.joblib
│
├── logs/
│
├── notebooks/
│   └── Loan_Eligibility_Model_Solution.ipynb
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── logger.py
    ├── data_loader.py
    ├── features.py
    ├── train.py
    └── evaluate.py
