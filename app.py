from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import MODEL_PATH
from src.features import preprocess_loan_data
from src.logger import setup_logger

logger = setup_logger()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Modern CSS
# ---------------------------
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2.4rem !important;
        padding-bottom: 2rem !important;
      }

      h1, h2, h3 { margin-top: 0.2rem !important; }

      .card {
        background: rgba(0,0,0,0.035);
        border: 1px solid rgba(0,0,0,0.07);
        border-radius: 18px;
        padding: 18px;
      }

      .subtle {
        color: rgba(0,0,0,0.65);
        font-size: 0.95rem;
      }

      .title {
        font-size: 2.05rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.15;
      }

      .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.10);
        font-size: 0.85rem;
        margin-right: 0.35rem;
        margin-top: 0.2rem;
      }

      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load model bundle
# ---------------------------
@st.cache_resource
def load_bundle():
    logger.info(f"Loading model bundle from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    metrics = bundle["metrics"]
    model_name = bundle["model_name"]
    all_results = bundle.get("all_results", {})
    logger.info("Loan Eligibility model bundle loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model bundle: {e}")
    st.error(
        "❌ Could not load the trained model.\n\n"
        f"Make sure the file exists: `{MODEL_PATH}`\n\n"
        "Run: `python -m src.train`"
    )
    st.stop()

# ---------------------------
# Header with logo + credits
# ---------------------------
logo_path = Path("assets/algonquin_logo.png")

hcol1, hcol2 = st.columns([0.18, 0.82], vertical_alignment="center")

with hcol1:
    if logo_path.exists():
        try:
            img = Image.open(logo_path)
            st.image(img, use_container_width=True)
        except UnidentifiedImageError:
            st.warning("Logo file exists but is not a valid image.")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")
    else:
        st.caption("📌 Add `assets/algonquin_logo.png` to show the logo")

with hcol2:
    st.markdown('<p class="title">🏦 Loan Eligibility Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <span class="badge">CST2216</span>
        <span class="badge">Modularizing & Deploying ML Code</span>
        <span class="badge">Classification App</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtle'>Predict whether a loan application is likely to be approved based on applicant information.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtle'><b>Created by:</b> Mohammed Laalahmi &nbsp; • &nbsp; "
        "<b>Instructor:</b> Dr. Umer Altaf &nbsp; • &nbsp; <b>Algonquin College</b></p>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.title("📝 Applicant Information")
st.sidebar.caption("Fill in the information below, then click **Predict Loan Decision**.")

with st.sidebar.expander("👤 Personal Information", expanded=True):
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

with st.sidebar.expander("💰 Financial Information", expanded=True):
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150, step=1)
    loan_term = st.sidebar.selectbox("Loan Amount Term", [360, 180, 120, 60])
    credit_history = st.sidebar.selectbox("Credit History", [1, 0])

with st.sidebar.expander("🏘️ Property Information", expanded=True):
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.divider()
predict_btn = st.sidebar.button("🚀 Predict Loan Decision", use_container_width=True)

st.sidebar.markdown(
    "<p class='subtle'>Credits:<br><b>Mohammed Laalahmi</b><br>"
    "<b>Dr. Umer Altaf</b> (Instructor)<br>"
    "<b>Algonquin College</b></p>",
    unsafe_allow_html=True,
)

# ---------------------------
# Main tabs
# ---------------------------
tab_pred, tab_info, tab_about = st.tabs(["🔮 Prediction", "📊 Model Info", "ℹ️ About"])

with tab_pred:
    left, right = st.columns([1.2, 1], vertical_alignment="top")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if predict_btn:
            try:
                logger.info("Loan prediction requested.")

                input_df = pd.DataFrame(
                    {
                        "Gender": [gender],
                        "Married": [married],
                        "Dependents": [dependents],
                        "Education": [education],
                        "Self_Employed": [self_employed],
                        "ApplicantIncome": [applicant_income],
                        "CoapplicantIncome": [coapplicant_income],
                        "LoanAmount": [loan_amount],
                        "Loan_Amount_Term": [loan_term],
                        "Credit_History": [credit_history],
                        "Property_Area": [property_area],
                    }
                )

                processed = preprocess_loan_data(input_df)

                for col in feature_names:
                    if col not in processed.columns:
                        processed[col] = 0

                processed = processed[feature_names]
                scaled = scaler.transform(processed)

                prediction = int(model.predict(scaled)[0])
                probability = float(model.predict_proba(scaled)[0][1])

                if prediction == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Not Approved")

                st.metric("Approval Probability", f"{probability:.2%}")

                if probability >= 0.75:
                    st.info("Strong approval likelihood based on the model.")
                elif probability >= 0.50:
                    st.warning("Moderate approval likelihood.")
                else:
                    st.warning("Low approval likelihood based on the model.")

                logger.info(f"Prediction={prediction}, probability={probability:.4f}")

            except Exception as e:
                logger.exception(f"Prediction failed: {e}")
                st.error("❌ Prediction failed. Check logs for details.")
        else:
            st.info("Use the sidebar to enter applicant details, then click **Predict Loan Decision**.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.subheader("Quick Notes")
        st.write(
            "- `Credit_History` is one of the most influential variables in this dataset.\n"
            "- The app applies the same preprocessing logic used during training.\n"
            "- The best model was selected automatically using **F1-score**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Summary")

        summary_df = pd.DataFrame(
            {
                "Feature": [
                    "Gender",
                    "Married",
                    "Dependents",
                    "Education",
                    "Self Employed",
                    "Applicant Income",
                    "Coapplicant Income",
                    "Loan Amount",
                    "Loan Amount Term",
                    "Credit History",
                    "Property Area",
                ],
                "Value": [
                    gender,
                    married,
                    dependents,
                    education,
                    self_employed,
                    applicant_income,
                    coapplicant_income,
                    loan_amount,
                    loan_term,
                    credit_history,
                    property_area,
                ],
            }
        )

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Selected Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{metrics['precision']:.2f}")
    c3.metric("Recall", f"{metrics['recall']:.2f}")
    c4.metric("F1-score", f"{metrics['f1']:.2f}")

    st.divider()
    st.write(f"**Selected Model:** {model_name}")
    st.write("The final deployment model was selected automatically based on the highest **F1-score**.")

    with st.expander("Show confusion matrix"):
        st.write(metrics["confusion_matrix"])

    with st.expander("Show all model comparison results"):
        comparison_rows = []
        for name, vals in all_results.items():
            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": vals["accuracy"],
                    "Precision": vals["precision"],
                    "Recall": vals["recall"],
                    "F1-score": vals["f1"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About This Project")
    st.write(
        "This application is part of **CST2216** at **Algonquin College**. "
        "The project modularizes a machine learning notebook into reusable Python modules "
        "for data loading, preprocessing, training, evaluation, and deployment using Streamlit."
    )
    st.write(
        "**Credits:** Mohammed Laalahmi • Dr. Umer Altaf • Algonquin College"
    )
    st.caption("If the model file is missing, run training first: `python -m src.train`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<hr/>"
    "<p class='subtle'>© 2026 • Mohammed Laalahmi • CST2216 • Dr. Umer Altaf • Algonquin College</p>",
    unsafe_allow_html=True,
)