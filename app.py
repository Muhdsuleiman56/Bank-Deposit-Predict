# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="centered")

st.title("Bank Term Deposit Subscription Predictor")
st.write(
    "This app predicts whether a customer will subscribe to a term deposit "
    "(**y = 1** yes, **y = 0** no) using a trained scikit-learn pipeline."
)

# -----------------------------
# Load trained pipeline model
# -----------------------------
MODEL_PATH = "bank_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Put bank_model.pkl in the same folder as app.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Optional: Load CSV to populate dropdown options (recommended)
# If not found, app still works using fallback options.
# -----------------------------
DATA_PATH = "bank-full.csv"
fallback_options = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["yes", "no"],
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "poutcome": ["failure", "nonexistent", "success", "unknown"],
}

if os.path.exists(DATA_PATH):
    try:
        df_ref = pd.read_csv(DATA_PATH, sep=";")
        df_ref.columns = df_ref.columns.str.strip()

        # Build options from real data (safer than guessing)
        options = {}
        for col in fallback_options.keys():
            if col in df_ref.columns:
                options[col] = sorted(df_ref[col].astype(str).str.strip().dropna().unique().tolist())
            else:
                options[col] = fallback_options[col]
    except Exception:
        options = fallback_options
else:
    options = fallback_options

# -----------------------------
# Input form
# -----------------------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("age", min_value=18, max_value=100, value=35)
    job = st.selectbox("job", options["job"])
    marital = st.selectbox("marital", options["marital"])
    education = st.selectbox("education", options["education"])
    default = st.selectbox("default", options["default"])
    balance = st.number_input("balance", value=1000)

with col2:
    housing = st.selectbox("housing", options["housing"])
    loan = st.selectbox("loan", options["loan"])
    contact = st.selectbox("contact", options["contact"])
    day = st.number_input("day", min_value=1, max_value=31, value=15)
    month = st.selectbox("month", options["month"])
    duration = st.number_input("duration (seconds)", min_value=0, value=180)

st.markdown("### Campaign History")
campaign = st.number_input("campaign (contacts during this campaign)", min_value=1, value=1)
pdays = st.number_input("pdays (days since last contact; -1 if never)", value=-1)
previous = st.number_input("previous (contacts before this campaign)", min_value=0, value=0)
poutcome = st.selectbox("poutcome", options["poutcome"])

# -----------------------------
# Feature Engineering (must match training notebook)
# -----------------------------
contacted_before = int(previous > 0)

# age_group: bins [0,30,50,100] -> young/middle/senior
if age <= 30:
    age_group = "young"
elif age <= 50:
    age_group = "middle"
else:
    age_group = "senior"

# balance_level: created using qcut in notebook.
# In app, we approximate with simple thresholds (fallback).
# If you used qcut, your model will still handle these categories via OneHotEncoder,
# but the exact bin edges may differ. This approximation is usually fine for demo.
if balance < 0:
    balance_level = "low"
elif balance < 1000:
    balance_level = "mid"
elif balance < 5000:
    balance_level = "high"
else:
    balance_level = "very_high"

# -----------------------------
# Build input row (must match training feature names)
# -----------------------------
input_row = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,

    # engineered features
    "contacted_before": contacted_before,
    "age_group": age_group,
    "balance_level": balance_level
}])

# -----------------------------
# Predict
# -----------------------------
st.subheader("Prediction")

if st.button("Predict"):
    try:
        pred = int(model.predict(input_row)[0])
        proba = None

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_row)[0][1])

        if pred == 1:
            st.success("✅ Prediction: Customer is **likely to subscribe** (y = 1)")
        else:
            st.warning("❌ Prediction: Customer is **unlikely to subscribe** (y = 0)")

        if proba is not None:
            st.write(f"Probability of subscription (y=1): **{proba:.2f}**")

        st.caption("Note: Results depend on the trained model and preprocessing pipeline saved in bank_model.pkl.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Common cause: the app's input columns do not match the model's training features.\n"
            "Fix: ensure your notebook trained the model using the same engineered features and column names."
        )
