"""
Streamlit App: Early Liver Disease Risk Screening
Loads the trained Random Forest model + scaler and predicts risk
from clinical blood-panel inputs.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Liver Disease Risk Screening",
    page_icon="🩺",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    return model, scaler

model, scaler = load_artifacts()

FEATURE_ORDER = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_Globulin_Ratio",
]

st.title("🩺 Liver Disease Risk Screening")
st.caption(
    "Research/demo tool trained on the Indian Liver Patient Dataset (ILPD). "
    "**Not a diagnostic device** — for education and screening-workflow demonstration only."
)

st.markdown("### Enter clinical & blood panel values")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.3, step=0.1, format="%.2f")
    alk_phos = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0, value=200)

with col2:
    alt = st.number_input("Alamine Aminotransferase / ALT (IU/L)", min_value=0, value=30)
    ast = st.number_input("Aspartate Aminotransferase / AST (IU/L)", min_value=0, value=35)
    total_protein = st.number_input("Total Proteins (g/dL)", min_value=0.0, value=6.8, step=0.1, format="%.2f")
    albumin = st.number_input("Albumin (g/dL)", min_value=0.0, value=3.3, step=0.1, format="%.2f")
    ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, value=1.0, step=0.05, format="%.2f")

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    gender_val = 1 if gender == "Male" else 0

    row = pd.DataFrame([[
        age, gender_val, total_bilirubin, direct_bilirubin, alk_phos,
        alt, ast, total_protein, albumin, ag_ratio
    ]], columns=FEATURE_ORDER)

    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0][1]

    st.markdown("### Result")
    if pred == 1:
        st.error(f"⚠️ Elevated risk pattern detected — estimated probability: **{proba:.1%}**")
        st.write("This blood-panel pattern resembles cases flagged as liver disease in the training data. Recommend clinical follow-up (hepatology referral, imaging, further labs).")
    else:
        st.success(f"✅ Lower risk pattern — estimated probability of disease: **{proba:.1%}**")
        st.write("This blood-panel pattern resembles the non-disease cases in the training data. Routine monitoring is still advisable if risk factors are present.")

    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("Feature importance (what drives this model)"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=False)
        st.bar_chart(importances)

st.markdown("---")
st.caption(
    "⚕️ This tool is for educational/demo purposes only and is **not a substitute for professional "
    "medical diagnosis**. The underlying model was trained on the ILPD dataset (general liver-disease "
    "labels), not on imaging-confirmed NAFLD cases. Always consult a qualified clinician."
)
