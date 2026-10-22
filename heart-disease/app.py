"""
Heart Disease Prediction - Student Demo App
--------------------------------------------
A small Streamlit web app that lets anyone interact with the trained
Support Vector Classifier from the "Heart_Disease_Prediction" notebook,
without installing Python or running Jupyter.

It loads the same heart_disease_svc_model.pkl produced by that notebook,
and recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["ChestPain"] = LabelEncoder().fit(
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    encoders["MajorVessels"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Thalassemia"] = LabelEncoder().fit(["Normal", "Fixed Defect", "Reversible Defect"])
    encoders["STDepression"] = LabelEncoder().fit(["None", "Mild", "Moderate", "Severe"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction** notebook
    (UCI Heart Disease dataset). Pick the four patient details below and the trained
    Support Vector Classifier (SVC) will predict whether heart disease is present.

    *Model accuracy on held-out test data: **78%** (UCI Cleveland dataset, 297 records).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        index=3,
        help="UCI attribute: cp",
    )
    thalassemia = st.selectbox(
        "Thalassemia Result",
        ["Normal", "Fixed Defect", "Reversible Defect"],
        help="UCI attribute: thal",
    )
with col2:
    majorvessels = st.selectbox(
        "Major Vessels Coloured by Fluoroscopy",
        ["Zero", "One", "Two", "Three"],
        help="UCI attribute: ca (number of major vessels, 0-3)",
    )
    stdepression = st.selectbox(
        "ST Depression (exercise vs rest)",
        ["None", "Mild", "Moderate", "Severe"],
        help="UCI attribute: oldpeak, binned. None=0, Mild=0-1, Moderate=1-2, Severe=>2",
    )

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    # Build the feature vector in the SAME column order used for training
    user_input = pd.DataFrame({
        "ChestPain": [chestpain],
        "MajorVessels": [majorvessels],
        "Thalassemia": [thalassemia],
        "STDepression": [stdepression],
    })

    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "MajorVessels": encoders["MajorVessels"].transform(user_input["MajorVessels"]),
        "Thalassemia": encoders["Thalassemia"].transform(user_input["Thalassemia"]),
        "STDepression": encoders["STDepression"].transform(user_input["STDepression"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ❤️‍🩹 HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "⚠️ Teaching demo only — NOT medical advice. Built from the UCI Heart Disease "
    "(Cleveland) dataset using 4 data-driven attributes (cp, ca, thal, oldpeak). "
    "Predictions reflect patterns in a 297-record research dataset and must never be "
    "used for real diagnosis."
)
