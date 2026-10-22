"""
Heart Disease Prediction - Streamlit app.
Loads the trained Support Vector Classifier (heart_disease_svc_model.pkl) from
the Heart_Disease_Prediction notebook and rebuilds the same LabelEncoders used
during training, so the app's predictions match the notebook.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["ChestPain"] = LabelEncoder().fit(
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    encoders["MajorVessels"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Thalassemia"] = LabelEncoder().fit(["Normal", "Fixed Defect", "Reversible Defect"])
    encoders["STDepression"] = LabelEncoder().fit(["None", "Mild", "Moderate", "Severe"])
    return encoders


model = load_model()
encoders = build_label_encoders()

st.title("Heart Disease Predictor")
st.write(
    "Predict the presence of heart disease from four clinical attributes using a "
    "Support Vector Classifier trained on the UCI Heart Disease (Cleveland) dataset. "
    "Select the patient's details below and click Predict."
)
st.caption("Test accuracy: 78%.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        index=3,
    )
    thalassemia = st.selectbox(
        "Thalassemia Result",
        ["Normal", "Fixed Defect", "Reversible Defect"],
    )
with col2:
    majorvessels = st.selectbox(
        "Major Vessels (coloured by fluoroscopy)",
        ["Zero", "One", "Two", "Three"],
    )
    stdepression = st.selectbox(
        "ST Depression (exercise vs rest)",
        ["None", "Mild", "Moderate", "Severe"],
    )

if st.button("Predict", type="primary", use_container_width=True):
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
        st.error("Heart disease predicted")
    else:
        st.success("No heart disease predicted")

    with st.expander("View model input"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Educational project using the UCI Heart Disease (Cleveland) dataset. Not a diagnostic tool.")
