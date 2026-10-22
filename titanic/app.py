"""
Titanic Passenger Survival Prediction - Streamlit app.
Loads the trained Support Vector Classifier (svc_trained_model.pkl) from the
Titanic survival notebook and rebuilds the same LabelEncoders used during
training, so the app's predictions match the notebook.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["PClass"] = LabelEncoder().fit(["First", "Second", "Third"])
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["Sibling"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Embarked"] = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])
    return encoders


model = load_model()
encoders = build_label_encoders()

st.title("Titanic Passenger Survival Predictor")
st.write(
    "Predict Titanic passenger survival from four attributes using a Support Vector "
    "Classifier. Select the passenger's details below and click Predict."
)
st.caption("Test accuracy: 80%.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings / Spouses Aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked],
    })

    encoded = pd.DataFrame({
        "PClass": encoders["PClass"].transform(user_input["PClass"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "Sibling": encoders["Sibling"].transform(user_input["Sibling"]),
        "Embarked": encoders["Embarked"].transform(user_input["Embarked"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.success("Survived")
    else:
        st.error("Did not survive")

    with st.expander("View model input"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Educational project using the Titanic survival dataset.")
