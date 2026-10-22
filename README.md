# ML Assignment 2 - Streamlit Apps

Two machine learning web apps for AIC354, deployed on Streamlit Community Cloud.

| App | Main file | Model | Test accuracy |
|---|---|---|---|
| Titanic Survival Predictor | `titanic/app.py` | SVC with label encoding, 4 features | 80% |
| Heart Disease Predictor | `heart-disease/app.py` | SVC with label encoding, 4 features | 78% |

Each app folder contains its own `app.py`, trained `.pkl` model, `requirements.txt` and `runtime.txt`.

## Deploy
1. Push this repository to a public GitHub repo.
2. Go to https://share.streamlit.io, create an app and select this repo.
3. Set the main file path to `titanic/app.py` or `heart-disease/app.py` and deploy.

## Run locally
```
pip install -r heart-disease/requirements.txt
streamlit run heart-disease/app.py
```
