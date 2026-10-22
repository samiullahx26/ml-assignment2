# ML Assignment 2 — Streamlit Apps

Two end-to-end machine learning demos (AIC354, COMSATS), deployed on Streamlit Community Cloud.

| App | Main file path | Model | Test accuracy |
|---|---|---|---|
| 🚢 Titanic Survival Predictor | `titanic/app.py` | SVC (LabelEncoder + 4 categorical features) | 0.80 |
| ❤️ Heart Disease Predictor | `heart-disease/app.py` | SVC (LabelEncoder + 4 data-driven features) | 0.78 |

Each app is self-contained (its own `app.py`, trained `.pkl`, `requirements.txt`, `runtime.txt`).

## Deploy
1. Push this repo to **public** GitHub.
2. <https://share.streamlit.io> → **Create app** → pick this repo.
3. Set **Main file path** to `titanic/app.py` (or `heart-disease/app.py`) → **Deploy**.

## Run locally
```bash
pip install -r heart-disease/requirements.txt
streamlit run heart-disease/app.py
```

> ⚠️ The Heart Disease app is a teaching demo built from the UCI Cleveland dataset — **not medical advice.**
