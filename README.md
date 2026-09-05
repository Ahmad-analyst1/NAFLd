# Liver Disease Risk Screening — Streamlit App

A deployable Streamlit app that serves the trained Random Forest model
for early liver disease risk screening from clinical/blood-panel inputs.

## Files
- `app.py` — Streamlit application
- `model.joblib` — trained Random Forest classifier
- `scaler.joblib` — fitted StandardScaler (must be applied to inputs before prediction)
- `requirements.txt` — Python dependencies

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (keep `app.py`, `model.joblib`, `scaler.joblib`, `requirements.txt` together).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", pick the repo/branch, set main file to `app.py`.
4. Deploy — you'll get a public URL in ~1-2 minutes.

## Deploy on other platforms
- **Docker**: wrap with a simple Dockerfile (`FROM python:3.11-slim`, copy files, `pip install -r requirements.txt`, `CMD ["streamlit","run","app.py","--server.port=8080","--server.address=0.0.0.0"]`) and deploy to Cloud Run, Render, Railway, etc.
- **Hugging Face Spaces**: create a Space with SDK = Streamlit, upload these files.

## Important notes
- The model was trained on the **Indian Liver Patient Dataset (ILPD)** — general liver-disease labels based on bilirubin, liver enzymes, proteins, and albumin. It is **not** trained on imaging-confirmed NAFLD cases, so treat this as a screening/triage demo, not a diagnostic tool.
- Test-set performance: ~70% accuracy, ~0.76 ROC-AUC (see `final_metrics.json` from the training run).
- Input order matters: the app already handles feature ordering and scaling internally — don't change the `FEATURE_ORDER` list unless you retrain the model.
- Add authentication/rate-limiting before exposing this publicly if it will ever touch real patient data (HIPAA/PHI considerations).
