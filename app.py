# app.py
"""
Streamlit app — clinician enters patient data manually, model predicts probability
of deterioration within 90 days and shows reasons (SHAP or feature importances).
Save as app.py and run: `streamlit run app.py`
"""
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="90-day Deterioration Risk (CSV Input)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers
# -------------------------
def safe_load_model(path: Path):
    payload = joblib.load(path)
    if isinstance(payload, dict):
        pipe = payload.get("pipeline") or payload.get("model") or payload.get("estimator")
        features = payload.get("features") or payload.get("columns") or payload.get("feature_names")
        return pipe, list(features) if features is not None else None, payload
    pipe = payload
    features = getattr(pipe, "feature_names_in_", None)
    return pipe, list(features) if features is not None else None, payload


def get_last_estimator(pipe):
    from sklearn.pipeline import Pipeline
    if isinstance(pipe, Pipeline):
        return list(pipe.named_steps.values())[-1]
    return pipe


# -------------------------
# Sidebar: Model selection
# -------------------------
st.sidebar.header("⚙️ Model Setup")
model_file = st.sidebar.text_input("Model file path (joblib)", value="models/rf_last90d.joblib")
uploaded_model = st.sidebar.file_uploader("Upload model (.joblib)", type=["joblib", "pkl"])

pipe, feature_list, raw_payload = None, None, None
try:
    if uploaded_model is not None:
        tmp = Path("models/_uploaded_model.joblib")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(uploaded_model.getbuffer())
        pipe, feature_list, raw_payload = safe_load_model(tmp)
        st.sidebar.success("✅ Uploaded model loaded")
    else:
        p = Path(model_file)
        if p.exists():
            pipe, feature_list, raw_payload = safe_load_model(p)
            st.sidebar.success(f"📂 Loaded model: {p.name}")
        else:
            st.sidebar.warning(f"⚠️ Model file not found: {model_file}")
except Exception as e:
    st.sidebar.error("❌ Failed to load model")
    st.sidebar.text(str(e))

# -------------------------
# Main UI
# -------------------------
st.title("🩺 90-day Deterioration Risk — CSV Input")
st.markdown("Upload a CSV file containing patient data. The model will predict the probability of deterioration within 90 days and display results.")

uploaded_csv = st.file_uploader("Upload CSV with patient records", type=["csv"])

if uploaded_csv is not None and pipe is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.success(f"📄 Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

        # Ensure correct columns
        if feature_list:
            missing = [f for f in feature_list if f not in df.columns]
            if missing:
                st.error(f"❌ Missing required features in CSV: {missing}")
                st.stop()
            X = df[feature_list]
        else:
            X = df

        # Predictions
        predictive = pipe["pipeline"] if isinstance(pipe, dict) and "pipeline" in pipe else pipe
        proba_arr = predictive.predict_proba(X)
        probs = proba_arr[:, 1]
        df["predicted_risk"] = probs

        # -------------------------
        # Summary Metrics (Horizontal Layout)
        # -------------------------
        st.subheader("📊 Prediction Summary")
        col1, col2, col3 = st.columns(3)

        # Risk categories
        high_risk = (df["predicted_risk"] >= 0.75).sum()
        moderate_risk = ((df["predicted_risk"] >= 0.40) & (df["predicted_risk"] < 0.75)).sum()
        low_risk = (df["predicted_risk"] < 0.40).sum()

        with col1:
            st.metric("High Risk Patients", high_risk)
            st.metric("Moderate Risk Patients", moderate_risk)
            st.metric("Low Risk Patients", low_risk)

        with col2:
            avg_risk = df["predicted_risk"].mean() * 100
            st.progress(int(avg_risk))
            st.write(f"**Average Risk:** {avg_risk:.1f}%")

        with col3:
            fig, ax = plt.subplots()
            sizes = [high_risk, moderate_risk, low_risk]
            labels = ["High", "Moderate", "Low"]
            colors = ["#ff4d4d", "#ffcc00", "#4CAF50"]
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors)
            ax.set_title("Risk Distribution")
            st.pyplot(fig)

        # -------------------------
        # Detailed Results
        # -------------------------
        st.subheader("📋 Detailed Predictions")
        st.dataframe(df[["predicted_risk"] + [c for c in df.columns if c != "predicted_risk"]].head(50), use_container_width=True)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv_out, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.text(traceback.format_exc())

elif uploaded_csv is None:
    st.info("⬆️ Please upload a CSV file to proceed.")
elif pipe is None:
    st.info("⚠️ Please load a valid model to make predictions.")




