# app.py
"""
Streamlit app — clinician enters patient data manually, model predicts probability
of deterioration within 90 days and shows reasons (SHAP or feature importances).
Save as app.py and run: `streamlit run app.py`
"""
import json
from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -------------------------
# IMPORTANT: set_page_config MUST be first Streamlit call
# -------------------------
st.set_page_config(page_title="90-day Deterioration Risk (Clinician UI)",
                   page_icon="🩺",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Now it's safe to use other st.* APIs and decorators
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Helpers
# -------------------------
def safe_load_model(path: Path):
    """
    Load a joblib file and return (predict_pipeline, features_list_or_None, raw_payload)
    Works with:
      - a saved sklearn Pipeline / estimator
      - a dict containing {'pipeline': pipeline, 'features': [...]} (our training helper saved this format)
    """
    payload = joblib.load(path)
    # case: dict wrapper
    if isinstance(payload, dict):
        # common keys: 'pipeline', 'model', 'estimator'
        pipe = payload.get("pipeline") or payload.get("model") or payload.get("estimator")
        if pipe is None:
            # maybe the dictionary itself is the pipeline-like object -- try to find sklearn estimator inside
            # fallback: if payload contains 'clf' or similar
            for k in ("clf", "estimator", "rf", "model"):
                if k in payload:
                    pipe = payload[k]; break
        features = payload.get("features") or payload.get("columns") or payload.get("feature_names")
        return pipe, list(features) if (features is not None) else None, payload

    # case: plain estimator or pipeline
    pipe = payload
    features = None
    # try to get feature names if available on pipeline or final estimator
    try:
        if hasattr(pipe, "feature_names_in_"):
            features = list(pipe.feature_names_in_)
        elif hasattr(pipe, "named_steps"):
            # check last estimator for feature_names_in_
            last = list(pipe.named_steps.values())[-1]
            if hasattr(last, "feature_names_in_"):
                features = list(last.feature_names_in_)
    except Exception:
        features = None

    return pipe, features, payload

def get_last_estimator(pipe):
    """Return the final estimator object from a Pipeline or the object itself"""
    if isinstance(pipe, Pipeline):
        return list(pipe.named_steps.values())[-1]
    return pipe

def safe_predict_proba(pipe, X_df: pd.DataFrame):
    """
    Use the pipe to predict_proba when possible.
    If pipe is a Pipeline/estimator that accepts DataFrame -> ok.
    """
    # If pipeline exists, prefer pipe.predict_proba(X_df)
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_df)
    # If pipeline is dict wrapper with inner pipeline
    if isinstance(pipe, dict):
        # unlikely; caller should have unwrapped earlier
        if "pipeline" in pipe and hasattr(pipe["pipeline"], "predict_proba"):
            return pipe["pipeline"].predict_proba(X_df)
    raise ValueError("Model does not support predict_proba")

def try_shap_explain(pipe, features, X_df):
    """
    Attempt SHAP explanation for the single row X_df.
    Returns: (explanation_df, message) where explanation_df columns = ['feature','shap_value']
    If SHAP can't be computed, raises/returns None and message explains why.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap not installed (pip install shap)")

    # get raw estimator (tree-based) and prepared array to feed to explainer
    est = get_last_estimator(pipe)
    # If SHAP TreeExplainer works only for tree models. We'll still try and fallback gracefully.
    try:
        # Try to produce the transformed input that RF sees.
        # If pipe is Pipeline: try to find an 'imputer' or use the pipeline's transform if safe.
        if isinstance(pipe, Pipeline):
            # Try to use pipeline's preprocessing steps only (all except final estimator)
            # Build a shallow pipeline without final estimator if possible
            try:
                from sklearn.pipeline import Pipeline as SKPipeline
                steps = list(pipe.named_steps.items())
                if len(steps) > 1:
                    # create pipeline without last step
                    pre_steps = steps[:-1]
                    prepipe = SKPipeline(pre_steps)
                    X_for_model = prepipe.transform(X_df)
                else:
                    # no preprocess steps
                    X_for_model = X_df.values
            except Exception:
                # fallback: try imputer if present
                imputer = pipe.named_steps.get("imputer", None)
                if imputer is not None:
                    X_for_model = imputer.transform(X_df)
                else:
                    X_for_model = X_df.values
        else:
            X_for_model = X_df.values

        # Build TreeExplainer
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(X_for_model)
        # shap_values may be list (binary classification -> [class0, class1])
        if isinstance(shap_values, list):
            # pick class 1 contributions when possible
            if len(shap_values) > 1:
                shap_vals_sample = shap_values[1][0]
            else:
                shap_vals_sample = shap_values[0][0]
        else:
            shap_vals_sample = shap_values[0] if shap_values.ndim > 1 else shap_values

        # feature names mapping: if features provided use them, else try est.feature_names_in_
        if features is not None:
            names = features
        elif hasattr(est, "feature_names_in_"):
            names = list(est.feature_names_in_)
        else:
            # fallback: use DataFrame column names if dimension matches
            if X_for_model.ndim == 1:
                names = [f"f{i}" for i in range(X_for_model.shape[0])]
            else:
                names = list(X_df.columns) if hasattr(X_df, "columns") else [f"f{i}" for i in range(X_for_model.shape[1])]

        # If lengths mismatch, try to slice/zip with min length
        L = min(len(names), len(shap_vals_sample))
        df = pd.DataFrame({
            "feature": names[:L],
            "shap_value": shap_vals_sample[:L].astype(float)
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        return df, "shap"
    except Exception as e:
        # bubble exception to caller for graceful fallback
        raise

# -------------------------
# Sidebar: model selection / upload
# -------------------------
st.sidebar.header("Model")
model_file = st.sidebar.text_input("Model file path (joblib)", value="models/rf_last90d.joblib")
st.sidebar.markdown("Or upload a `.joblib` model file below (it will be loaded in-memory).")
uploaded_model = st.sidebar.file_uploader("Upload model (.joblib)", type=["joblib", "pkl"])

# -------------------------
# Main title and instructions
# -------------------------
st.title("🩺 90-day Deterioration Risk — Clinician Input UI")
st.markdown(
    "Enter the patient's measurements below. The app will predict the probability that the patient's "
    "health will deteriorate in the next 90 days and explain the top drivers (SHAP if available)."
)

# -------------------------
# Load model (from file or upload)
# -------------------------
model_load_error = None
pipe = None
feature_list = None
raw_payload = None

try:
    if uploaded_model is not None:
        # save to a temp path and load
        tmp = Path("models/_uploaded_model.joblib")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(uploaded_model.getbuffer())
        pipe, feature_list, raw_payload = safe_load_model(tmp)
        st.sidebar.success("Loaded uploaded model")
    else:
        p = Path(model_file)
        if not p.exists():
            st.sidebar.warning(f"Model file not found: {model_file}")
        else:
            pipe, feature_list, raw_payload = safe_load_model(p)
            st.sidebar.success(f"Loaded model: {p.name}")
except Exception as e:
    model_load_error = f"Error loading model: {e}\n{traceback.format_exc()}"
    st.sidebar.error("Failed to load model. See message below.")
    st.sidebar.text(model_load_error)

# -------------------------
# Build input form
# -------------------------
st.subheader("Patient data (manual entry)")

# If we detected model features, build input widgets for each feature.
# Otherwise present a clinician-friendly default form of typical vitals + counters,
# and provide an advanced JSON input area.
user_values = {}
if pipe is None:
    st.warning("Model not loaded yet. Enter values but prediction will be disabled until a model is loaded.")
# Build two-column layout for inputs
col1, col2 = st.columns(2)

if feature_list:
    st.info(f"Detected model expects {len(feature_list)} features. Auto-generated input fields below.")
    with st.form("feature_form"):
        # create inputs in order, but keep UI usable by splitting into two columns
        for i, fname in enumerate(feature_list):
            label = fname.replace("_", " ")
            # decide column to place input
            target_col = col1 if (i % 2 == 0) else col2
            # default: 0.0, allow float
            try:
                default = float(0.0)
            except Exception:
                default = 0.0
            # use step and format generic floats
            user_values[fname] = target_col.number_input(label, value=default, format="%.4f", key=f"f_{i}")
        submitted = st.form_submit_button("Predict")
else:
    st.info("No model feature list detected → showing a clinician-friendly input form. "
            "If your saved model exposes exact feature names, upload it in the sidebar to auto-generate inputs.")
    with st.form("clinician_form"):
        # Typical vitals + utilization used commonly in your pipeline examples
        age = col1.number_input("Age (years)", min_value=0, max_value=130, value=65)
        heart_rate = col1.number_input("Heart rate (bpm)", min_value=10, max_value=250, value=80)
        systolic_bp = col1.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
        diastolic_bp = col1.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
        resp_rate = col1.number_input("Respiratory rate (/min)", min_value=5, max_value=60, value=18)

        spo2 = col2.number_input("SpO2 (%)", min_value=50, max_value=100, value=96)
        bmi = col2.number_input("BMI", min_value=8.0, max_value=70.0, value=24.5)
        glucose = col2.number_input("Blood glucose (mg/dL)", min_value=40.0, max_value=500.0, value=110.0)
        enc_180d = col2.number_input("Encounters (last 180 days)", min_value=0, max_value=100, value=2)
        ed_180d = col2.number_input("ED visits (last 180 days)", min_value=0, max_value=50, value=1)
        inpatient_180d = col2.number_input("Inpatient visits (last 180 days)", min_value=0, max_value=50, value=0)
        med_coverage_pct = col2.slider("Medication coverage (last 90 days) %", 0, 100, 80)

        submitted = st.form_submit_button("Predict")
        # convert to user_values mapping that's compatible with training features if later auto-detected
        user_values = {
            "age": float(age),
            "heart_rate_last": float(heart_rate),
            "systolic_bp_last": float(systolic_bp),
            "diastolic_bp_last": float(diastolic_bp),
            "resp_rate_last": float(resp_rate),
            "spo2_last": float(spo2),
            "bmi_last": float(bmi),
            "glucose_last": float(glucose),
            "enc_count_180d": float(enc_180d),
            "ed_count_180d": float(ed_180d),
            "inpatient_count_180d": float(inpatient_180d),
            # many models expect med_coverage_frac_90d (0..1)
            "med_coverage_frac_90d": float(med_coverage_pct / 100.0),
        }

# Advanced: allow clinician to paste a JSON mapping of feature->value (overrides auto fields)
st.markdown("---")
st.markdown("**Advanced:** paste JSON mapping of feature name -> numeric value (optional).")
json_input = st.text_area("Paste JSON here (e.g. {\"heart_rate_last\": 82, \"bmi_last\": 28.1})", height=80)
if json_input.strip():
    try:
        parsed = json.loads(json_input)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                # override user_values
                try:
                    user_values[str(k)] = float(v)
                except Exception:
                    user_values[str(k)] = v
            st.success("Parsed JSON and merged values.")
        else:
            st.error("JSON must be an object/dictionary.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

# -------------------------
# Perform prediction when form submitted
# -------------------------
if 'submitted' in locals() and submitted:
    if pipe is None:
        st.error("Model not loaded. Upload or specify a valid model in the sidebar.")
    else:
        # Build DataFrame X that matches model's expected feature ordering if possible
        if feature_list:
            # Ensure all model features have a numeric value (default 0.0 if missing)
            row = {}
            for f in feature_list:
                if f in user_values:
                    row[f] = user_values[f]
                else:
                    # fallback default 0
                    row[f] = 0.0
            X = pd.DataFrame([row], columns=feature_list)
        else:
            # use provided user_values keys as columns
            X = pd.DataFrame([user_values])

        st.markdown("### 🔎 Prediction")
        try:
            proba = None
            # If pipe is wrapper dict, unwrap to actual predictive object
            predictive = pipe
            if isinstance(pipe, dict) and "pipeline" in pipe:
                predictive = pipe["pipeline"]
            # If pipeline is sklearn Pipeline or estimator, use it directly for predict_proba
            if hasattr(predictive, "predict_proba"):
                # many sklearn pipelines accept DataFrame; if not, fallback to numpy array
                try:
                    proba_arr = predictive.predict_proba(X)
                except Exception:
                    proba_arr = predictive.predict_proba(X.values)
                # probability for positive class
                if proba_arr.shape[1] == 1:
                    # some models return single column (rare) -> treat as proba for class1
                    proba = float(proba_arr[:, 0][0])
                else:
                    proba = float(proba_arr[:, 1][0])
            else:
                raise RuntimeError("Loaded model does not support predict_proba. Ensure it's a classifier pipeline with predict_proba.")
            pct = proba * 100.0
            # show result
            if pct >= 75:
                st.error(f"Predicted risk (90 days): **{pct:.1f}%** — HIGH RISK")
            elif pct >= 40:
                st.warning(f"Predicted risk (90 days): **{pct:.1f}%** — MODERATE RISK")
            else:
                st.success(f"Predicted risk (90 days): **{pct:.1f}%** — LOW RISK")
            st.caption("Probability shown is model output for deterioration within next 90 days.")
        except Exception as e:
            st.error(f"Prediction failed: {e}\n{traceback.format_exc()}")

        # -------------------------
        # Explanation (SHAP preferred; fallback to feature importance)
        # -------------------------
        st.markdown("### 🧾 Explanation / Reasons")
        explanation_shown = False
        # try SHAP
        try:
            df_shap, method = try_shap_explain(pipe, feature_list, X)
            # Show top contributors
            top = df_shap.head(8).copy()
            # signed contributions bar chart
            top_pos = top.sort_values("shap_value", ascending=True)
            st.write("Top contributions (signed SHAP values — positive increases risk, negative decreases risk):")
            st.table(top[["feature", "shap_value"]].assign(shap_value=lambda d: d["shap_value"].round(4)))
            # matplotlib horizontal bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, min(0.35 * len(top_pos), 4)))
            ax.barh(top_pos["feature"], top_pos["shap_value"])
            ax.set_xlabel("SHAP value (signed)")
            ax.axvline(0, color="k", linewidth=0.6)
            st.pyplot(fig)
            # textual reasons
            pos = top[top["shap_value"] > 0].head(3)
            neg = top[top["shap_value"] < 0].head(3)
            reasons = []
            if not pos.empty:
                reasons.append("Increased risk because: " + ", ".join([f"{r['feature']} ({r['shap_value']:.3f})" for _, r in pos.iterrows()]))
            if not neg.empty:
                reasons.append("Decreased risk because: " + ", ".join([f"{r['feature']} ({r['shap_value']:.3f})" for _, r in neg.iterrows()]))
            if reasons:
                st.info("\n\n".join(reasons))
            explanation_shown = True
        except Exception as e:
            # SHAP not available or failed: fallback to feature importances if available
            st.info("SHAP explanation not available or failed. Falling back to feature importances if possible.")
            try:
                est = get_last_estimator(pipe)
                if hasattr(est, "feature_importances_"):
                    fi = np.array(est.feature_importances_, dtype=float)
                    # determine feature names to pair with importances
                    if feature_list and len(feature_list) == fi.shape[0]:
                        names = feature_list
                    elif hasattr(est, "feature_names_in_"):
                        names = list(est.feature_names_in_)
                    else:
                        # fallback: dataframe column names if lengths match
                        if X.shape[1] == fi.shape[0]:
                            names = list(X.columns)
                        else:
                            names = [f"f{i}" for i in range(fi.shape[0])]
                    df_fi = pd.DataFrame({"feature": names, "importance": fi})
                    df_fi = df_fi.sort_values("importance", ascending=False).reset_index(drop=True)
                    st.write("Model feature importances (higher => more important):")
                    st.table(df_fi.head(10).assign(importance=lambda d: d["importance"].round(4)))
                    explanation_shown = True
                else:
                    st.warning("Model has no feature_importances_. Cannot provide an automated explanation.")
            except Exception as e2:
                st.error(f"Failed to compute fallback explanation: {e2}\n{traceback.format_exc()}")

        if not explanation_shown:
            st.info("No explanation could be produced. You may install `shap` (`pip install shap`) and/or "
                    "ensure the saved model is a tree-based estimator (RandomForest/XGBoost) wrapped in a Pipeline "
                    "or a dict containing 'pipeline' and 'features' keys.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("If you want me to auto-generate a simpler UI with only specific vitals shown, tell me which exact feature names your model expects (or upload the model).")
