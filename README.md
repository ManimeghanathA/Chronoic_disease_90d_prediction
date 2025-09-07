# Chronic Diseases — 90-day Deterioration Risk

**Status:** prototype — **not for clinical use**. Read the *Limitations & Next Steps* section carefully.

This repository contains the end-to-end prototype pipeline that converts raw Synthea CSVs into per-patient features, builds a strict model-ready dataset, trains a Random Forest baseline, generates explainability & diagnostic plots (SHAP, calibration, confusion matrix), and ships a Streamlit clinician UI for manual predictions.

---

## Contents / Quick Map

- `scripts/ingest_synthea.py` — Reads `observations.csv`, cleans, converts `VALUE` to numeric, floors to the day, pivots to a patient-day table, and saves `patient_daily_observations.csv`.
- `scripts/prepare_dataset.py` — Takes per-patient aggregated features (`features_labels_last90d_per_patient.csv`), computes coverage stats, medication coverage, follow-up checks, filters low-coverage and constant features, and saves `features_labels_cleaned_for_model.csv`.
- `scripts/make_model_ready.py` — Performs final feature selection (≥5% non-null), fills med coverage NaNs, and produces:
  - `features_labels_model_ready_inclusive.csv`
  - `features_labels_model_ready_strict.csv` (only patients with confirmed ≥90-day follow-up)
- `train_model.py` — Trains a RandomForest (n=200, max_depth=10, class_weight='balanced'), evaluates (classification report, ROC-AUC, AUPRC), and saves the model & plots to `models/`.
- `app.py` — Streamlit clinician UI for single-patient predictions & explanations (SHAP or feature importance fallback).
- `models/` — Saved artifacts (e.g., `rf_strict.joblib`, `shap_summary_strict.png`, `calibration_strict.png`, `confusion_matrix_strict.png`).

---

## 1) How the Dataset Was Created — Preprocessing & Cleaning (Detailed)

### Raw Files Used (Synthea Output)
- `observations.csv` — A flat table of observations with columns: `DATE`, `PATIENT`, `ENCOUNTER`, `CATEGORY`, `CODE`, `DESCRIPTION`, `VALUE`, `UNITS`, `TYPE`.
- `medications.csv` (optional) — Used to compute medication coverage in the last 90 days.
- `encounters.csv` (optional) — Used to compute whether each patient has ≥90 days of follow-up after their `last_date`.

### `ingest_synthea.py` (Observations → Patient-Day Table)
**Behavior**
- Reads `observations.csv` robustly (tries a standard CSV reader, falls back to the Python engine on failure).
- Auto-detects columns: `PATIENT` → `patient_id`, `DATE` → `date`, `DESCRIPTION` → `obs_name`, `VALUE` → `obs_value`.
- Cleans fields:
  - `date` → `datetime` (coerces failures to `NaT` and drops them).
  - `patient_id`, `obs_name` → trimmed strings; `obs_name` lowercased.
  - `obs_value` → converted to numeric where possible; non-numeric → `NaN`.
  - Drops rows missing `patient_id` or `date`.
- Adds `date_day` by flooring timestamps to the day (grouping on day granularity).
- Pivot: one row per (`patient_id`, `date_day`), columns = distinct `obs_name`, values = the first numeric observation for that day (`pivot_table(..., aggfunc='first')`).
- **Output:** `patient_daily_observations.csv` — a wide table useful for deriving features (last value, counts, trends).

### `prepare_dataset.py` (Aggregate → Cleaned for Modeling)
**Behavior**
- Loads `features_labels_last90d_per_patient.csv` — should contain per-patient aggregated features computed from the daily table for the last 90 days and a binary label `deterioration_90d`.
- Ensures `last_date` is parsed to `datetime`.
- Reports label balance & basic diagnostics.
- **Feature coverage:**
  - Computes non-null counts per feature.
  - Flags features with extremely low coverage (e.g., `<1%` non-null).
  - Keeps a candidate set of features with coverage ≥ 5% (configurable).
  - Drops constant (zero-variance) features.
- **Medication coverage:**
  - If `medications.csv` exists, detects patient/start/stop columns, computes `med_coverage_days_90d` = number of days in the 90-day window a patient had medication active, and `med_coverage_frac_90d` = fraction (0..1).
- **Follow-up availability:**
  - If `encounters.csv` exists, computes the global maximum encounter date and checks whether each patient’s `last_date` is at least 90 days before that global max. Adds `has_90d_followup` (0/1).
- **Save:** `features_labels_cleaned_for_model.csv`.

### `make_model_ready.py` (Final Model-Ready Filtering)
**Behavior**
- Re-loads the cleaned file and:
  - Drops features with `<5%` non-null (consistent threshold).
  - Drops constant columns.
  - Fills `med_coverage_*` NaNs with `0` (explicit: no medication recorded → 0 coverage).
- **Saves two versions:**
  - `features_labels_model_ready_inclusive.csv` — all patients.
  - `features_labels_model_ready_strict.csv` — subset with `has_90d_followup == 1`.

**Notes**
- The strict dataset aims to ensure the outcome label is observable (90 days of follow-up exists) to avoid label leakage or censoring artifacts.

---

## 2) What the “Strict” Dataset Represents (Why We Made It)

**Goal:** Ensure every patient in the training/testing set has at least 90 days of follow-up available after `last_date`, so the `deterioration_90d` label is reliably observed (not censored).

**Tradeoff:** This increases label reliability but reduces sample size. The observed strict dataset is small (training output shows only 148 rows), causing statistical and training challenges discussed below.

---

## 3) Model: Design, Training, and Saved Artifacts

### Model Used (Baseline)
- **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters:** 
  - `n_estimators=200`
  - `max_depth=10`
  - `class_weight="balanced"`
  - `random_state=42`
- **Train/test split:** `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`

### Saved Files
- Model: `models/rf_strict.joblib`
- SHAP summary: `models/shap_summary_strict.png`
- Calibration plot: `models/calibration_strict.png`
- Confusion matrix: `models/confusion_matrix_strict.png`

### Why Random Forest Was Used (Rationale)
- Robust baseline: works out-of-the-box without heavy feature scaling and captures nonlinearities and feature interactions.
- Interpretability: tree-based models expose `feature_importances_` and are compatible with SHAP `TreeExplainer` for local/global explanations.
- Fast prototyping: relatively quick to train and tolerant of noisy features (but not to NaNs unless preprocessed).

### Training & Explainability
- Script trains on 80% of data and evaluates on 20%:
  - Predicts class labels and probabilities (`predict_proba`).
  - Calculates classification report (precision, recall, f1), AUROC, AUPRC.
  - Produces a SHAP summary (global feature contributions) and stores a static image.
  - Produces calibration curve and confusion matrix images for diagnostics.

---

## 4) Interpretation of Your Current Training Results (Honest Forensic Read)

**Exact Printed Metrics From Your Run**

- ✔ Loaded strict dataset with label column: `deterioration_90d`
- Dataset test set (from classification report): `support`: **class 0 = 143**, **class 1 = 5** → **total 148 rows**.

**Classification report:**

```
Class 0: precision 0.97, recall 1.00, f1 0.98
Class 1: precision 0.00, recall 0.00, f1 0.00 (support 5)
Accuracy: 0.97
ROC-AUC: 0.6951
AUPRC: 0.0508
Model and plots saved to models/.
```

**What This Means**
- High accuracy (0.97) is misleading: the dataset is heavily imbalanced (very few positives), so predicting everyone as negative yields high accuracy.
- Recall = 0.00 for positives: the model predicted no positives in the test set — it failed to detect any deterioration cases.
- ROC-AUC ≈ 0.70: suggests the model’s predicted probabilities have some ranking ability (better than random), but with very few positives this metric can be unstable.
- AUPRC extremely low (0.05): typical for highly imbalanced problems and indicates a poor precision/recall tradeoff for the positive class.
- **Bottom line:** With only ~148 labeled rows (5 positives), the dataset is too small and too imbalanced to train a clinically useful classifier. This run is a prototype sanity check, not a validated model.

---

## 5) Explainability & Diagnostics (How to Read the Plots)

- **SHAP summary (`shap_summary_strict.png`):** Shows global feature impact distribution. Features are ordered by importance; points per instance show the sign & magnitude of their contribution. Positive SHAP values increase predicted probability of deterioration; negative decrease it. Use SHAP to validate whether top drivers make clinical sense.
- **Calibration curve (`calibration_strict.png`):** Plots mean predicted probability (x) vs. observed fraction of positives (y) in bins. If the curve follows the diagonal, probabilities are well calibrated. If not, consider Platt scaling or isotonic calibration.
- **Confusion matrix (`confusion_matrix_strict.png`):** Shows counts of true vs. predicted classes. Given the current run, the matrix will show many true negatives and likely zero true positives.

---

## 6) Streamlit UI (`app.py`) — What It Does & How to Use It

### Key Features
- Sidebar: specify model file path (default `models/rf_last90d.joblib`) or upload a `.joblib` model. Your training saved `models/rf_strict.joblib`, so either change the sidebar text or rename the file.
- Auto-generate input form: when the model contains an explicit features list or `feature_names_in_`:
  - If the model exposes feature names, the app creates one numeric input per feature.
  - Otherwise, the app shows a clinician-friendly form (vitals + counts) and a JSON paste box for advanced users.
- On prediction:
  - Uses the model's `predict_proba` → shows probability and a risk band:
    - `≥ 75%` — **HIGH risk**
    - `≥ 40%` — **MODERATE risk**
    - else — **LOW risk**
  - Attempts a SHAP explanation for the single instance (preferred). If SHAP fails or isn't installed, it falls back to a `feature_importances_` table.
- Advanced: accepts a dictionary-wrapped model saved as `{'pipeline': pipeline, 'features': [...], ...}` — useful when you store preprocessing steps with the model.

### How to Run the UI
```bash
# from repo root
pip install -r requirements.txt   # see Dependencies below
streamlit run app.py
```

**Notes / Tips**
- If the model expects strict ordering of features, upload the same model used for training (or a dict that includes `features`).
- The app will attempt to apply pipeline preprocessing (if the saved object is a `Pipeline`) before SHAP / explanation.

---

## 7) Recommendations — Immediate Next Steps (Priority List)

### Data & Labeling (Highest Priority)
- Increase positive samples. 5 positives in the test set is far too few. Collect more labeled events or relax strictness temporarily to enlarge data (but watch label quality).
- Avoid over-pruning early — validate how many patients remain after each filter (especially `has_90d_followup`).
- Stratify by demographic groups and check class balance across subgroups to detect sampling bias.

### Modeling & Evaluation
- Build a proper preprocessing `Pipeline` (`sklearn.pipeline.Pipeline`) that includes `SimpleImputer` (median/mean), missingness indicators, and any encoders — save the pipeline + model together.
- Address imbalance:
  - Try oversampling (SMOTE for numeric), undersampling the majority, or use stronger class weighting / focal loss.
- Use AUPRC as the main metric for imbalanced detection tasks.
- Cross-validation & repeated runs: use stratified k-fold CV and report mean ± std; consider nested CV for tuning.
- Probability calibration: apply `CalibratedClassifierCV` if probabilities are essential.
- Try gradient boosting (LightGBM / XGBoost / CatBoost) as alternatives — they often perform better on tabular medical data.
- Feature engineering: last value, min/max, slope, trend, missingness fraction, days since last obs, lab flags (abnormal/normal), comorbidity counts.
- Model monitoring: log predictions and outcomes to monitor data drift and recalibrate periodically.

### Explainability & Clinical Validation
- Review SHAP explanations with clinicians — do top drivers make clinical sense?
- External validation on an independent dataset before claiming generalizability.

### UI & Deployment
- UI improvements: display model name, version, training date, and dataset summary (rows, positives) in the sidebar.
- Add CSV upload for batch predictions and an audit log for predictions/outcomes.
- Allow clinicians to mark predictions as correct/incorrect for a feedback loop (active learning).
- Security & privacy: remove PHI, add authentication, HTTPS, and logging controls before real-world deployment.
- CI/CD & reproducibility: containerize the app, freeze the environment, and version models with tags.

---

## 8) Limitations — Why This Is a Prototype and Not Production
- Small sample & severe class imbalance → unstable models and inflated metrics by accuracy.
- Synthea synthetic data may not represent real clinical distributions.
- No formal external validation or clinical safety checks.
- No auditing / drift monitoring in the current code; data and the model may degrade over time.
- Regulatory & ethical considerations: clinical decision support requires rigorous validation, documentation, and possibly clearance depending on jurisdiction.

---

## 9) How to Reproduce (Commands & Environment)

### 1. Create Python Environment (Recommended)
```bash
python -m venv .venv
# on Windows
.venv\Scripts\activate
# on macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Dependencies (Example `requirements.txt`)
Create `requirements.txt` with at least:
```
pandas
numpy
scikit-learn
matplotlib
shap
joblib
streamlit
```
Then:
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing (Adjust paths in scripts or pass args)
```bash
python scripts/ingest_synthea.py
# generate features_labels_last90d_per_patient.csv using your feature engineering step (not included)
python scripts/prepare_dataset.py
python scripts/make_model_ready.py
```

### 4. Train Model
```bash
python train_model.py
# This will save the model to models/rf_strict.joblib and diagnostics in models/
```

### 5. Run Streamlit UI
```bash
streamlit run app.py
```

---

## 10) Helpful Development Notes & Gotchas
- The training code expects the strict CSV at `/content/features_labels_model_ready_strict.csv` in the current version — update dataset paths in the script to match your local layout (e.g., `data/processed/...`).
- Streamlit's default model path is `models/rf_last90d.joblib` — either rename the saved `rf_strict.joblib` or change the default path in the sidebar variable.
- When saving a model for the UI, prefer saving a wrapped dict:
```python
joblib.dump({'pipeline': pipeline, 'features': feature_list}, "models/my_model.joblib")
```
This makes the Streamlit auto-UI generation robust.

---

## 11) Suggested Roadmap (Short / Medium / Long-Term)

### Short (Days)
- Fix paths & run the pipeline end-to-end on a larger sample.
- Save the pipeline + features list; update Streamlit default path.

### Medium (Weeks)
- Improve feature engineering and imputation.
- Balance classes; run robust CV; tune hyperparameters.
- Get clinician feedback on SHAP drivers.

### Long (Months)
- External validation on real patient data.
- CI/CD, monitoring, privacy & regulatory compliance, usability testing with clinicians.

---

## 12) Contribution, Issues, License
Feel free to open issues / PRs in the GitHub repo for:
- Adding/improving preprocessing steps.
- Productionizing the pipeline.
- Creating tests and example notebooks.

Choose an appropriate license for your project (e.g., MIT for research prototypes).

---

## Final Pep Talk
You’ve built a very useful prototype pipeline: structured ingestion from Synthea → thoughtful coverage filtering → a strict labeled dataset → model training with explainability and a clinician UI. The current metrics show the right direction (a realistic prototype) but also clearly highlight the classical ML lifecycle needs: more labeled positives, robust preprocessing pipelines, and careful evaluation. With additional data and the engineering improvements above, this can evolve into a much stronger system suitable for proper validation.


