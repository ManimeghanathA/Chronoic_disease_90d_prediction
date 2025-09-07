# scripts/ingest_synthea.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

CSV_DIR = Path(r"C:\Users\manim\synthea\output\csv")
OUT_DIR = Path(r"C:\Users\manim\synthea_project\data\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

obs_path = CSV_DIR / "observations.csv"
if not obs_path.exists():
    print(f"ERROR: {obs_path} not found. Check your Synthea output folder.")
    sys.exit(1)

print("Reading observations.csv ... (this may take a moment for large files)")
# Try a normal read first; if it fails, fallback with engine='python'
try:
    obs = pd.read_csv(obs_path, low_memory=False)
except Exception as e:
    print("Standard read failed, retrying with engine='python' ...", e)
    obs = pd.read_csv(obs_path, low_memory=False, engine='python', encoding='utf-8', errors='replace')

print("Columns in observations.csv:", list(obs.columns)[:100])

# find patient id column
patient_col = None
patient_candidates = ['patient', 'patient_id', 'patientidentifier', 'subject', 'patientuuid']
for c in obs.columns:
    if any(pc in c.lower() for pc in patient_candidates):
        patient_col = c
        break

# date/time column
date_col = None
date_candidates = ['date', 'time', 'effective', 'start', 'timestamp', 'obs_date']
for c in obs.columns:
    if any(dc in c.lower() for dc in date_candidates):
        date_col = c
        break

# description/name column
desc_col = None
desc_candidates = ['description', 'display', 'name', 'code_display', 'obs_name', 'code']
for c in obs.columns:
    if any(dc in c.lower() for dc in desc_candidates):
        desc_col = c
        break

# value column (numeric results)
value_col = None
value_candidates = ['value', 'value_quantity', 'value_numeric', 'result', 'valuequantity']
for c in obs.columns:
    if any(vc in c.lower() for vc in value_candidates):
        value_col = c
        break

# fallback tries
if patient_col is None:
    for c in obs.columns:
        if 'patient' in c.lower():
            patient_col = c; break

if date_col is None:
    for c in obs.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            date_col = c; break

if desc_col is None:
    for c in obs.columns:
        if 'desc' in c.lower() or 'display' in c.lower() or 'code' in c.lower():
            desc_col = c; break

if value_col is None:
    # try numeric-like columns
    for c in obs.columns:
        sample = obs[c].dropna().astype(str).head(200)
        if not sample.empty and sample.str.match(r'^[+-]?\d+(\.\d+)?$').any():
            value_col = c
            break

if not all([patient_col, date_col, desc_col, value_col]):
    print("Failed to auto-detect all required columns.")
    print("Detected -> patient:", patient_col, " date:", date_col, " desc:", desc_col, " value:", value_col)
    print("Please open observations.csv and inspect headers or edit this script to set the correct column names.")
    sys.exit(1)

print("Using columns:", patient_col, date_col, desc_col, value_col)

# Subset and rename
obs2 = obs[[patient_col, date_col, desc_col, value_col]].rename(columns={
    patient_col: "patient_id",
    date_col: "date",
    desc_col: "obs_name",
    value_col: "obs_value"
})

# Parse date/time
obs2['date'] = pd.to_datetime(obs2['date'], errors='coerce')
print("Parsed dates - sample nulls:", obs2['date'].isna().sum(), " of ", len(obs2))

# Clean strings
obs2['patient_id'] = obs2['patient_id'].astype(str).str.strip()
obs2['obs_name'] = obs2['obs_name'].astype(str).str.strip().str.lower()

# numeric conversion helper
def to_numeric_safe(x):
    try:
        return float(str(x).strip())
    except:
        return np.nan

obs2['obs_value_num'] = obs2['obs_value'].apply(to_numeric_safe)

# drop rows without date or patient
obs2 = obs2.dropna(subset=['patient_id','date']).copy()

# floor to day
obs2['date_day'] = obs2['date'].dt.floor('d')

print("Rows after cleaning:", len(obs2))
print("Pivoting observations to patient-day table (this may take a moment)...")

# reduce unique observation names: keep top N frequent to avoid huge columns (optional)
# top_obs = obs2['obs_name'].value_counts().nlargest(500).index.tolist()
# obs2 = obs2[obs2['obs_name'].isin(top_obs)]

pivot = obs2.pivot_table(index=['patient_id','date_day'],
                         columns='obs_name',
                         values='obs_value_num',
                         aggfunc='first').reset_index()

out_path = OUT_DIR / "patient_daily_observations.csv"
pivot.to_csv(out_path, index=False)
print("Saved patient daily observations to:", out_path)
print("Columns in pivot (sample):", list(pivot.columns)[:80])
print("Number of patient-day rows:", pivot.shape[0])
