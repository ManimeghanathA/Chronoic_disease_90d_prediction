# build_features_and_labels.py (fixed + more robust)
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

CSV_DIR = Path(r"C:\Users\manim\synthea\output\csv")
PROCESSED_DIR = Path(r"C:\Users\manim\synthea_project\data\processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

patient_daily_path = PROCESSED_DIR / "patient_daily_observations.csv"
if not patient_daily_path.exists():
    raise SystemExit("Run ingest_synthea.py first to create patient_daily_observations.csv")

print("Loading patient daily observations...")
patient_daily = pd.read_csv(patient_daily_path, parse_dates=['date_day'], low_memory=False)
# Ensure patient id is string and clean
patient_daily['patient_id'] = patient_daily['patient_id'].astype(str).str.strip()

# Load encounters and meds
enc = pd.read_csv(CSV_DIR / "encounters.csv", low_memory=False)
meds = pd.read_csv(CSV_DIR / "medications.csv", low_memory=False)

# Helper: pick a column name from candidates (case-insensitive substring match)
def pick_col(cols, candidates):
    cols_lower = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_lower):
            if cand in c:
                return list(cols)[i]
    return None

# Identify encounter columns
enc_cols = list(enc.columns)
enc_patient_col = pick_col(enc_cols, ['patient', 'patient_id', 'patientidentifier'])
enc_date_col = pick_col(enc_cols, ['start', 'date', 'date_time', 'timestamp', 'time'])
enc_class_col = pick_col(enc_cols, ['class', 'encounter', 'encounterclass', 'type'])

if enc_patient_col is None:
    raise SystemExit("Cannot find patient id column in encounters.csv")
if enc_date_col is None:
    # fallback: any column name that looks like a date/time
    for c in enc_cols:
        if 'date' in c.lower() or 'time' in c.lower() or 'start' in c.lower():
            enc_date_col = c
            break

print("Encounter columns used:", enc_patient_col, enc_date_col, enc_class_col)

# Normalize patient id types in encounters and meds
enc[enc_patient_col] = enc[enc_patient_col].astype(str).str.strip()
if 'patient' in meds.columns:
    meds['patient'] = meds['patient'].astype(str).str.strip()
else:
    # try to detect a patient column in meds
    meds_patient_col = pick_col(list(meds.columns), ['patient', 'patient_id'])
    if meds_patient_col:
        meds[meds_patient_col] = meds[meds_patient_col].astype(str).str.strip()
        meds.rename(columns={meds_patient_col: 'patient'}, inplace=True)

# parse encounter dates safely
enc[enc_date_col] = pd.to_datetime(enc[enc_date_col], errors='coerce')

# Robust slope function: returns slope in units per day (float) or np.nan
def compute_slope(dates, values, min_points=2):
    """
    dates: Series or index-like of datetimes
    values: Series-like numeric (or convertible)
    returns: slope (value change per day) or np.nan
    """
    # Build df and coerce types
    d_ser = pd.to_datetime(pd.Series(dates).copy(), errors='coerce')
    v_ser = pd.to_numeric(pd.Series(values).copy(), errors='coerce')
    df = pd.DataFrame({'date': d_ser, 'y': v_ser}).dropna().sort_values('date')
    if df.shape[0] < min_points:
        return np.nan
    # convert to days since epoch (float)
    # use int64 ns -> seconds -> days
    try:
        ts_days = df['date'].view('int64') / 1e9 / 86400.0
    except Exception:
        ts_days = df['date'].astype('int64') / 1e9 / 86400.0
    x = ts_days - ts_days.mean()  # center to reduce numerical issues
    y = df['y'].values.astype(float)
    try:
        slope = np.polyfit(x, y, 1)[0]  # units: value / day
        return float(slope)
    except Exception:
        return np.nan

# pick measure columns from patient_daily (exclude patient_id and date_day)
measure_cols = [c for c in patient_daily.columns if c not in ('patient_id', 'date_day')]
print("Detected measure columns (sample):", measure_cols[:40])

patients = patient_daily['patient_id'].unique()
rows = []

for pid in tqdm(patients, desc="Patients"):
    g = patient_daily[patient_daily['patient_id'] == pid].sort_values('date_day')
    if g.empty:
        continue
    last_date = g['date_day'].max()
    if pd.isna(last_date):
        continue
    window_start = last_date - pd.Timedelta(days=90)
    slope_start = last_date - pd.Timedelta(days=30)

    # Only use rows in the last 90-day window (strictly after window_start up to last_date)
    window_df = g[(g['date_day'] > window_start) & (g['date_day'] <= last_date)].copy()
    if window_df.shape[0] < 1:
        continue

    record = {'patient_id': pid, 'last_date': last_date}

    # baseline (before the 90-day window) for delta-from-baseline features
    baseline_df = g[g['date_day'] <= window_start]

    for m in measure_cols:
        # cast column to numeric where possible
        series = window_df[['date_day', m]].dropna()
        # ensure we can convert values to numeric where necessary
        if series.shape[0] == 0:
            record[f'{m}_last'] = np.nan
            record[f'{m}_mean_7'] = np.nan
            record[f'{m}_mean_30'] = np.nan
            record[f'{m}_std_30'] = np.nan
            record[f'{m}_slope_30d'] = np.nan
            record[f'{m}_missing_frac'] = 1.0
            record[f'{m}_delta_baseline'] = np.nan
            continue

        # last value (take last available numeric value)
        last_val = pd.to_numeric(series[m], errors='coerce').dropna()
        record[f'{m}_last'] = float(last_val.iloc[-1]) if not last_val.empty else np.nan

        # rolling aggregates: make a time-indexed series
        try:
            rec_series_full = window_df.set_index('date_day')[m].sort_index()
            rec_series_full = pd.to_numeric(rec_series_full, errors='coerce')
            # mean over last 7 and 30 days (these yield NaN if not enough data)
            try:
                record[f'{m}_mean_7'] = rec_series_full.rolling('7D').mean().dropna().iloc[-1]
            except Exception:
                record[f'{m}_mean_7'] = rec_series_full.mean()  # fallback
            try:
                record[f'{m}_mean_30'] = rec_series_full.rolling('30D').mean().dropna().iloc[-1]
            except Exception:
                record[f'{m}_mean_30'] = rec_series_full.mean()
            try:
                record[f'{m}_std_30'] = rec_series_full.rolling('30D').std().dropna().iloc[-1]
            except Exception:
                record[f'{m}_std_30'] = np.nan
        except Exception:
            record[f'{m}_mean_7'] = np.nan
            record[f'{m}_mean_30'] = np.nan
            record[f'{m}_std_30'] = np.nan

        # slope over last 30 days using compute_slope (require at least 2 points)
        slope_df = window_df[window_df['date_day'] > slope_start][['date_day', m]].dropna()
        record[f'{m}_slope_30d'] = compute_slope(slope_df['date_day'], slope_df[m], min_points=2) if slope_df.shape[0] >= 2 else np.nan

        # missing fraction: proportion of days in last 90-day window without a measurement
        days_with_val = window_df[window_df[m].notna()]['date_day'].nunique()
        # use fixed 90-day denominator (consistent across patients) — you can change to actual window length if desired
        record[f'{m}_missing_frac'] = 1.0 - (days_with_val / 90.0)

        # delta baseline: last - median(before-window)
        if baseline_df.shape[0] > 0 and m in baseline_df.columns:
            bmedian = pd.to_numeric(baseline_df[m], errors='coerce').median()
            last_value = record[f'{m}_last']
            record[f'{m}_delta_baseline'] = (last_value - bmedian) if (not pd.isna(last_value) and not pd.isna(bmedian)) else np.nan
        else:
            record[f'{m}_delta_baseline'] = np.nan

    # encounter-based counts from encounters.csv
    pid_enc = enc[enc[enc_patient_col] == pid]
    pid_enc = pid_enc.dropna(subset=[enc_date_col])
    if not pid_enc.empty:
        pid_enc_dates = pd.to_datetime(pid_enc[enc_date_col], errors='coerce')
        record['enc_count_180d'] = ((pid_enc_dates >= (last_date - pd.Timedelta(days=180))) & (pid_enc_dates <= last_date)).sum()
        record['enc_count_365d'] = ((pid_enc_dates >= (last_date - pd.Timedelta(days=365))) & (pid_enc_dates <= last_date)).sum()
        if enc_class_col and enc_class_col in pid_enc.columns:
            recent = pid_enc[(pd.to_datetime(pid_enc[enc_date_col], errors='coerce') >= (last_date - pd.Timedelta(days=180))) & (pd.to_datetime(pid_enc[enc_date_col], errors='coerce') <= last_date)]
            record['ed_count_180d'] = recent[recent[enc_class_col].astype(str).str.contains('emerg', case=False, na=False)].shape[0]
            record['inpatient_count_180d'] = recent[recent[enc_class_col].astype(str).str.contains('inpatient', case=False, na=False)].shape[0]
        else:
            record['ed_count_180d'] = 0
            record['inpatient_count_180d'] = 0
    else:
        record['enc_count_180d'] = 0
        record['enc_count_365d'] = 0
        record['ed_count_180d'] = 0
        record['inpatient_count_180d'] = 0

    # medication coverage proxy
    med_cov_days = np.nan
    if 'start' in meds.columns and 'stop' in meds.columns:
        try:
            pid_meds = meds[meds['patient'] == pid]
            pid_meds = pid_meds.copy()
            pid_meds['start'] = pd.to_datetime(pid_meds['start'], errors='coerce')
            pid_meds['stop'] = pd.to_datetime(pid_meds['stop'], errors='coerce')
            covered_days = 0
            for _, r in pid_meds.iterrows():
                s = r['start']; e = r['stop']
                if pd.isna(s) or pd.isna(e):
                    continue
                s2 = max(s, window_start)
                e2 = min(e, last_date)
                if e2 >= s2:
                    covered_days += (e2 - s2).days + 1
            record['med_coverage_days_90d'] = covered_days
            record['med_coverage_frac_90d'] = covered_days / 90.0
        except Exception:
            record['med_coverage_days_90d'] = np.nan
            record['med_coverage_frac_90d'] = np.nan
    else:
        record['med_coverage_days_90d'] = np.nan
        record['med_coverage_frac_90d'] = np.nan

    rows.append(record)

# Build features DF
feats = pd.DataFrame(rows)

# Make sure last_date is datetime
feats['last_date'] = pd.to_datetime(feats['last_date'], errors='coerce')

# Now create labels: look forward 90 days in encounters for ED/inpatient to mark deterioration
feats['deterioration_90d'] = 0
enc[enc_date_col] = pd.to_datetime(enc[enc_date_col], errors='coerce')

for idx, row in feats.iterrows():
    pid = row['patient_id']
    last_date = row['last_date']
    if pd.isna(last_date):
        feats.at[idx, 'deterioration_90d'] = np.nan
        continue
    look_start = last_date + pd.Timedelta(days=1)
    look_end = last_date + pd.Timedelta(days=90)
    pid_enc = enc[enc[enc_patient_col] == pid]
    pid_enc_future = pid_enc[(pd.to_datetime(pid_enc[enc_date_col], errors='coerce') >= look_start) & (pd.to_datetime(pid_enc[enc_date_col], errors='coerce') <= look_end)]
    if not pid_enc_future.empty:
        # if encounter class exists, demand ER/inpatient; otherwise count any encounter as deterioration
        if enc_class_col and enc_class_col in pid_enc_future.columns:
            if pid_enc_future[enc_class_col].astype(str).str.contains('emerg', case=False, na=False).any() or pid_enc_future[enc_class_col].astype(str).str.contains('inpatient', case=False, na=False).any():
                feats.at[idx, 'deterioration_90d'] = 1
        else:
            feats.at[idx, 'deterioration_90d'] = 1
    else:
        feats.at[idx, 'deterioration_90d'] = 0

out_path = PROCESSED_DIR / "features_labels_last90d_per_patient.csv"
feats.to_csv(out_path, index=False)
print("Saved features+labels to:", out_path)
print("Rows:", feats.shape[0], "Positives:", int(feats['deterioration_90d'].sum()))
