# scripts/make_model_ready.py
import pandas as pd
from pathlib import Path
pd.set_option('display.max_rows', 200)

PROJECT_DIR = Path(r"C:\Users\manim\OneDrive\Desktop\Mani_ma\Projects\Chronic_Diseases_prediction")
IN = PROJECT_DIR / "data" / "processed" / "processed_clean" / "features_labels_cleaned_for_model.csv"
OUT_DIR = IN.parent
OUT_STRICT = OUT_DIR / "features_labels_model_ready_strict.csv"
OUT_INCL = OUT_DIR / "features_labels_model_ready_inclusive.csv"
OBS_CSV = Path(r"C:\Users\manim\synthea\output\csv\observations.csv")

print("Loading:", IN)
df = pd.read_csv(IN, low_memory=False, parse_dates=['last_date'])
print("Initial shape:", df.shape)

# List initial positives
print("Initial positives:", int(df['deterioration_90d'].sum()), "of", len(df))

# Common columns to exclude from feature set
meta_cols = ['patient_id','last_date','deterioration_90d','has_90d_followup']

# 1) Drop features with >95% missing (i.e., keep those with >=5% non-null)
feat_cols = [c for c in df.columns if c not in meta_cols]
keep = [c for c in feat_cols if df[c].notna().mean() >= 0.05]
print("Features before:", len(feat_cols), " -> kept (>=5% non-null):", len(keep))

# 2) Drop constant columns among kept
df_keep = df[meta_cols + keep].copy()
const_cols = [c for c in keep if df_keep[c].nunique(dropna=True) <= 1]
if const_cols:
    print("Dropping constant cols:", len(const_cols))
    for c in const_cols[:10]: print("  -", c)
    df_keep.drop(columns=const_cols, inplace=True)
else:
    print("No constant cols to drop.")

# 3) Fill med coverage NaNs with 0 if present
for c in ['med_coverage_days_90d','med_coverage_frac_90d']:
    if c in df_keep.columns:
        df_keep[c] = df_keep[c].fillna(0)

# 4) Save inclusive (all patients)
df_keep.to_csv(OUT_INCL, index=False)
print("Saved inclusive model-ready CSV:", OUT_INCL, "shape:", df_keep.shape)
print("Inclusive positives:", int(df_keep['deterioration_90d'].sum()))

# 5) Save strict (only has_90d_followup == 1)
if 'has_90d_followup' in df_keep.columns:
    df_strict = df_keep[df_keep['has_90d_followup'] == 1].copy()
    df_strict.to_csv(OUT_STRICT, index=False)
    print("Saved strict model-ready CSV (has 90d followup):", OUT_STRICT, "shape:", df_strict.shape)
    print("Strict positives:", int(df_strict['deterioration_90d'].sum()))
else:
    print("Warning: 'has_90d_followup' not present; cannot create strict dataset.")

# 6) Show basic class balance and rows
print("\nInclusive class balance:\n", df_keep['deterioration_90d'].value_counts())
if 'has_90d_followup' in df_keep.columns:
    print("\nStrict class balance:\n", df_strict['deterioration_90d'].value_counts())

# 7) Quick top correlated features with label (absolute Pearson)
num_feats = [c for c in df_keep.columns if c not in meta_cols and df_keep[c].dtype.kind in 'fi']
corrs = df_keep[num_feats + ['deterioration_90d']].corr(method='pearson')['deterioration_90d'].abs().sort_values(ascending=False)
top_feats = corrs.head(20).index.tolist()
print("\nTop features by absolute Pearson correlation with label (top 20):")
print(corrs.head(20).to_string())

# 8) Map feature codes to descriptions (if observations.csv exists) for top features
if OBS_CSV.exists():
    try:
        obs = pd.read_csv(OBS_CSV, low_memory=False, usecols=['CODE','DESCRIPTION'])
        # normalize columns
        obs['CODE'] = obs['CODE'].astype(str).str.strip()
        mapping = obs.drop_duplicates('CODE').set_index('CODE')['DESCRIPTION'].to_dict()
        print("\nTop features mapped to DESCRIPTION (best effort):")
        for f in top_feats:
            # remove suffixes like _missing_frac/_last etc to match CODE
            base = f.split('_')[0]
            desc = mapping.get(base, "(no mapping found)")
            print(f"{f} -> {base} -> {desc}")
    except Exception as e:
        print("Could not load observations.csv mapping:", e)
else:
    print("\nobservations.csv not found at expected path; skipping mapping.")

print("\nDONE.")
