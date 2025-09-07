# train_and_explain.py
import os
import shap
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


def load_dataset(file_path):
    """
    Load dataset and automatically detect label column.
    Returns X (features), y (labels).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    # Candidate label column names
    label_candidates = ["deterioration_90d", "label", "target", "y"]
    label_col = None
    for cand in label_candidates:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        raise ValueError(f"No label column found! Columns: {df.columns.tolist()}")

    # Features = everything except IDs, dates, and label
    drop_cols = [c for c in ["patient_id", "last_date", label_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[label_col]

    return X, y, label_col


def train_and_evaluate(X, y, dataset_name, output_dir="models"):
    """
    Train RandomForest, evaluate, save model and SHAP plots.
    """
    print(f"\n📌 Training on {dataset_name} dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {dataset_name} Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"{dataset_name} ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"rf_{dataset_name}.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Saved model: {model_path}")

    # SHAP explainability (safe handling of outputs)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle both possible SHAP output formats
        if isinstance(shap_values, list):  # binary classification => list of arrays
            shap_values_to_plot = shap_values[1]  # take class=1 contributions
        else:
            shap_values_to_plot = shap_values

        plt.figure()
        shap.summary_plot(shap_values_to_plot, X_test, show=False)
        plt.title(f"SHAP Summary - {dataset_name}")
        shap_plot_path = os.path.join(output_dir, f"shap_summary_{dataset_name}.png")
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        print(f"📊 Saved SHAP summary plot: {shap_plot_path}")

    except Exception as e:
        print(f"⚠️ Could not generate SHAP plots for {dataset_name}: {e}")


def main():
    datasets = {
        "strict": r"C:\Users\manim\OneDrive\Desktop\Mani_ma\Projects\Chronic_Diseases_prediction\data\processed\processed_clean\features_labels_model_ready_strict.csv",
        "inclusive": r"C:\Users\manim\OneDrive\Desktop\Mani_ma\Projects\Chronic_Diseases_prediction\data\processed\processed_clean\features_labels_model_ready_inclusive.csv",
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                X, y, label_col = load_dataset(path)
                print(f"✔ Loaded {name} dataset with label column: {label_col}")
                train_and_evaluate(X, y, dataset_name=name)
            except Exception as e:
                print(f"❌ Error processing {name} dataset: {e}")
        else:
            print(f"⚠ {name} dataset not found at {path}")


if __name__ == "__main__":
    main()
