"""Baseline models script.

Purpose:
- Provide simple, transparent baseline performances (DecisionTree, RandomForest quick, KNN) before advanced tuned models.
- Reuse same preprocessing (impute + one-hot) to ensure fair comparison and avoid leakage.
- Save metrics & confusion matrices for report.

Run (fast):
    FAST_BASELINES=1 python notebooks/baselines.py

Artifacts saved into data/processed/:
    baseline_metrics_<model>.json
    confusion_matrix_<model>.csv

Note: TARGET_COL must match your target label column (currently 'sii').
"""
from __future__ import annotations
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "sii"  # adjust if different
FAST = os.environ.get("FAST_BASELINES", "0") == "1"
print(f"[BASELINES] FAST_MODE={FAST}")

def load_df():
    candidates = [
        DATA_DIR / "train_clean.csv",
        DATA_DIR / "train_clean_preprocessed.csv",
        DATA_DIR / "train_with_clusters.csv",
    ]
    for c in candidates:
        if c.exists():
            print(f"[LOAD] {c}")
            return pd.read_csv(c)
    raise FileNotFoundError("Nessun dataset processato trovato.")

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])
    return pre

def eval_and_save(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    # Save
    with open(DATA_DIR / f"baseline_metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "f1_macro": f1m, "report": report}, f, indent=2)
    pd.DataFrame(cm).to_csv(DATA_DIR / f"confusion_matrix_{name}.csv", index=False)
    print(f"[RESULT] {name}: acc={acc:.4f} f1_macro={f1m:.4f}")


def main():
    df = load_df()
    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' non trovato"
    # Drop enriched columns for baseline fairness
    drop_cols = [c for c in ["cluster_id", "is_outlier"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    X = df.drop(columns=[c for c in [TARGET_COL, "id"] if c in df.columns])
    y = df[TARGET_COL]
    pre = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = []
    # Decision Tree (baseline very simple)
    dt_params = dict(max_depth=8 if FAST else None, random_state=42, class_weight="balanced")
    models.append(("DecisionTree", DecisionTreeClassifier(**dt_params)))
    # Small RandomForest baseline (different from tuned one)
    rf_params = dict(n_estimators=150 if FAST else 300, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced")
    models.append(("RandomForestBase", RandomForestClassifier(**rf_params)))
    # KNN (sensitive to scaling but categorical handled via one-hot)
    knn_params = dict(n_neighbors=3 if FAST else 7, weights="distance")
    models.append(("KNN", KNeighborsClassifier(**knn_params)))

    for name, clf in models:
        pipe = Pipeline([
            ("prep", pre),
            ("model", clf),
        ])
        eval_and_save(name, pipe, X_train, X_test, y_train, y_test)

    print("[BASELINES] Completato. Artefatti salvati in data/processed.")

if __name__ == "__main__":
    main()
