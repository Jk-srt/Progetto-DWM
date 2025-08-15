"""Modelling pipeline with CV, HPO, feature importance and error analysis.

Meets project requirements:
- Compare at least 2 methods: RandomForest (sklearn) and LightGBM (beyond sklearn)
- Hyperparameter tuning via CV
- Export feature importances and analyze most confident correct/wrong predictions
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV,
    cross_validate,
    cross_val_predict,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_processed_df() -> pd.DataFrame:
    """Load processed dataset, trying common filenames."""
    candidates = [
        DATA_DIR / "train_clean.csv",
        DATA_DIR / "train_clean_preprocessed.csv",
    ]
    for p in candidates:
        if p.exists():
            print(f"[INFO] Loading: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"None of processed files found: {[str(p) for p in candidates]}"
    )


def _get_feature_names(preprocessor: ColumnTransformer, num_cols, cat_cols):
    """Resolve output feature names from ColumnTransformer + OneHot."""
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        names = []
        # numeric passthrough names
        names.extend([f"num__{c}" for c in num_cols])
        # cat onehot names (approximate if encoder present)
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for c, cats in zip(cat_cols, ohe.categories_):
                names.extend([f"cat__{c}__{str(k)}" for k in cats])
        except Exception:
            names.extend([f"cat__{c}" for c in cat_cols])
        return np.array(names)


def _evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    """Return CV metrics dict and prob-based metrics via CV predictions."""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    # prob-based metrics
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    # Handle multiclass safely
    try:
        roc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc = np.nan
    try:
        pr_auc = average_precision_score(y, y_proba, average="macro")
    except Exception:
        pr_auc = np.nan

    summary = {
        "accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
        "accuracy_std": float(np.std(cv_res["test_accuracy"])),
        "f1_macro_mean": float(np.mean(cv_res["test_f1_macro"])),
        "f1_macro_std": float(np.std(cv_res["test_f1_macro"])),
        "roc_auc_ovr_weighted": float(roc) if np.isfinite(roc) else None,
        "average_precision_macro": float(pr_auc) if np.isfinite(pr_auc) else None,
    }
    return summary


def _plot_feature_importances(importances, names, title, out_png):
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(order)), np.array(importances)[order][::-1])
    plt.yticks(range(len(order)), np.array(names)[order][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    warnings.filterwarnings("ignore")
    df = _load_processed_df()

    # Target e ID
    assert "sii" in df.columns, "Colonna target 'sii' mancante nel dataset."
    id_series = df["id"] if "id" in df.columns else pd.Series(df.index, name="id")

    # Features/target
    X = df.drop(columns=[c for c in ["sii", "id"] if c in df.columns])
    y = df["sii"]

    # Tipi di colonne
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numeriche: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # Preprocessori
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1) RandomForest (sklearn) + HPO
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions={
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        },
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rf_search.fit(X, y)
    rf_best = rf_search.best_estimator_
    print("[RF] Best params:", rf_search.best_params_)
    rf_cv = _evaluate_cv(rf_best, X, y, cv)
    print("[RF] CV:", rf_cv)

    # 2) LightGBM (beyond scikit-learn) + HPO
    lgbm_best = None
    lgbm_cv = None
    try:
        from lightgbm import LGBMClassifier

        lgbm_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", LGBMClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        lgbm_search = RandomizedSearchCV(
            lgbm_pipe,
            param_distributions={
                "clf__n_estimators": [300, 600, 1000],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__num_leaves": [31, 63, 127],
                "clf__max_depth": [-1, 6, 12],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
                "clf__reg_lambda": [0.0, 0.1, 1.0],
            },
            n_iter=25,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        lgbm_search.fit(X, y)
        lgbm_best = lgbm_search.best_estimator_
        print("[LGBM] Best params:", lgbm_search.best_params_)
        lgbm_cv = _evaluate_cv(lgbm_best, X, y, cv)
        print("[LGBM] CV:", lgbm_cv)
    except Exception as e:
        print("[LGBM] LightGBM non disponibile:", e)

    # Scegli best model per analisi holdout
    def score_from(cv_dict):
        return -1 if cv_dict is None else cv_dict.get("f1_macro_mean", -1)

    best_model, best_name, best_cv = (rf_best, "RandomForest", rf_cv)
    if lgbm_best is not None and score_from(lgbm_cv) > score_from(rf_cv):
        best_model, best_name, best_cv = (lgbm_best, "LightGBM", lgbm_cv)
    print(f"[BEST] {best_name} selected for holdout analysis")

    # Holdout per error analysis e report leggibile
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, id_series,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)
    except Exception:
        pass

    # Probabilità per error analysis
    y_proba = None
    try:
        y_proba = best_model.predict_proba(X_test)
    except Exception:
        pass
    if y_proba is not None:
        conf = np.max(y_proba, axis=1)
        pred_labels = np.argmax(y_proba, axis=1)
        # Mappa in class labels veri (nel caso non siano 0..K-1)
        classes_ = best_model.classes_ if hasattr(best_model, "classes_") else np.unique(y)
        pred_mapped = np.array([classes_[i] for i in pred_labels])
        analysis_df = pd.DataFrame({
            "id": id_test.values,
            "y_true": y_test.values,
            "y_pred": pred_mapped,
            "confidence": conf,
        })
        analysis_df["correct"] = analysis_df["y_true"] == analysis_df["y_pred"]
        top_wrong = analysis_df.loc[~analysis_df["correct"]].sort_values("confidence", ascending=False).head(30)
        top_right = analysis_df.loc[analysis_df["correct"]].sort_values("confidence", ascending=False).head(30)
        top_wrong.to_csv(DATA_DIR / f"top_wrong_{best_name}.csv", index=False)
        top_right.to_csv(DATA_DIR / f"top_correct_{best_name}.csv", index=False)
        print(f"[BEST] Salvati: top_wrong_{best_name}.csv, top_correct_{best_name}.csv")

    # Feature importance (se disponibile)
    try:
        # Estrai nomi colonne post-preprocessing
        prep = best_model.named_steps.get("prep")
        feat_names = _get_feature_names(prep, num_cols, cat_cols)
        clf = best_model.named_steps.get("clf")
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df.sort_values("importance", ascending=False).to_csv(
                DATA_DIR / f"feature_importances_{best_name}.csv", index=False
            )
            _plot_feature_importances(
                importances, feat_names,
                title=f"Top 20 Feature Importances - {best_name}",
                out_png=DATA_DIR / f"feature_importances_{best_name}.png",
            )
            print(f"[BEST] Salvate importanze: feature_importances_{best_name}.csv/.png")
    except Exception as e:
        print("[WARN] Feature importance non disponibile:", e)


if __name__ == "__main__":
    main()
