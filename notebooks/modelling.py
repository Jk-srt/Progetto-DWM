"""Modelling pipeline with CV, HPO, feature importance and error analysis.

Meets project requirements:
- Compare at least 2 methods: RandomForest (sklearn) and LightGBM (beyond sklearn)
- Hyperparameter tuning via CV
- Export feature importances and analyze most confident correct/wrong predictions
"""

from pathlib import Path
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

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
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = PROJECT_ROOT / "data/raw"

# Performance / debug flags via environment variables
FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"  # simplify searches & computations
SKIP_LGBM = os.environ.get("SKIP_LGBM", "0") == "1"  # skip LightGBM entirely
print(f"[CONFIG] FAST_MODE={FAST_MODE} SKIP_LGBM={SKIP_LGBM}")


def _load_processed_df() -> pd.DataFrame:
    """Load processed dataset, trying common filenames."""
    candidates = [
    DATA_DIR / "train_with_clusters.csv",
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


def _plot_roc_pr_curves(y_true, y_proba, classes_, out_prefix: Path):
    """Save multiclass ROC (one-vs-rest macro) and macro PR curve (average)."""
    try:
        # For multiclass: compute macro-average curve
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=classes_)
        # ROC
        fpr_list, tpr_list = [], []
        for i in range(len(classes_)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        # Simple macro average by interpolation over common grid
        all_fpr = np.unique(np.concatenate([f for f in fpr_list]))
        mean_tpr = np.zeros_like(all_fpr)
        for tpr, fpr in zip(tpr_list, fpr_list):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes_)
        plt.figure(figsize=(6,5))
        plt.plot(all_fpr, mean_tpr, label="Macro-average ROC")
        plt.plot([0,1],[0,1],'--',color='grey')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Macro ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + "_roc.png"))
        plt.close()
        # PR macro (take precision-recall for each class then average interpolated)
        recall_grid = np.linspace(0,1,200)
        prec_accum = np.zeros_like(recall_grid)
        for i in range(len(classes_)):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            prec_interp = np.interp(recall_grid, rec[::-1], prec[::-1])  # ensure increasing
            prec_accum += prec_interp
        prec_macro = prec_accum / len(classes_)
        plt.figure(figsize=(6,5))
        plt.plot(recall_grid, prec_macro, label="Macro-average PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Macro Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix.with_name(out_prefix.name + "_pr.png"))
        plt.close()
    except Exception as e:
        print("[WARN] ROC/PR plot skipped:", e)


def _export_submission(best_model, X_full, y_full, id_full, best_name: str):
    """Attempt to load test set and create submission.csv (id, sii)."""
    # Kaggle test file likely in raw dir as test.csv
    test_candidates = [
        RAW_DIR / "test.csv",
        RAW_DIR / "test_clean.csv",
        DATA_DIR / "test_clean.csv",
    ]
    test_df = None
    for p in test_candidates:
        if p.exists():
            print(f"[SUB] Found test file: {p}")
            test_df = pd.read_csv(p)
            break
    if test_df is None:
        print("[SUB] Nessun file test trovato, salto export submission.")
        return
    # Keep id
    if 'id' not in test_df.columns:
        print("[SUB] Colonna 'id' mancante nel test, impossibile creare submission.")
        return
    X_test_pred = test_df.drop(columns=[c for c in ['id','sii'] if c in test_df.columns])
    # Refit on full training set first for maximal performance
    print("[SUB] Re-fitting best model on ENTIRE training set prima della submission...")
    best_model.fit(X_full, y_full)
    preds = best_model.predict(X_test_pred)
    sub = pd.DataFrame({
        'id': test_df['id'],
        'sii': preds,
    })
    out_path = DATA_DIR / f"submission_{best_name}.csv"
    sub.to_csv(out_path, index=False)
    print(f"[SUB] Submission salvata in {out_path}")


def main():
    warnings.filterwarnings("ignore")
    df = _load_processed_df()
    # Leakage safeguard: enriched columns created outside CV can cause unrealistically perfect scores.
    enriched_cols = [c for c in ['cluster_id', 'is_outlier'] if c in df.columns]
    if enriched_cols:
        if os.environ.get('KEEP_ENRICHED','0') == '1':
            print(f"[WARN] Keeping enriched columns (possible leakage): {enriched_cols}")
        else:
            print(f"[INFO] Dropping enriched columns to avoid leakage: {enriched_cols} (set KEEP_ENRICHED=1 to keep)")
            df = df.drop(columns=enriched_cols)

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
    rf_param_dist = {
        "clf__n_estimators": [200, 400, 800] if not FAST_MODE else [200, 400],
        "clf__max_depth": [None, 10, 20, 40] if not FAST_MODE else [None, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=20 if not FAST_MODE else 6,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1 if not FAST_MODE else 1,
        random_state=42,
        verbose=1 if FAST_MODE else 0,
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
        if SKIP_LGBM:
            raise RuntimeError("SKIP_LGBM flag attivo")
        from lightgbm import LGBMClassifier

        lgbm_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", LGBMClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1 if not FAST_MODE else 1,
                verbose=-1,
            )),
        ])
        lgbm_param_dist = {
            "clf__n_estimators": [300, 600, 1000] if not FAST_MODE else [400, 800],
            "clf__learning_rate": [0.05, 0.1, 0.2] if not FAST_MODE else [0.1, 0.2],
            "clf__num_leaves": [31, 63, 127] if not FAST_MODE else [31, 63],
            "clf__max_depth": [-1, 6, 12],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__reg_lambda": [0.0, 0.1, 1.0],
        }
        lgbm_search = RandomizedSearchCV(
            lgbm_pipe,
            param_distributions=lgbm_param_dist,
            n_iter=25 if not FAST_MODE else 8,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1 if not FAST_MODE else 1,
            random_state=42,
            verbose=1 if FAST_MODE else 0,
        )
        lgbm_search.fit(X, y)
        lgbm_best = lgbm_search.best_estimator_
        print("[LGBM] Best params:", lgbm_search.best_params_)
        lgbm_cv = _evaluate_cv(lgbm_best, X, y, cv)
        print("[LGBM] CV:", lgbm_cv)
    except Exception as e:
        print("[LGBM] LightGBM non disponibile o saltato:", e)

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

    # ROC / PR plots (macro average)
    if y_proba is not None:
        _plot_roc_pr_curves(y_test.values, y_proba, best_model.classes_, DATA_DIR / f"{best_name}_curves")
        print(f"[BEST] Salvati ROC/PR: {best_name}_curves_roc.png / _pr.png")

    # Permutation importance (computationally expensive - limit n_repeats)
    try:
        repeats = 3 if FAST_MODE else 5
        print(f"[PERM] Calcolo permutation importance (n_repeats={repeats})...")
        X_test_copy = X_test.copy()
        perm = permutation_importance(best_model, X_test_copy, y_test, n_repeats=repeats, random_state=42, n_jobs=-1 if not FAST_MODE else 1)
        # Need feature names again
        prep = best_model.named_steps.get("prep")
        feat_names = _get_feature_names(prep, num_cols, cat_cols)
        perm_df = pd.DataFrame({
            'feature': feat_names,
            'perm_mean': perm.importances_mean,
            'perm_std': perm.importances_std,
        }).sort_values('perm_mean', ascending=False)
        perm_df.to_csv(DATA_DIR / f"permutation_importance_{best_name}.csv", index=False)
        _plot_feature_importances(perm_df['perm_mean'].values, perm_df['feature'].values,
                                  f"Permutation Importance Top20 - {best_name}",
                                  DATA_DIR / f"permutation_importance_{best_name}.png")
        print(f"[PERM] Salvati permutation importance CSV/PNG")
    except Exception as e:
        print("[PERM] Skipped:", e)
        try:
            print(f"[PERM-DEBUG] X_test shape={X_test.shape} y_test shape={y_test.shape}")
            prep = best_model.named_steps.get("prep")
            feat_names = _get_feature_names(prep, num_cols, cat_cols)
            print(f"[PERM-DEBUG] Expected feature count {len(feat_names)}")
        except Exception as e2:
            print("[PERM-DEBUG] Secondary failure:", e2)

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

    # Salva metriche aggregate (CV + holdout)
    metrics = {
        'model_selected': best_name,
        'cv_metrics': best_cv,
        'holdout': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        }
    }
    with open(DATA_DIR / f"metrics_{best_name}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[BEST] Metriche salvate metrics_{best_name}.json")

    # Export submission se test disponibile
    _export_submission(best_model, X, y, id_series, best_name)


if __name__ == "__main__":
    main()
