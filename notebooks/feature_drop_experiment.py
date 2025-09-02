"""Compare feature dropping strategies to ensure we are not over-removing features.

Strategies:
  D_all  : No feature dropping (tutte le feature originali dopo rimozione col target/id)
  A_full : Current manual strategy (drop all BIA-*, PCIAT_Total, PCIAT_* items, PCIAT-Season)
  B_min  : Minimal strategy: drop only features missing in test OR purity >= PURE_THRESHOLD (only for BIA / exact totals)
  C_high : Stricter high-purity (purity >= STRICT_THRESHOLD) only, plus those missing in test

Models:
  - RandomForest (baseline)
  - LightGBM (se disponibile / non disabilitato)

Outputs:
  data/processed/feature_drop_comparison.csv (righe: strategy + model)
  data/processed/top_purity_features.csv (top 50 purity)

Run:
  python notebooks/feature_drop_experiment.py
Optional env vars:
  FAST_EXP=1           -> riduce n_estimators
  PURE_THRESHOLD=0.995
  STRICT_THRESHOLD=0.998
  SKIP_LGBM_EXP=1      -> salta LightGBM
"""
from __future__ import annotations
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed'
RAW_DIR = PROJECT_ROOT / 'data/raw'
DATA_DIR.mkdir(exist_ok=True, parents=True)

FAST = os.environ.get('FAST_EXP','0')=='1'
PURE_THRESHOLD = float(os.environ.get('PURE_THRESHOLD','0.995'))
STRICT_THRESHOLD = float(os.environ.get('STRICT_THRESHOLD','0.998'))
SKIP_LGBM_EXP = os.environ.get('SKIP_LGBM_EXP','0')=='1'

TRAIN_CANDIDATES = [DATA_DIR/ 'train_with_clusters.csv', DATA_DIR/'train_clean.csv', DATA_DIR/'train_clean_preprocessed.csv']
for c in TRAIN_CANDIDATES:
    if c.exists():
        train_path = c
        break
else:
    raise FileNotFoundError('No processed training file found.')

df = pd.read_csv(train_path)
assert 'sii' in df.columns, "Target 'sii' not present"

# Remove enrichment leakage columns for fairness
for col in ['cluster_id','is_outlier']:
    if col in df.columns:
        df = df.drop(columns=col)

y = df['sii']
X_full = df.drop(columns=[c for c in ['sii','id'] if c in df.columns])

# Test columns (schema alignment)
test_cols = None
for p in [RAW_DIR/'test.csv', RAW_DIR/'test_clean.csv', DATA_DIR/'test_clean.csv']:
    if p.exists():
        try:
            test_cols = pd.read_csv(p, nrows=5).columns.tolist()
            break
        except Exception:
            pass

# Purity calculation (solo una volta)
purity_scores = {}
Y_np = y.values
for col in X_full.columns:
    try:
        grp = pd.DataFrame({'feat': X_full[col], 'y': Y_np})
        purity_vals = grp.groupby('feat')['y'].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        purity_scores[col] = float(purity_vals.mean())
    except Exception:
        purity_scores[col] = np.nan

# Groups
bia_cols = [c for c in X_full.columns if c.startswith('BIA-BIA_')]
pciat_items = [c for c in X_full.columns if c.startswith('PCIAT-PCIAT_')]
manual_exact = ['PCIAT-PCIAT_Total','PCIAT-Season']

# Strategy builders
def strat_D_all():
    return X_full.copy(), 'D_all'

def strat_A_full():
    drop = set(bia_cols + pciat_items + manual_exact)
    return X_full.drop(columns=[c for c in drop if c in X_full.columns], errors='ignore'), 'A_full'

def strat_B_min():
    drop = set()
    for c in X_full.columns:
        if test_cols is not None and c not in test_cols:
            drop.add(c)  # missing in test
        elif purity_scores.get(c,0) >= PURE_THRESHOLD and (c.startswith('BIA-BIA_') or c in manual_exact):
            drop.add(c)  # high-purity subset (BIA or exact total)
    return X_full.drop(columns=list(drop), errors='ignore'), 'B_min'

def strat_C_high():
    drop = set()
    for c in X_full.columns:
        if test_cols is not None and c not in test_cols:
            drop.add(c)
        elif purity_scores.get(c,0) >= STRICT_THRESHOLD:
            drop.add(c)
    return X_full.drop(columns=list(drop), errors='ignore'), 'C_high'

strategies = [strat_D_all(), strat_A_full(), strat_B_min(), strat_C_high()]

# Try LightGBM import
lgbm_available = False
if not SKIP_LGBM_EXP:
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        lgbm_available = True
    except Exception:
        print('[INFO] LightGBM non disponibile: salto test LightGBM (installare lightgbm per includerlo)')
else:
    print('[INFO] SKIP_LGBM_EXP=1 -> salto LightGBM')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

def build_preprocessor(X_variant: pd.DataFrame):
    num_cols = X_variant.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_variant.select_dtypes(include=['object','category']).columns.tolist()
    preproc = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', SkPipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    return preproc

# Helper for evaluation
from typing import Dict

def eval_model(model_name: str, pipe: Pipeline, X_variant: pd.DataFrame, strategy_name: str) -> Dict:
    scoring = {'accuracy':'accuracy','f1_macro':'f1_macro'}
    cv_res = cross_validate(pipe, X_variant, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    return {
        'strategy': strategy_name,
        'model': model_name,
        'n_features': X_variant.shape[1],
        'dropped_features': len(X_full.columns)-X_variant.shape[1],
        'accuracy_mean': float(np.mean(cv_res['test_accuracy'])),
        'accuracy_std': float(np.std(cv_res['test_accuracy'])),
        'f1_macro_mean': float(np.mean(cv_res['test_f1_macro'])),
        'f1_macro_std': float(np.std(cv_res['test_f1_macro'])),
    }

for X_variant, strat_name in strategies:
    preproc = build_preprocessor(X_variant)
    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=400 if not FAST else 160,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_pipe = Pipeline([
        ('prep', preproc),
        ('clf', rf)
    ])
    results.append(eval_model('RandomForest', rf_pipe, X_variant, strat_name))

    # LightGBM
    if lgbm_available:
        lgbm = LGBMClassifier(
            n_estimators=600 if not FAST else 300,
            learning_rate=0.1,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            max_depth=-1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_pipe = Pipeline([
            ('prep', preproc),
            ('clf', lgbm)
        ])
        results.append(eval_model('LightGBM', lgbm_pipe, X_variant, strat_name))

res_df = pd.DataFrame(results).sort_values(['strategy','model'])
out_path = DATA_DIR / 'feature_drop_comparison.csv'
res_df.to_csv(out_path, index=False)

print('\n=== Feature Drop Strategy Comparison (per model) ===')
print(res_df.to_string(index=False))
print(f'\nSaved: {out_path}')

# Purity top 50
purity_series = pd.Series(purity_scores).sort_values(ascending=False).head(50)
purity_series.to_csv(DATA_DIR / 'top_purity_features.csv', header=['purity'])
print('Saved: top_purity_features.csv (top 50)')
