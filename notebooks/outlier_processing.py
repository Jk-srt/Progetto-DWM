"""Outlier analysis and capping for selected numeric features.

Steps:
1. Load processed training dataset.
2. Select top K numeric features (default 8) using mutual information with target 'sii'.
   - Criterion documented: mutual_info_classif on median-imputed numeric matrix.
3. For each selected feature compute Q1, Q3, IQR, bounds = Q1 - 1.5*IQR, Q3 + 1.5*IQR, count outliers.
4. Apply capping (winsorizing) to bring values within bounds (chosen over dropping to avoid data loss).
5. Produce summary CSV with before/after outlier counts.
6. Save capped dataset variant.

Outputs:
    data/processed/outlier_summary.csv
    data/processed/train_clean_outliers_capped.csv
    data/processed/train_clean_cleaned.csv (alias richiesto da TODO)
    data/processed/outlier_stats.json (statistiche righe con almeno un valore cap)

Env vars:
  TOPK_OUTLIER_NUM=8  (change number of numeric features to analyze)
  TRAIN_FILE_OVERRIDE=filename.csv (optional override)

Run:
  python notebooks/outlier_processing.py
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed'
DATA_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_CANDIDATES = [
    os.environ.get('TRAIN_FILE_OVERRIDE', ''),
    DATA_DIR / 'train_with_clusters.csv',
    DATA_DIR / 'train_clean.csv',
    DATA_DIR / 'train_clean_preprocessed.csv',
]

train_path = None
for c in TRAIN_CANDIDATES:
    if not c:
        continue
    p = Path(c)
    if p.exists():
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError('No training file found in processed directory.')

print(f'[OUTLIER] Loading {train_path}')
df = pd.read_csv(train_path)
assert 'sii' in df.columns, "Target 'sii' missing"

# Remove known leakage enrichment if present (treat as modelling does)
for col in ['cluster_id','is_outlier']:
    if col in df.columns:
        df = df.drop(columns=col)

# Identify numeric columns (exclude id & target)
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['sii','id']]
print(f'[OUTLIER] Numeric columns total: {len(numeric_cols)}')

if len(numeric_cols) == 0:
    raise RuntimeError('No numeric columns available.')

TOPK = int(os.environ.get('TOPK_OUTLIER_NUM', '8'))

# Mutual information selection
num_df = df[numeric_cols].copy()
# Median impute temporarily for MI
impute_values = num_df.median()
num_df_imputed = num_df.fillna(impute_values)
mi = mutual_info_classif(num_df_imputed, df['sii'], discrete_features=False, random_state=42)
mi_series = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
selected = mi_series.head(TOPK).index.tolist()
print(f'[OUTLIER] Selected top {TOPK} numeric features by mutual information: {selected}')

records = []
# Copy for capping
capped_df = df.copy()
changed_masks = []  # store per-feature masks of changed values
for feat in selected:
    col = df[feat]
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # Outliers before
    mask_low = col < lower
    mask_high = col > upper
    n_low = int(mask_low.sum())
    n_high = int(mask_high.sum())
    n_out = n_low + n_high
    # Apply capping
    capped_col = col.clip(lower, upper)
    changed_masks.append(capped_col != col)
    capped_df[feat] = capped_col
    # After counts
    n_out_after = int(((capped_col < lower) | (capped_col > upper)).sum())
    records.append({
        'feature': feat,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'lower_bound': lower,
        'upper_bound': upper,
        'outliers_before': n_out,
        'outliers_low': n_low,
        'outliers_high': n_high,
        'percent_outliers_before': n_out / len(df) * 100.0,
        'outliers_after': n_out_after,
        'strategy': 'capping'
    })

summary_df = pd.DataFrame(records)
summary_path = DATA_DIR / 'outlier_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f'[OUTLIER] Saved summary -> {summary_path}')

# Save capped dataset variant + alias + stats
changed_df = pd.concat(changed_masks, axis=1) if changed_masks else pd.DataFrame(index=df.index)
rows_affected = int(changed_df.any(axis=1).sum()) if not changed_df.empty else 0
percent_rows_affected = rows_affected / len(df) * 100.0

capped_path = DATA_DIR / 'train_clean_outliers_capped.csv'
capped_df.to_csv(capped_path, index=False)
clean_alias_path = DATA_DIR / 'train_clean_cleaned.csv'
capped_df.to_csv(clean_alias_path, index=False)
print(f'[OUTLIER] Saved capped dataset -> {capped_path} (alias {clean_alias_path})')

stats = {
    'n_rows': int(len(df)),
    'rows_with_any_capped': rows_affected,
    'percent_rows_with_any_capped': percent_rows_affected,
    'selected_features': selected,
    'note': 'No rows removed; values outside IQR bounds clipped.'
}
import json
with open(DATA_DIR / 'outlier_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)
print(f"[OUTLIER] Stats: rows_with_any_capped={rows_affected} ({percent_rows_affected:.2f}%) -> outlier_stats.json")

# Provide plain text note for report snippet
txt_note = DATA_DIR / 'outlier_note.txt'
with open(txt_note, 'w', encoding='utf-8') as f:
    f.write('Outlier handling performed via IQR capping on features: ' + ', '.join(selected) + '\n')
    total_out_before = int(sum(summary_df['outliers_before']))
    f.write(f'Total outlier instances across selected features (counting duplicates per feature) before capping: {total_out_before}\n')
    f.write(f'Rows with at least one capped value: {rows_affected} ({percent_rows_affected:.2f}% of dataset)\n')
    f.write('Chosen strategy: capping (winsorizing) to retain all rows and reduce influence of extreme values.')
print(f'[OUTLIER] Wrote note -> {txt_note}')
