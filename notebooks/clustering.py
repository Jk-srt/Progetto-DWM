"""Unsupervised clustering & outlier feature engineering.

Steps:
1. Load processed training data.
2. Select numeric features (exclude id, target).
3. Standardize, PCA (retain 90% variance) + save variance plot.
4. KMeans over candidate k values (4,6,8,10) -> highest silhouette.
5. Fit final KMeans on full PCA data -> add cluster_id.
6. IsolationForest for outlier flag (is_outlier 0/1).
7. Save enriched dataset as train_with_clusters.csv and support artifacts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import warnings

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILES = [
    DATA_DIR / "train_with_clusters.csv",  # allow re-run
    DATA_DIR / "train_clean_preprocessed.csv",
    DATA_DIR / "train_clean.csv",
]
RANDOM_STATE = 42


def load_base_df() -> pd.DataFrame:
    for p in CANDIDATE_FILES[1:]:  # skip train_with_clusters for base load
        if p.exists():
            print(f"[INFO] Loading base dataset: {p.name}")
            return pd.read_csv(p)
    raise FileNotFoundError("No base processed dataset found.")


def main():
    warnings.filterwarnings("ignore")
    df = load_base_df()
    assert 'sii' in df.columns, "Target 'sii' missing."

    # Numeric feature selection
    drop_cols = [c for c in ['sii', 'id'] if c in df.columns]
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.select_dtypes(include=[np.number]).columns])

    print(f"[INFO] Numeric features count: {num_df.shape[1]}")
    # Handle case of too few numeric features
    if num_df.shape[1] < 3:
        print("[WARN] Too few numeric features for clustering; abort.")
        return

    # Impute missing numeric values (median) then Standardize
    if num_df.isnull().any().any():
        num_df = num_df.fillna(num_df.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df.values)

    # PCA retain 90% variance (or cap components to avoid over-dimensionality)
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[INFO] PCA components kept: {X_pca.shape[1]} (variance={pca.explained_variance_ratio_.sum():.3f})")

    # Explained variance plot
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
    plt.axhline(0.9, color='red', linestyle='--', label='90%')
    plt.xlabel('Component')
    plt.ylabel('Cumulative Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'pca_cumulative_variance.png')
    plt.close()

    # Choose k by silhouette
    candidates = [4,6,8,10]
    best = None
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        print(f"[KMEANS] k={k} silhouette={sil:.4f}")
        if not best or sil > best[0]:
            best = (sil, k, km)
    best_sil, best_k, best_km = best
    print(f"[KMEANS] Selected k={best_k} silhouette={best_sil:.4f}")
    cluster_labels = best_km.fit_predict(X_pca)
    df['cluster_id'] = cluster_labels

    # IsolationForest for outliers (fit on PCA space or scaled original numeric)
    iso = IsolationForest(random_state=RANDOM_STATE, n_estimators=200, contamination='auto')
    iso.fit(X_pca)
    outlier_flag = (iso.predict(X_pca) == -1).astype(int)
    df['is_outlier'] = outlier_flag

    # Target distribution per cluster
    target_dist = (df.groupby('cluster_id')['sii'].value_counts(normalize=True)
                     .rename('ratio').reset_index())
    target_pivot = target_dist.pivot(index='cluster_id', columns='sii', values='ratio').fillna(0)
    target_pivot.to_csv(DATA_DIR / 'cluster_target_distribution.csv')

    # 2D projection for visualization (first two PCA components)
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='tab10', s=10)
        plt.title(f'PCA(2D) + KMeans k={best_k}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(DATA_DIR / 'pca_kmeans_clusters.png')
        plt.close()

    # Save enriched dataset
    out_file = DATA_DIR / 'train_with_clusters.csv'
    df.to_csv(out_file, index=False)
    print(f"[SAVE] Enriched dataset saved: {out_file}")
    print("Artifacts: pca_cumulative_variance.png, pca_kmeans_clusters.png, cluster_target_distribution.csv")

if __name__ == '__main__':
    main()
