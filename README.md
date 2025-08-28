# Progetto-DWM

Project: Prediction of problematic internet use (Kaggle CMI-PIP)

What this repo includes:
- EDA and preprocessing scripts under `notebooks/`
- Modelling pipeline with CV, hyperparameter tuning, feature importances, permutation importance, ROC/PR curves, submission export, and error analysis
- Processed data artifacts under `data/processed/`
 - Optional clustering / unsupervised enrichment (`notebooks/clustering.py`) producing `train_with_clusters.csv` with `cluster_id` and `is_outlier`

Quick start
1) Place the dataset CSV at `data/processed/train_clean.csv` or run your preprocessing to produce `train_clean_preprocessed.csv`.
2) (Optional) Run clustering enrichment: `python notebooks/clustering.py` (adds `cluster_id`, `is_outlier`).
3) Run the modelling pipeline (compares RandomForest and LightGBM, saves metrics and artifacts, creates Kaggle submission if `data/raw/test.csv` present):
	- From VS Code or shell: `python notebooks/modelling.py`
	- Key artifacts (in `data/processed/`):
	  * `feature_importances_*.csv/.png`
	  * `permutation_importance_*.csv/.png`
	  * `*_curves_roc.png`, `*_curves_pr.png`
	  * `metrics_*.json`
	  * `top_wrong_*.csv`, `top_correct_*.csv`
	  * `submission_*.csv` (if test present)

Notes
- The pipeline uses 5-fold Stratified CV and tunes models with RandomizedSearch.
- If LightGBM isn’t installed, the script will skip it and proceed with RandomForest.
- For imbalanced classes, we report macro-F1 and use class_weight='balanced'. Consider SMOTE for further improvements.
- Artifacts now include: metrics JSON, ROC/PR plots, permutation importance, and optional submission CSV.
- Permutation importance: in alcuni run può comparire l'errore "All arrays must be of the same length" (issue noto sporadico con ColumnTransformer + joblib). Il modello comunque genera le impurity importances. Workaround: rilanciare con FAST_MODE=1 (meno parallelismo) oppure impostare manualmente n_jobs=1 / ridurre le feature categoriali.

Next steps (optional)
- Add KNN/DecisionTree baselines with learning curves.
- Add clustering notebook (k-means/k-medoids/Hierarchical/DBSCAN) with intrinsic metrics.
- Add PR/ROC plots and probability calibration.