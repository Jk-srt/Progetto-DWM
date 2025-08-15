# Progetto-DWM

Project: Prediction of problematic internet use (Kaggle CMI-PIP)

What this repo includes:
- EDA and preprocessing scripts under `notebooks/`
- Modelling pipeline with CV, hyperparameter tuning, feature importances, and error analysis
- Processed data artifacts under `data/processed/`

Quick start
1) Place the dataset CSV at `data/processed/train_clean.csv` or run your preprocessing to produce `train_clean_preprocessed.csv`.
2) Run the modelling pipeline (compares RandomForest and LightGBM, saves metrics and artifacts):
	- From VS Code: run `notebooks/modelling.py`
	- Artifacts: `feature_importances_*.csv/.png`, `top_wrong_*.csv`, `top_correct_*.csv` in `data/processed/`

Notes
- The pipeline uses 5-fold Stratified CV and tunes models with RandomizedSearch.
- If LightGBM isn’t installed, the script will skip it and proceed with RandomForest.
- For imbalanced classes, we report macro-F1 and use class_weight='balanced'. Consider SMOTE for further improvements.

Next steps (optional)
- Add KNN/DecisionTree baselines with learning curves.
- Add clustering notebook (k-means/k-medoids/Hierarchical/DBSCAN) with intrinsic metrics.
- Add PR/ROC plots and probability calibration.