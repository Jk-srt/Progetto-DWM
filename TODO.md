# Progetto DWM – To‑Do List

Formato rapido:
- [ ] (P1) Task ad alta priorità – aggiungere a mano chi lo prende (es: "(G)").
- Aggiungere data completamento tra parentesi: (done 2025-08-28).

Legenda priorità:
P1 = Critico per la relazione / valutazione
P2 = Migliora qualità e copertura syllabus
P3 = Nice-to-have / opzionale

---
## P1 – Critici / Prima della relazione

### 1. Kaggle Submission
- [x] (P1) Aggiungere `data/raw/test.csv` (scaricare da Kaggle)
- [ ] (P1) Eseguire `python notebooks/modelling.py` (run completo non FAST)
- [ ] (P1) Caricare submission su Kaggle e annotare score in `data/processed/submission_log.csv` (timestamp, model, public_score)
- [ ] (P1) Verificare riproducibilità (seed, FAST_MODE=0) nel log

### 2. Outlier / Valori irrealistici
- [ ] (P1) Selezionare 5–10 feature numeriche principali (documentare criterio)
- [ ] (P1) Calcolare IQR bounds e contare outlier (prima/dopo)
- [ ] (P1) Decidere strategia (drop o capping) e applicarla
- [ ] (P1) Salvare `data/processed/outlier_summary.csv`
- [ ] (P1) Salvare dataset pulito come `data/processed/train_clean_cleaned.csv` (se migliora)
- [ ] (P1) Annotare nel report % righe rimosse / capped

### 3. Baselines & Confronto Modelli
- [ ] (P1) Eseguire `python notebooks/baselines.py` e generare metriche baseline
- [ ] (P1) Estrarre metriche RF & LGBM tunati (da `metrics_*.json`)
- [ ] (P1) Creare tabella comparativa (baseline_DT, baseline_RFbase, baseline_KNN, RF_tuned, LGBM_tuned)
- [ ] (P1) Aggiungere tabella al report (sezione Risultati)

### 4. Error Analysis Specifico (richiesta prof)
- [ ] (P1) Identificare top feature da `feature_importances_*.csv`
- [ ] (P1) Costruire dataset holdout con flag corretto/errato (già in modelling)
- [ ] (P1) Generare plot distribuzione valore feature vs count corrette & errate (`error_feature_<name>.png`)
- [ ] (P1) Scrivere breve testo interpretazione (`report_snippets/error_feature_<name>.txt`)

### 5. Hyperparameter Tuning Documentazione
- [ ] (P1) Copiare best params RF & LGBM in `data/processed/hpo_summary.json`
- [ ] (P1) Elencare range provati (RF: n_estimators, max_depth, ecc.; LGBM: learning_rate, num_leaves...) nel report

### 6. Report (REPORT_TEMPLATE.md)
- [ ] (P1) Sezione Data Processing (drop NA, imputazioni, encoding, outlier)
- [ ] (P1) Sezione Modelli & Tuning (metodologia CV + param search)
- [ ] (P1) Sezione Risultati (tabella + commento differenze)
- [ ] (P1) Sezione Error Analysis (top_wrong/top_correct + feature plot)
- [ ] (P1) Sezione Limiti & Future Work (SMOTE, stacking, calibration, interpretabilità avanzata)
- [ ] (P1) Revisione finale (tono narrativo, niente codice)

---
## P2 – Miglioramenti (dopo P1)

### Imbalance & Threshold
- [ ] (P2) Script `notebooks/imbalance_experiments.py` con Pipeline (SMOTE -> RF)
- [ ] (P2) Confronto macro-F1 prima/dopo SMOTE (`smote_comparison.csv`)
- [ ] (P2) Curva precision-recall e scelta threshold ottimale (`threshold_optimization.png`)

### Modelli Addizionali
- [ ] (P2) `extended_models.py` (LogisticRegression, SVM lineare, SVM RBF, ExtraTrees)
- [ ] (P2) Validation curve per C/logreg & C/SVM (`validation_curve_logreg.png`, `validation_curve_svm.png`)
- [ ] (P2) Metriche in JSON per ciascun modello

### Calibration & Stacking
- [ ] (P2) `calibration_stacking.py` (CalibratedClassifierCV su best + stacking RF+LGBM)
- [ ] (P2) Reliability diagram (`calibration_curve.png`)
- [ ] (P2) Metriche stacking (`metrics_stacking.json`)

### Permutation / SHAP
- [ ] (P2) Permutation importance stabile su holdout (`permutation_importance_final.csv/png`)
- [ ] (P2) (Optional) SHAP summary LightGBM (`shap_summary.png`)

### Learning Curves
- [ ] (P2) Learning curve per RandomForest (`learning_curve_rf.png`)
- [ ] (P2) Learning curve per MLP (se implementato)

---
## P3 – Nice-to-Have
- [ ] (P3) MLP (`mlp.py`) + metriche/learning curve
- [ ] (P3) Feature selection (SelectKBest o RFECV) (`feature_selection_curve.png`)
- [ ] (P3) DBSCAN & Agglomerative clustering (`cluster_dbscan.png`, `dendrogram.png`)
- [ ] (P3) UMAP / t-SNE embedding (`umap_train.png`)
- [ ] (P3) Anomaly detection LOF confronto IsolationForest
- [ ] (P3) Aggregatore metriche (`experiments_summary.csv`)
- [ ] (P3) Salvataggio modelli finali (`models/best_<model>.joblib`)

---
## Completati (log)
- [x] (P1) Pipeline RF + LGBM con HPO
- [x] (P1) Feature importances + error analysis base
- [x] (P1) Baselines script (DecisionTree, RF base, KNN)
- [x] (P1) Clustering enrichment (PCA + KMeans + IsolationForest) con leakage safeguard
- [x] (P1) README aggiornato + template report

(Aggiungere qui man mano: data ed eventuale breve nota)

---
## Note Operative
- Usare FAST_MODE/FAST_BASELINES=1 solo per sviluppo; per risultati finali disattivarli.
- Evitare leakage: tutte le trasformazioni dentro Pipeline o eseguite prima di train/test split e non ricombinate.
- Ogni script salva artefatti in `data/processed/` e figure PNG: mantenere nomi coerenti.
- A fine giornata: fare commit con messaggio che lista tasks completati.

## Template commit message suggerito
feat: <breve> (tasks: P1-3, P2-1)
fix: <se bug>
docs: update report (sections ...)

---
Aggiornare questo file ad ogni nuova attività significativa.
