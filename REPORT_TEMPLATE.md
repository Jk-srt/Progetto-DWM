# Report (Max 5 pagine)

## 1. Obiettivo e Dati
- Competizione: Child Mind Institute Problematic Internet Use
- Target: `sii`
- Dimensioni dataset: (da compilare dopo run) righe / colonne
- Preprocessing eseguito: drop >50% NA, imputazione mediane numeriche, imputazione categoriche (pipeline), one-hot encoding.
- Colonne eliminate: (inserire elenco / count)

## 2. Analisi Esplorativa (Sintesi)
- Class distribution: (inserire counts / imbalance ratio)
- Principali variabili informative (prime 5 importances)
- Pattern di missing rilevanti

## 3. Metodologia Supervised
### Pipeline
- ColumnTransformer (num: median impute; cat: most_frequent + one-hot)
- Modelli confrontati: RandomForest (sklearn), LightGBM
- Cross-Validation: StratifiedKFold (5 folds)
- Metriche CV: Accuracy, Macro-F1, ROC-AUC OVR weighted, Average Precision Macro

### Tuning Iperparametri
- RF: spazio {n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features}
- LGBM: spazio {n_estimators, learning_rate, num_leaves, max_depth, subsample, colsample_bytree, reg_lambda}
- Strategia: RandomizedSearchCV

## 4. Risultati
Inserire tabella (CV mean ± std)

| Modello | Acc | F1 Macro | ROC-AUC | Avg Precision |
|---------|-----|----------|---------|---------------|
| RF      |     |          |         |               |
| LGBM    |     |          |         |               |

- Modello selezionato: (nome) motivazione (miglior F1 Macro / trade-off)
- Holdout: Accuracy = , F1 Macro = , Confusion Matrix = allegata

## 5. Interpretabilità
- Feature Importances (sintesi): prime 10
- Permutation vs impurity importances (coerenza / differenze)
- Error Analysis: pattern nei top wrong ad alta confidenza (es. valori estremi di X, missing pattern)

## 6. Export Kaggle
- File submission prodotto: `submission_<MODEL>.csv`
- Strategia: refitting su tutto il training prima della predizione di test.

## 7. Limiti e Lavori Futuri
- Possibile nested CV per ridurre bias di tuning.
- Aggiunta baseline KNN/DecisionTree per bias-variance.
- Calibrazione probabilità e threshold tuning.
- Feature selection ablation curve.

## 8. Conclusioni
- Sintesi miglior modello, metriche chiave.
- Principali driver della classe.

---
Appendici (se spazio): grafici ROC / PR, top 20 importances.
