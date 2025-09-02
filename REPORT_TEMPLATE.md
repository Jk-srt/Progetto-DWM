# Report (Max 5 pagine)

## 1. Obiettivo e Dati
- Competizione: Child Mind Institute Problematic Internet Use
- Target: `sii`
- Dimensioni dataset: (da compilare dopo run) righe / colonne
- Preprocessing eseguito: drop >50% NA, imputazione mediane numeriche, imputazione categoriche (pipeline), one-hot encoding.
- Colonne eliminate: (inserire elenco / count)
- Rimozione automatica di 14 feature ad alta purezza (>=0.995) per evitare leakage; calo controllato delle metriche ma maggiore realismo.
- Elenco feature ad alta purezza rimosse (documentare qui): `BIA-BIA_BMC, BIA-BIA_BMR, BIA-BIA_DEE, BIA-BIA_ECW, BIA-BIA_FFM, BIA-BIA_FFMI, BIA-BIA_FMI, BIA-BIA_Fat, BIA-BIA_ICW, BIA-BIA_LDM, BIA-BIA_LST, BIA-BIA_SMM, BIA-BIA_TBW, PCIAT-PCIAT_Total`.
 - Tabella conteggio feature:
	 | Stato | #Feature |
	 |-------|----------|
	 | Iniziale (train) |   |
	 | Dopo drop manuale |   |
	 | Dopo auto-drop high-purity |   |

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
Strategia: RandomizedSearchCV (5-fold Stratified) ottimizzando Macro-F1.

RandomForest:
- Search space:
	- n_estimators: {200, 400, 800}
	- max_depth: {None, 10, 20, 40}
	- min_samples_split: {2, 5, 10}
	- min_samples_leaf: {1, 2, 4}
	- max_features: {"sqrt", "log2", None}
- Iterazioni campionate: 20
- Best params: n_estimators=400, max_depth=None, min_samples_split=10, min_samples_leaf=4, max_features=sqrt

LightGBM:
- Search space:
	- n_estimators: {300, 600, 1000}
	- learning_rate: {0.05, 0.1, 0.2}
	- num_leaves: {31, 63, 127}
	- max_depth: {-1, 6, 12}
	- subsample: {0.7, 0.9, 1.0}
	- colsample_bytree: {0.7, 0.9, 1.0}
	- reg_lambda: {0.0, 0.1, 1.0}
- Iterazioni campionate: 25
- Best params: n_estimators=300, learning_rate=0.1, num_leaves=63, max_depth=-1, subsample=0.9, colsample_bytree=0.7, reg_lambda=1.0

Nota: sono stati scelti spazi moderati per coprire ampiezza (profondità/alberi) e regolarizzazione senza esplosione combinatoria; ulteriori raffinamenti (es. tuning fine learning_rate + early stopping) indicati come lavoro futuro.

## 4. Risultati
Inserire tabella (CV mean ± std)

| Modello | Acc | F1 Macro | ROC-AUC | Avg Precision |
|---------|-----|----------|---------|---------------|
| RF      |     |          |         |               |
| LGBM    |     |          |         |               |

- Modello selezionato: (nome) motivazione (miglior F1 Macro / trade-off)
- Holdout: Accuracy = , F1 Macro = , Confusion Matrix = allegata

### 4.1 Leakage Mitigation & Ablation
Abbiamo confrontato quattro strategie di gestione feature sospette/leakage (incluso uno scenario di controllo con tutte le feature) su 5-fold CV.

| Strategia | Modello | #Feat | #Drop | Acc mean | Acc std | F1 Macro mean | F1 Macro std |
|-----------|---------|-------|-------|----------|---------|---------------|--------------|
| D_all (nessun drop) | RandomForest | 80 | 0 | 0.9960 | 0.0027 | 0.9542 | 0.0309 |
| D_all (nessun drop) | LightGBM | 80 | 0 | 0.9993 | 0.0009 | 0.9922 | 0.0096 |
| C_high (soglia purezza alta) | RandomForest | 47 | 33 | 0.5983 | 0.0126 | 0.2806 | 0.0206 |
| C_high (soglia purezza alta) | LightGBM | 47 | 33 | 0.5724 | 0.0188 | 0.3348 | 0.0200 |
| B_min (drop minimo) | RandomForest | 45 | 35 | 0.6005 | 0.0077 | 0.2899 | 0.0135 |
| B_min (drop minimo) | LightGBM | 45 | 35 | 0.5673 | 0.0201 | 0.3186 | 0.0158 |
| A_full (strategia adottata) | RandomForest | 42 | 38 | 0.6042 | 0.0092 | 0.2973 | 0.0164 |
| A_full (strategia adottata) | LightGBM | 42 | 38 | 0.5702 | 0.0266 | 0.3230 | 0.0213 |

Osservazioni principali:
1. Scenario D_all produce metriche quasi perfette (macro-F1 >0.95 / >0.99) non realistiche: forte indicazione di leakage (feature derivate o surrogate del target).
2. Rimuovendo i blocchi sospetti (BIA-* e PCIAT_* + indici aggregati) le metriche crollano su valori plausibili (~0.28–0.33 Macro-F1) indicando valutazione più onesta.
3. Le differenze tra strategie senza leakage (A_full, B_min, C_high) sono piccole; A_full massimizza lievemente F1 RF, mentre C_high massimizza lievemente F1 LGBM. A_full è scelta per criteri di dominio e semplicità (elimina interi gruppi ad alto rischio di codifica indiretta del target e mismatch con test).
4. La riduzione di ~47% delle feature (80→42) elimina il segnale spurio mantenendo performance comparabili alle strategie alternative pulite.

Motivazione scelta finale: privilegiare robustezza e prevenzione leakage rispetto a guadagno marginale di una singola configurazione modello.

## 5. Interpretabilità
- Feature Importances (sintesi): prime 10
- Permutation vs impurity importances (coerenza / differenze) – Nota: permutation importance instabile (errori shape) con setup corrente; usate impurity importances validate. Possibile riallineare calcolando su fit unico (TODO).
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
