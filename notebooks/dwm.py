import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def log(msg):
    print(f"[LOG] {msg}")

DATA_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Carica tutto il dataset, ma con tipi di dato ottimizzati
log("Caricamento di tutte le colonne con dtype ottimizzati...")
# Ricava i nomi delle colonne e i tipi di dato più efficienti:
dtypes = {}
first_rows = pd.read_csv(DATA_DIR / "train.csv", nrows=100)
for col in first_rows:
    if first_rows[col].dtype == 'float64':
        dtypes[col] = 'float32'
    elif first_rows[col].dtype == 'int64':
        dtypes[col] = 'int16'

df = pd.read_csv(DATA_DIR / "train.csv", dtype=dtypes)
log(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne.")

# Visualizza info base
log("Info dataset:")
print(df.info())
log("Conteggio valori mancanti per colonna (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Heatmap valori mancanti (campione di 30 colonne, per non saturare RAM/CPU)
log("Plot mappa valori mancanti di 30 colonne a caso...")
import random
sample_cols = random.sample(list(df.columns), min(30, len(df.columns)))
plt.figure(figsize=(20,6))
sns.heatmap(df[sample_cols].isnull(), cbar=False, yticklabels=False)
plt.title('Mappa valori mancanti (30 colonne campione)')
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "missingmap_all_cols.png")
plt.close()
log("Plot salvato.")

# Distribuzione target (controlla sempre che esista la colonna 'sii')
if 'sii' in df.columns:
    log("Plot distribuzione della variabile target 'sii'...")
    plt.figure(figsize=(6,4))
    df['sii'].value_counts(dropna=False).sort_index().plot(kind='bar',
        color=['skyblue', 'orange', 'salmon', 'green', 'grey'])
    plt.title('Distribuzione classe target (sii)')
    plt.xlabel('sii')
    plt.ylabel('Conteggio')
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / 'target_distribution_all_cols.png')
    plt.close()
    log("Plot distribuzione target salvato.")

# Matrice di correlazione solo su numeriche principali (max 20)
log("Calcolo matrice di correlazione (max 20 colonne numeriche)...")
num_cols = list(df.select_dtypes(include=[np.number]).columns)
num_corr_cols = num_cols if len(num_cols) <= 20 else num_cols[:20]
corr = df[num_corr_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm', mask=np.triu(np.ones_like(corr)))
plt.title('Matrice di Correlazione (max 20 colonne numeriche)')
plt.tight_layout()
plt.savefig(PROCESSED_DIR / 'correlation_matrix_all_cols.png')
plt.close()
log("Matrice di correlazione salvata.")

log(f"Analisi COMPLETA su tutte le colonne terminata! Output in: {PROCESSED_DIR}")
