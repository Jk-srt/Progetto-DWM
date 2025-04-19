# notebooks/preprocessing.py
import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Caricamento dati
    df = pd.read_csv(DATA_DIR / "train.csv")

    # 1. Rimuovi colonne con >50% valori mancanti
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    df = df.drop(columns=cols_to_drop)

    # 2. Elimina righe senza target
    if 'sii' in df.columns:
        df = df.dropna(subset=['sii'])

    # 3. Imputazione valori mancanti
    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    # 4. Salva
    df.to_csv(PROCESSED_DIR / "train_clean_preprocessed.csv", index=False)
    print("Preprocessing completato!")


if __name__ == "__main__":
    main()
