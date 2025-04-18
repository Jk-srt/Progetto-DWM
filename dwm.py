import pandas as pd

df = pd.read_csv("train.csv")  
df.head()  # Visualizza le prime 5 righe del dataset
df.shape  # (numero di righe, numero di colonne)
df.columns
df.info()   # Tipi di dati e valori nulli
df.describe()  # Statistiche generali
df.isnull().sum()  # Valori mancanti
df.duplicated().sum()  # Duplicati

# Pulizia dati
df.isnull().sum() # Valori mancanti