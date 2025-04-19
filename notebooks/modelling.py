# notebooks/modelling.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"


def main():
    # Carica dati preprocessati
    df = pd.read_csv(DATA_DIR / "train_clean_preprocessed.csv")

    # 1. Identifica colonne
    X = df.drop(columns=['sii', 'id'])
    y = df['sii']

    # 2. Separa colonne numeriche e categoriche
    num_cols = X.select_dtypes(include=['int16', 'float32']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    print(f"Colonne numeriche: {num_cols.tolist()}")
    print(f"Colonne categoriche: {cat_cols.tolist()}")

    # 3. Crea transformers
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. Combina in ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 5. Crea pipeline completa
    model = Pipeline([
        ('prep', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 6. Split e addestramento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 7. Valutazione
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
