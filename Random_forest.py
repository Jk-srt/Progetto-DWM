import pandas as pd
df = pd.read_csv('train_clean.csv')
print(df.columns)
print(df.head())

X = df.drop('nome_colonna_target', axis=1)
y = df['nome_colonna_target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
