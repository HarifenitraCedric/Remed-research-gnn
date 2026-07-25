import pandas as pd

try:
    df = pd.read_csv("C:/RMD/remed-research-gnn/interactions_ansm.csv", nrows=5)
    print("Colonnes détectées :", df.columns.tolist())
    print("\nPremières lignes :")
    print(df)
except Exception as e:
    print("Erreur de lecture :", e)