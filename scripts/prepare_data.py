# scripts/prepare_data.py
import pandas as pd

def prepare_data():
    print("Préparation des données...")
    # Exemple : Charger des données brutes et les transformer
    raw_data = pd.read_excel("/medmouheb-bouzidi-4DS6-ml_project/churn-bigml-80.xlsx")
    processed_data = raw_data.dropna()
    processed_data.to_csv("data/processed/processed_data.csv", index=False)
