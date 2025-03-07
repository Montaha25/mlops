import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from model_ml_pipeline import preparedata, train_model, evaluate_model

def test_prepare_data():
    try:
        x_train_scaled, x_test_scaled, y_train, y_test = preparedata()
        assert x_train_scaled.shape[0] > 0, "Aucune donnée d'entraînement préparée"
        assert x_test_scaled.shape[0] > 0, "Aucune donnée de test préparée"
        print("✅ test_prepare_data : Préparation des données réussie")
    except Exception as e:
        print(f"❌ test_prepare_data : Erreur - {e}")

def test_train_model():
    try:
        x_train_scaled, x_test_scaled, y_train, y_test = preparedata()
        model = train_model(x_train_scaled, y_train)
        assert model is not None, "Le modèle n'a pas été entraîné"
        print("✅ test_train_model : Entraînement du modèle réussi")
    except Exception as e:
        print(f"❌ test_train_model : Erreur - {e}")

def test_model_accuracy():
    try:
        x_train_scaled, x_test_scaled, y_train, y_test = preparedata()
        model = train_model(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.7, f"Précision insuffisante : {acc:.2f}"
        print(f"✅ test_model_accuracy : Précision acceptable ({acc:.2f})")
    except Exception as e:
        print(f"❌ test_model_accuracy : Erreur - {e}")

if _name_ == "_main_":
    print("🔍 Démarrage des tests...")
    test_prepare_data()
    test_train_model()
    test_model_accuracy()