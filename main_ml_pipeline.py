import numpy as np
import mlflow
from model_ml_pipeline import preparedata, train_model, evaluate_model, save_model, load_model
from elasticsearch import Elasticsearch
import logging
import mlflow.sklearn
import datetime
# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri("http://localhost:5000")
# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Configuration du logger
logger = logging.getLogger("mlflow_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

# Fonction pour envoyer les logs vers Elasticsearch
# Fonction pour envoyer des logs
def log_to_elasticsearch(message):
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "message": message
    }
    es.index(index="mlflow-metrics", body=log_data)
    print(f"✅ Log envoyé à Elasticsearch: {message}")

# Test d’envoi d’un log
log_to_elasticsearch("Test de log MLflow vers Elasticsearch !")

def predict():
    print("\n=== Prédiction ===")
    try:
        # Assuming the model is loaded using joblib or pickle
        model = load_model('xgboost_model.pkl')
        
        # Example data (9 features here, needs to be adjusted to 19 features)
        new_data = np.array([[1, 25, 200, 2, 0, 1, 300, 3, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # Add dummy features
        
        # Directly pass new_data to the model's predict method
        prediction = model.predict(new_data)
        print(f"Prediction: {prediction}")
        
    except Exception as e:
        print(f"Une erreur est survenue pendant la prédiction : {e}")    

def main():

    X_train_scaled, X_test_scaled, y_train, y_test = preparedata()    

    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    save_model(model, 'xgboost_model.pkl')

    loaded_model = load_model('xgboost_model.pkl')

    evaluate_model(loaded_model, X_test_scaled, y_test)



main()
predict()
