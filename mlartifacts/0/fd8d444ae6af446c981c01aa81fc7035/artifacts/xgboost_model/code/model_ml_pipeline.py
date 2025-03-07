import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
def preparedata(x1='/home/montahar/montaha_rebhi_4ds6_ml_project/churn-bigml-20.csv',x2='/home/montahar/montaha_rebhi_4ds6_ml_project/churn-bigml-80.csv'):
	df_20=pd.read_csv(x1)
	df_80=pd.read_csv(x2)
	dfm=pd.merge(df_80,df_20,on=None,how='outer')
	
	dfm['International plan'] = dfm['International plan'].map({'Yes': 1, 'No': 0})
	dfm['Voice mail plan'] = dfm['Voice mail plan'].map({'Yes': 1, 'No': 0})
	encoder = LabelEncoder()

	dfm['State'] = encoder.fit_transform(dfm['State'])
	
	X = dfm.drop('Churn', axis=1)
	y = dfm['Churn']

	x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train_scaled)
	x_test_scaled = scaler.transform(x_test_scaled)
	
	return x_train_scaled, x_test_scaled, y_train, y_test

def train_model(x_train_scaled, y_train, n_estimators=100, max_depth=3, learning_rate=0.3):
    """
    Entraîne un modèle XGBoost avec les hyperparamètres spécifiés et enregistre le modèle avec MLflow.
    """

    with mlflow.start_run():  # 🔴 Démarre un run MLflow
        reg = XGBClassifier(
            n_estimators=n_estimators,  # 🟢 Hyperparamètre dynamique
            max_depth=max_depth,        # 🟢 Hyperparamètre dynamique
            learning_rate=learning_rate,  # 🟢 Hyperparamètre dynamique
            eval_metric="logloss",
            use_label_encoder=False
        )

        # 🔄 Entraînement du modèle
        reg.fit(x_train_scaled, y_train, eval_set=[(x_train_scaled, y_train)], verbose=True)

        # ✅ Log des hyperparamètres dans MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # ✅ Évaluer la performance et logguer l'accuracy
        y_pred = reg.predict(x_train_scaled)
        acc = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", acc)

        # 📦 Enregistrer le modèle avec MLflow
        mlflow.xgboost.log_model(reg, "xgboost_model", code_paths=["model_ml_pipeline.py"])

        print(f"✅ Modèle entraîné et loggé avec accuracy: {acc:.2f}")

    return reg  # 🔁 Retourne le modèle entraîné
def evaluate_model(reg,x_test_scaled,y_test):
	y_pred=reg.predict(x_test_scaled)
	acc=accuracy_score(y_test,y_pred)
	print(f"Accuracy: {acc:.2f}")
	print("Classification Report:")
	print(classification_report(y_test, y_pred))

def save_model(reg, filename='model.pkl'):
    
    joblib.dump(reg, filename)
    print(f"Model saved as {filename}")

def load_model(filename='model.pkl'):
    
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

# Fonction predict avec MLflow (XGBoost Model)

def predict(input_features):
    import mlflow.pyfunc
    import numpy as np

    model_uri = "models:/xgboost_model/Production"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction avec XGBoost : {str(e)}"}
