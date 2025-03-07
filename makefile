# Define Python environment variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
MODEL_FILE=xgboost_model.pkl

# Set shell explicitly to Bash
SHELL := /bin/bash

# 1️⃣ Setup environment
setup:
	@echo "🔹 Creating virtual environment and installing dependencies..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@bash -c "source $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)"

# 2️⃣ Prepare data
data:
	@echo "🔹 Preparing data..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main_ml_pipeline.py --prepare"

# 3️⃣ Train the model
train:
	@echo "🔹 Training the model..."
	@bash -c "source $(ENV_NAME)/bin/activate && \
	export MLFLOW_TRACKING_URI=http://localhost:5000 && \
	$(PYTHON) main_ml_pipeline.py --train"

# 4️⃣ Evaluate the model
evaluate:
	@echo "🔹 Evaluating the model..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main_ml_pipeline.py --evaluate"

# 5️⃣ Save the trained model
save:
	@echo "🔹 Saving the model..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main_ml_pipeline.py --save"

# 6️⃣ Load and re-evaluate the model
load:
	@echo "🔹 Loading and evaluating the model..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) main_ml_pipeline.py --load"

# 7️⃣ Run inline unit tests
test:
	@echo "🔹 Running inline unit tests..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) -c '\
import sys; \
from model_ml_pipeline import preparedata, train_model, evaluate_model; \
X_train, X_test, y_train, y_test = preparedata(); \
assert X_train.shape[0] > 0, \"❌ Test Failed: No training data\"; \
assert X_test.shape[0] > 0, \"❌ Test Failed: No test data\"; \
model = train_model(X_train, y_train); \
assert model is not None, \"❌ Test Failed: Model is None\"; \
y_pred = model.predict(X_test); \
from sklearn.metrics import accuracy_score; \
acc = accuracy_score(y_test, y_pred); \
assert acc > 0.7, f\"❌ Test Failed: Accuracy too low ({acc:.2f})\"; \
print(\"✅ All tests passed!\")'"
# 8️⃣ Run all steps
all: setup data train evaluate save load test notebook

# 9️⃣ Clean temporary files
clean:
	@echo "🔹 Cleaning up temporary files..."
	@rm -rf $(ENV_NAME) __pycache__ *.log $(MODEL_FILE)
#11 
run-api:
	uvicorn app:app --reload
# Règle Makefile pour la prédiction
predict:
	@echo "🔹 Exécution de la prédiction avec MLflow..."
	@bash -c "source $(ENV_NAME)/bin/activate && $(PYTHON) -c 'import mlflow; mlflow.set_tracking_uri(\"http://0.0.0.0:5000\"); import main_ml_pipeline; main_ml_pipeline.predict()'"

#detection du changement et make all automatique 
watch:
	@echo "🔹 Surveillance automatique des fichiers... (Arrêtez avec Ctrl+C)"
	@while true; do \
		CHANGED_FILES=$$(find . -type f -mmin -1 2>/dev/null); \
		if [ ! -z "$$CHANGED_FILES" ]; then \
			echo "🔄 Fichiers modifiés : $$CHANGED_FILES"; \
			make all; \
		fi; \
		sleep 2; \
	done

mlflow-ui:
	@echo "🚀 Starting MLflow UI..."
	@mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
# 1️⃣ Lancer Elasticsearch et Kibana
start-monitoring:
	@echo "🚀 Démarrage d'Elasticsearch et Kibana..."
	docker-compose up -d
	@echo "✅ Elasticsearch et Kibana sont lancés."

# 2️⃣ Lancer MLflow
start-mlflow:
	@echo "🚀 Lancement de MLflow..."
	@mlflow server --backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlruns \
	--host 0.0.0.0 --port 5000 &
	@echo "✅ MLflow est démarré sur http://localhost:5000"

#10 notebook
notebook:
	@echo "🔹 Démarrage de Jupyter Notebook..."
	@bash -c "source $(ENV_NAME)/bin/activate && nohup $(ENV_NAME)/bin/jupyter notebook --no-browser > jupyter.log 2>&1 &"
	@sleep 3
	@$(PYTHON) -c "import webbrowser; webbrowser.open('http://localhost:8888/tree')"
#docker
# Construire l’image Docker
build-docker:
	@echo "Construction de l’image Docker..."
	@docker build -t montaha_rebhi_4ds6_mlops .

# Exécuter le conteneur Docker
run-docker:
	@echo "Lancement du conteneur Docker..."
	@docker run -p 8000:8000 montaha_rebhi_4ds6_mlops
# Pousser l’image sur Docker Hub
push-docker:
	@echo "Poussée de l’image sur Docker Hub..."
	@docker tag montaha_rebhi_4ds6_mlops montaha25/montaha_rebhi_4ds6_mlops:v1
	@docker push montaha25/montaha_rebhi_4ds6_mlops:v1
