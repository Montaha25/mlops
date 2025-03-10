# Utiliser une image de base officielle Python en version 3.9 avec une version minimale du système d'exploitation
FROM python:3.9-slim
# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app
# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt /app
# Installer les dépendances Python listées dans requirements.txt sans mettre en cache pour réduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt
# Copier le contenu du dossier app local dans le répertoire de travail du conteneur
COPY . /app
# Ouvrir le port 5000 pour permettre les connexions à l'application
EXPOSE 8000
# Définir la commande par défaut à exécuter lorsque le conteneur démarre
CMD ["uvicorn", "app:app", "--port", "8000"]

