import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

from loguru import logger
import typer
from mlops_rakuten.pipelines.prediction import PredictionPipeline

# 1. Chargement des données
"""
Effectue une prédiction à partir d'un texte.

Exemple:
python -m mlops_rakuten.dataset predict "Super aspirateur sans fil"
"""
text = "Super aspirateur sans fil"

# 2. Définir le chemin vers le modèle MLflow
#EXPERIMENT_ID = '190155843876560128'
#RUN_ID = '4028188ca5cc4f6499a2614ff130c657'
#model_path = f"/home/ubuntu/Projet_MLPos/nov25cmlops_rakuten/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/SVC_rakuten"

# 2. Définir l'URI du modèle MLflow (CORRECTION ICI)
RUN_ID = '4028188ca5cc4f6499a2614ff130c657'
model_uri = f"runs:/{RUN_ID}/SVC_rakuten"  # Format correct MLflow


# 3. Charger le modèle
print("Chargement du modèle...")
model = mlflow.sklearn.load_model(model_uri)

# 4. Faire des prédictions sur l'ensemble du jeu de données
print("Calcul des prédictions...")

pipeline = PredictionPipeline()
pred = pipeline.run([text])

# 5. Calculer et afficher la moyenne des prédictions
logger.success(f"Texte : {text}")
logger.success(f"prdtypecode prédit : {pred[0]}")