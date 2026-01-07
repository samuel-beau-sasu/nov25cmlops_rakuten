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
EXPERIMENT_ID = '641549194285215590'
RUN_ID = '1332ab4ac58e48ee8eb9f9d1c1d64201'

model_path = f'/home/ubuntu/nov25cmlops_rakuten/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/SVC_rakuten'

# 3. Charger le modèle
print("Chargement du modèle...")
model = mlflow.sklearn.load_model(model_path)

# 4. Faire des prédictions sur l'ensemble du jeu de données
print("Calcul des prédictions...")

pipeline = PredictionPipeline()
pred = pipeline.run([text])

# 5. Calculer et afficher la moyenne des prédictions
logger.success(f"Texte : {text}")
logger.success(f"prdtypecode prédit : {pred[0]}")
