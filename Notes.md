# installation de mlflow

# Option 1: Version stable récente
uv add "mlflow>=2.16.0,<3.0"

uv sync

# lancer MLflow en local en exécutant la commande suivante dans le terminal 

mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri file:///home/ubuntu/nov25cmlops_rakuten/mlruns \
  --default-artifact-root file:///home/ubuntu/nov25cmlops_rakuten/mlruns \
  --serve-artifacts

#

source .venv/bin/activate

python src/train_model.py

# Exécuter le fichier MLproject en utilisant l'environnement virtuel local.

mlflow run src/ --experiment-id 641549194285215590 --run-name simple_run_reproduced --env-manager=local

#  Charger un Modèle avec MLflow Models