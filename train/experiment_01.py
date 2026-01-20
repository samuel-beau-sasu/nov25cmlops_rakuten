from loguru import logger

from mlops_rakuten.config.config_manager import ConfigurationManager
from mlops_rakuten.modules.model_trainer import ModelTrainer_mlflow, ModelTrainer

def main():
    """
    Entraîne un modèle avec tracking MLflow.
    """
    logger.info("Démarrage de l'expérimentation avec MLflow")
    
    # 1. Charger la configuration
    config_manager = ConfigurationManager()
    model_trainer_config = config_manager.get_model_trainer_config()
    
    # 2. Créer le trainer avec MLflow
    #trainer = ModelTrainer_mlflow(config=model_trainer_config )
    trainer = ModelTrainer(config=model_trainer_config )
    
    # 3. Lancer l'entraînement
    model_path = trainer.run()
    
    logger.success(f"✓ Modèle entraîné : {model_path}")
    #logger.success(f"✓ Consultez MLflow UI: http://127.0.0.1:5001")


if __name__ == "__main__":
    main()