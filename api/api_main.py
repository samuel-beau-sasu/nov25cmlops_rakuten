from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
import pandas as pd
import io
import logging
from loguru import logger
from datetime import datetime
import uuid
import subprocess
import os 
import sys
import re
import mlflow
from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='MLOps Rakuten API',
    description='API pour le training et les prédictions du modèle Rakuten',
    version='2.0.0'
)

security = HTTPBasic()

# Base de données des utilisateurs
USERS_DB = {
    "alice": "wonderland",
    "bob": "builder",
    "clementine": "mandarine"
}

ADMIN_CREDENTIALS = {
    "admin": "4dm1N"
}

# Modèles Pydantic
class FileInfo(BaseModel):
    filename: str
    upload_time: str
    rows: int
    columns: List[str]
    file_size_kb: float

class TrainingJobStatus(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    error: Optional[str] = None

class LoadAndTrainResponse(BaseModel):
    status: str
    message: str
    file_info: Optional[FileInfo] = None
    job_id: Optional[str] = None
    check_status_url: Optional[str] = None

class PredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="Liste de textes à prédire")

class PredictionResponse(BaseModel):
    predictions: List[int]
    count: int

# Stockage des fichiers et des jobs
training_data = {
    "X_train": None,
    "Y_train": None
}

training_jobs: Dict[str, Dict] = {}

# Fonctions d'authentification
def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    if username not in ADMIN_CREDENTIALS or ADMIN_CREDENTIALS[username] != password:
        raise HTTPException(
            status_code=401,
            detail="Accès administrateur requis",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

def validate_csv(file: UploadFile, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Lit et valide un fichier CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")
    
    content = file.file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier CSV est vide")
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Colonnes requises manquantes: {missing_columns}"
                )
        
        return df
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture du CSV: {str(e)}")


# ==================== FONCTION DE TRAINING EN BACKGROUND ====================
    
def run_training_job(job_id: str):

    
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        logger.info(f"🚀 Job {job_id}: Début du pipeline de training")
        logger.info(f"📁 Répertoire courant: {os.getcwd()}")
        
        # 4. Entraînement du modèle
        model_trainer_pipeline = ModelTrainerPipeline()
        model_path = model_trainer_pipeline.run()
        
        #logger.success(f"Modèle entraîné disponible à : {model_path}")       
        logger.info(f"Modèle entraîné disponible à : {model_path}")  
        logger.info(f"✅ Job {job_id}: Training terminé avec succès")
        
    except subprocess.TimeoutExpired:
        training_jobs[job_id].update({
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": "Timeout après 1 heure",
            "stdout": stdout[:500],
            "stderr": stderr[:500]
        })
        logger.error(f"⏰ Job {job_id}: Timeout après 1 heure")
        
    except Exception as e:
        training_jobs[job_id].update({
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": str(e),
            "stdout": stdout[:500],
            "stderr": stderr[:500]
        })
        logger.error(f"❌ Job {job_id}: Erreur - {str(e)}")
      
# ==================== ENDPOINTS ====================

@app.get('/')
def get_root():
    return {
        "message": "API MLOps Rakuten",
        "version": "2.0.0",
        "endpoints": {
            "admin": {
                "load_and_train": "POST /admin/load-and-train",
                "training_status": "GET /admin/training-job/{job_id}",
                "list_jobs": "GET /admin/training-jobs"
            },
            "public": {
                "health": "GET /health",
                "predict": "POST /predict"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "mlops_rakuten_api"
    }


# ==================== ENDPOINT PRINCIPAL : TRAINING ====================

@app.post("/admin/load-and-train", response_model=LoadAndTrainResponse)
async def load_and_train_data(
    background_tasks: BackgroundTasks,
    x_train_file: UploadFile = File(..., description="Fichier X_train.csv"),
    y_train_file: UploadFile = File(..., description="Fichier Y_train.csv"),
    admin: str = Depends(authenticate_admin)
    ):
    """
    Charge les données et lance le training en arrière-plan.
    
    Le training peut prendre plusieurs minutes. Utilisez le job_id 
    pour vérifier le statut via GET /admin/training-job/{job_id}
    """
    try:
        logger.info(f"Admin {admin} : Chargement et validation des fichiers...")
        
        # 1. Validation des fichiers
        x_train_df = validate_csv(
            x_train_file, 
            required_columns=['designation', 'description', 'productid', 'imageid']
        )
        
        y_train_df = validate_csv(
            y_train_file,
            required_columns=['prdtypecode']
        )
        
        # Vérifier la correspondance
        if len(x_train_df) != len(y_train_df):
            raise HTTPException(
                status_code=400,
                detail=f"Nombre de lignes différent. X: {len(x_train_df)}, Y: {len(y_train_df)}"
            )
        
        # 2. Stocker les données (optionnel, si vos pipelines en ont besoin)
        training_data["X_train"] = x_train_df
        training_data["Y_train"] = y_train_df
        
        # 3. Créer un job de training
        job_id = str(uuid.uuid4())
        training_jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "created_by": admin,
            "samples": len(x_train_df)
        }
        
        # 4. Lancer le training en arrière-plan
        background_tasks.add_task(run_training_job, job_id)
        
        logger.info(f"✅ Job {job_id} créé et mis en file d'attente")
        
        # 5. Informations sur le fichier
        file_info = FileInfo(
            filename=f"{x_train_file.filename}, {y_train_file.filename}",
            upload_time=datetime.now().isoformat(),
            rows=len(x_train_df),
            columns=list(x_train_df.columns) + list(y_train_df.columns),
            file_size_kb=0.0  # Calculé si nécessaire
        )
        
        return LoadAndTrainResponse(
            status="accepted",
            message=f"Training lancé en arrière-plan. {len(x_train_df)} échantillons chargés.",
            file_info=file_info,
            job_id=job_id,
            check_status_url=f"/admin/training-job/{job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.get("/admin/training-job/{job_id}", response_model=TrainingJobStatus)
async def get_training_job_status(
    job_id: str, 
    admin: str = Depends(authenticate_admin)
    ):
    """
    Vérifie le statut d'un job de training
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    job_data = training_jobs[job_id]
    
    return TrainingJobStatus(
        job_id=job_id,
        status=job_data.get("status"),
        created_at=job_data.get("created_at"),
        started_at=job_data.get("started_at"),
        completed_at=job_data.get("completed_at"),
        message=job_data.get("message"),
        model_path=job_data.get("model_path"),
        metrics_path=job_data.get("metrics_path"),
        error=job_data.get("error")
    )


@app.get("/admin/training-jobs")
async def list_training_jobs(admin: str = Depends(authenticate_admin)):
    """
    Liste tous les jobs de training
    """
    return {
        "total_jobs": len(training_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job_data.get("status"),
                "created_at": job_data.get("created_at"),
                "created_by": job_data.get("created_by")
            }
            for job_id, job_data in training_jobs.items()
        ]
    }


# ==================== ENDPOINT DE PRÉDICTION ====================

@app.post("/predict", response_model=PredictionResponse)
async def predict_texts(request: PredictionRequest):
    try:
        # 1. Charger le modèle une seule fois
        #EXPERIMENT_ID = '641549194285215590'
        #RUN_ID = '1332ab4ac58e48ee8eb9f9d1c1d64201'
        #model_path = f'/home/ubuntu/nov25cmlops_rakuten/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/SVC_rakuten'
        
        RUN_ID = 'd99ad58b60b04e1f83403f2035f08db0'
        model_path = f"runs:/{RUN_ID}/LR_rakuten"  # Format correct MLflow


        try:
            model = mlflow.sklearn.load_model(model_path)
            pipeline = PredictionPipeline()
        except FileNotFoundError as e:
            logger.error(f"Modèle introuvable: {str(e)}")
            raise HTTPException(status_code=404, detail="Modèle introuvable.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise HTTPException(status_code=500, detail="Erreur interne lors du chargement du modèle.")

        # 2. Calculer les prédictions pour tous les textes
        predictions_list = []
        for text in request.texts:
            try:
                pred = pipeline.run([text])
                predictions_list.extend(pred)  # ou append(pred[0]) selon le format de retour
            except ValueError as e:
                logger.error(f"Erreur de prédiction pour le texte '{text}': {str(e)}")
                raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")
            except Exception as e:
                logger.error(f"Erreur inattendue pour le texte '{text}': {str(e)}")
                raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

        # 3. Retourner la réponse
        return PredictionResponse(
            predictions=predictions_list,
            count=len(predictions_list)
        )

    except HTTPException:
        # Les exceptions HTTP sont déjà levées, pas besoin de les recapturer ici
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue globale: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)