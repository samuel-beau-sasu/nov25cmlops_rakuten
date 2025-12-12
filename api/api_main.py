from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
import pandas as pd
import io
import logging
from datetime import datetime
import uuid
import subprocess
import os 
import sys
#import json
import re



# ✅ Import de votre fonction de training refactorisée
#from mlops_rakuten.main import run_training_pipeline, run_prediction


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
    """
    Exécute le pipeline de training en arrière-plan avec 'make train'
    Version simplifiée et robuste
    """
    # Initialiser les résultats
    result = None
    stdout = ""
    stderr = ""
    
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        logger.info(f"🚀 Job {job_id}: Début du pipeline de training")
        logger.info(f"📁 Répertoire courant: {os.getcwd()}")
        
        # ✅ Exécution de 'make train' via subprocess
        result = subprocess.run(
            #["make", "train"],
            [sys.executable, "mlops_rakuten/main.py", "train"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=3600
        )
        
        # Récupérer les sorties
        stdout = result.stdout
        stderr = result.stderr
        
        logger.info(f"📊 Job {job_id}: make train terminé avec code {result.returncode}")
        logger.info(f"📤 stdout: {len(stdout)} caractères")
        logger.info(f"📥 stderr: {len(stderr)} caractères")
        
        # Vérifier le code de retour
        if result.returncode != 0:
            # Le processus a échoué
            error_msg = f"make train a échoué (code: {result.returncode})"
            if stderr:
                # Prendre la première ligne d'erreur significative
                error_lines = [line for line in stderr.split('\n') if line.strip()]
                if error_lines:
                    error_msg += f" - {error_lines[0]}"
            
            raise Exception(error_msg)
        
        # ✅ SUCCÈS : Analyser la sortie
        training_results = {
            "status": "success",
            "return_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "message": "Training terminé avec succès"
        }
        
        # Extraire les chemins (si présents dans la sortie)
        if "Modèle :" in stdout:
            for line in stdout.split('\n'):
                if "Modèle :" in line:
                    training_results["model_path"] = line.split("Modèle :")[-1].strip()
                elif "Métriques :" in line:
                    training_results["metrics_path"] = line.split("Métriques :")[-1].strip()
        
        # Mettre à jour le job
        training_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "model_path": training_results.get("model_path"),
            "metrics_path": training_results.get("metrics_path"),
            "message": training_results["message"],
            "results": training_results,
            "stdout": stdout[:500],  # Stocker un extrait
            "stderr": stderr[:500]
        })
        
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

class PredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="Liste de textes à prédire")

class PredictionResponse(BaseModel):
    predictions: List[int]
    count: int

#@app.post("/predict", response_model=PredictionResponse)
async def predict_texts_old(request: PredictionRequest):
    """
    Effectue des prédictions sur une liste de textes
    """
    try:
        logger.info(f"Prédiction pour {len(request.texts)} texte(s)")
        
        # ✅ APPEL DE VOTRE FONCTION DE PRÉDICTION
        #predictions = run_prediction(request.texts)
        
        return PredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_texts(request: PredictionRequest):
    try:
        predictions_list = []

        for text in request.texts:
            process_result = subprocess.run(
                [
                    sys.executable,
                    "-m", "mlops_rakuten.main",
                    "predict",
                    text
                ],
                capture_output=True,
                text=True,
                timeout=3600
            )

            #stdout = process_result.stdout.strip()
            stderr = process_result.stderr.strip()

            #logger.info(f"Sortie stdout: {stdout}")
            logger.info(f"Sortie stderr: {stderr}")

            if process_result.returncode != 0:
                error_message = stderr
                logger.error(f"Erreur lors de l'exécution du script: {error_message}")
                raise HTTPException(status_code=500, detail=f"Erreur lors de l'exécution du script: {error_message}")

            # Utiliser une expression régulière pour extraire la prédiction
            match = re.search(r"prdtypecode prédit : (\d+)", stderr)
            
            prediction = int(match.group(1))
            predictions_list.append(prediction)

        return PredictionResponse(
            predictions=predictions_list,
            count=len(predictions_list)
        )

    except subprocess.TimeoutExpired:
        logger.error("Le processus a dépassé le temps limite autorisé.")
        raise HTTPException(status_code=504, detail="Le processus a dépassé le temps limite autorisé.")

    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)