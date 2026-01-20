from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import uuid
import io
import os
from datetime import datetime, timedelta

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from mlops_rakuten.auth.auth_simple import (
    authenticate_user,
    create_access_token,
    require_admin,
    require_user,
)
from mlops_rakuten.config.constants import (
    MODELS_DIR,
    REPORTS_DIR,
    UPLOADS_DIR,
)
from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.data_preprocessing import DataPreprocessingPipeline
from mlops_rakuten.pipelines.data_transformation import DataTransformationPipeline
from mlops_rakuten.pipelines.model_evaluation import ModelEvaluationPipeline
from mlops_rakuten.pipelines.model_trainer import ModelTrainerPipeline
from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.utils import create_directories, get_latest_run_dir

REQUIRED_COLUMNS = ["designation", "prdtypecode"]


app = FastAPI(
    title="Rakuten Product Classification API",
    description="API d'inférence pour la classification de produits Rakuten.",
    version="0.1.0",
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return path.read_text(encoding="utf-8")


def validate_uploaded_batch_csv(csv_path: Path) -> None:
    """
    Valide un batch uploadé :
    - colonnes requises
    - pas de NA sur colonnes requises
    - types : designation=str, prdtypecode=int
    """
    df = pd.read_csv(csv_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    # garder uniquement les colonnes qui nous intéressent
    df = df[REQUIRED_COLUMNS]

    if df[REQUIRED_COLUMNS].isna().any().any():
        # on liste pour debug minimal
        na_counts = df[REQUIRED_COLUMNS].isna().sum().to_dict()
        raise ValueError(f"Valeurs manquantes détectées: {na_counts}")

    # designation -> str (et non vide)
    df["designation"] = df["designation"].astype(str).str.strip()
    if (df["designation"].str.len() < 10).any():
        raise ValueError("Certaines 'designation' font moins de 10 caractères après strip().")

    # prdtypecode -> int strict
    # (si le CSV contient des "10.0" ou " 10", ça doit passer proprement)
    try:
        df["prdtypecode"] = pd.to_numeric(df["prdtypecode"], errors="raise").astype(int)
    except Exception as e:
        raise ValueError("La colonne 'prdtypecode' doit être convertible en int.") from e

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
    
# FONCTION DE TRAINING EN BACKGROUND   
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

# Initialisation Pipeline
logger.info("Initialisation globale du PredictionPipeline FastAPI")
prediction_pipeline = PredictionPipeline()
logger.success("PredictionPipeline initialisé")


# Schemas Pydantic
class PredictionRequest(BaseModel):
    designation: str = Field(
        min_length=10, description="Designation produit (au moins 10 caractères)."
    )
    top_k: int | None = 5

    @field_validator("designation")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("La designation doit contenir au moins 10 caractères non vides.")
        return v


class CategoryScore(BaseModel):
    prdtypecode: int
    category_name: Optional[str]
    proba: float


class PredictionResponse(BaseModel):
    designation: str
    predictions: List[CategoryScore]


class ModelConfigResponse(BaseModel):
    model_run_id: str
    model_configuration: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    report_run_id: str
    metrics_val: Dict[str, Any]


class ModelClassificationResponse(BaseModel):
    report_run_id: str
    classification_report_val: str

# Modèles Pydantic
class FileInfo(BaseModel):
    filename: str
    upload_time: str
    rows: int
    columns: List[str]
    file_size_kb: float


class LoadAndTrainResponse(BaseModel):
    status: str
    message: str
    file_info: Optional[FileInfo] = None
    job_id: Optional[str] = None
    check_status_url: Optional[str] = None
    


training_jobs: Dict[str, Dict] = {}

# Endpoints
@app.get("/health")
async def healthcheck(_=Depends(require_user)):
    return {"status": "ok"}

# ==================== ENDPOINT PRINCIPAL : TRAINING ====================

@app.post("/load-and-train", response_model=LoadAndTrainResponse)
async def load_and_train_data(
    background_tasks: BackgroundTasks,
    x_train_file: UploadFile = File(..., description="Fichier X_train.csv"),
    y_train_file: UploadFile = File(..., description="Fichier Y_train.csv"),
    _=Depends(require_admin)
    ):
    """
    Charge les données et lance le training en arrière-plan.
    
    Le training peut prendre plusieurs minutes. Utilisez le job_id 
    pour vérifier le statut via GET /admin/training-job/{job_id}
    """
    
    # Stockage des fichiers et des jobs
    training_data = {
        "X_train": None,
        "Y_train": None
    }
    
    try:
        logger.info(f"Chargement et validation des fichiers...")
        
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
            "created_by": "admin",
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
            check_status_url=f"/training-job/{job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, _=Depends(require_user)):
    logger.info("Requête /predict reçue")

    results_per_text = prediction_pipeline.run(
        texts=[request.designation],
        top_k=request.top_k,
    )

    preds_raw = results_per_text[0]
    preds = [
        CategoryScore(
            prdtypecode=p["prdtypecode"],
            category_name=p.get("category_name"),
            proba=p["proba"],
        )
        for p in preds_raw
    ]

    return PredictionResponse(
        designation=request.designation,
        predictions=preds,
    )


@app.post("/token")
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(username=user["username"], role=user["role"])
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/model/config", response_model=ModelConfigResponse)
async def get_model_config(_=Depends(require_admin)):
    """
    Retourne la config du dernier modèle:
    models/<latest>/model_config.json
    """
    try:
        latest_model_dir = get_latest_run_dir(MODELS_DIR)
        path = latest_model_dir / "model_config.json"

        logger.info(f"Lecture model_config.json : {path}")
        model_configuration = read_json(path)

        return ModelConfigResponse(
            model_run_id=latest_model_dir.name,
            model_configuration=model_configuration,
        )

    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON invalide: {e}")
        raise HTTPException(status_code=500, detail=f"JSON invalide: {e}") from e
    except Exception as e:
        logger.exception("Erreur inattendue dans /model/config")
        raise HTTPException(status_code=500, detail="Erreur interne serveur") from e


@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(_=Depends(require_admin)):
    """
    Retourne les métriques du dernier report:
    reports/<latest>/metrics_val.json
    """
    try:
        latest_report_dir = get_latest_run_dir(REPORTS_DIR)
        path = latest_report_dir / "metrics_val.json"

        logger.info(f"Lecture metrics_val.json : {path}")
        metrics_val = read_json(path)

        return ModelMetricsResponse(
            report_run_id=latest_report_dir.name,
            metrics_val=metrics_val,
        )

    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON invalide: {e}")
        raise HTTPException(status_code=500, detail=f"JSON invalide: {e}") from e
    except Exception as e:
        logger.exception("Erreur inattendue dans /model/metrics")
        raise HTTPException(status_code=500, detail="Erreur interne serveur") from e


@app.get("/model/classification", response_model=ModelClassificationResponse)
async def get_model_classification_report(_=Depends(require_admin)):
    """
    Retourne le classification report du dernier report:
    reports/<latest>/classification_report_val.txt
    """
    try:
        latest_report_dir = get_latest_run_dir(REPORTS_DIR)
        path = latest_report_dir / "classification_report_val.txt"

        logger.info(f"Lecture classification_report_val.txt : {path}")
        report_txt = read_text(path)

        return ModelClassificationResponse(
            report_run_id=latest_report_dir.name,
            classification_report_val=report_txt,
        )

    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Erreur inattendue dans /model/classification")
        raise HTTPException(status_code=500, detail="Erreur interne serveur") from e


@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...), _=Depends(require_admin)) -> Dict[str, Any]:
    """
    Upload d'un batch CSV (simulation admin/user), validation, append au dataset d'entraînement,
    puis exécution de la chaîne preprocessing -> transformation -> training -> evaluation.
    """
    logger.info("Requête /ingest reçue")

    # 0) basic checks
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .csv")

    create_directories([UPLOADS_DIR])

    uploads_path = UPLOADS_DIR / file.filename

    # 1) Save incoming file
    try:
        with uploads_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Fichier uploadé sauvegardé dans : {uploads_path}")
    except Exception as e:
        logger.exception("Erreur lors de la sauvegarde du fichier uploadé")
        raise HTTPException(
            status_code=500, detail="Erreur lors de la sauvegarde du fichier"
        ) from e
    finally:
        await file.close()

    # 2) Validate
    try:
        validate_uploaded_batch_csv(uploads_path)
        logger.success("Validation du CSV OK")
    except Exception as e:
        logger.error(f"Validation échouée: {e}")
        raise HTTPException(status_code=422, detail=f"CSV invalide: {e}") from e

    # 3) Ingestion (append au dataset courant) + training chain
    try:
        ingestion_pipeline = DataIngestionPipeline()
        ingested_dataset_path = ingestion_pipeline.run(uploaded_csv_path=uploads_path)

        preprocessing_path = DataPreprocessingPipeline().run()
        transformation_path = DataTransformationPipeline().run()
        model_path = ModelTrainerPipeline().run()
        metrics_path = ModelEvaluationPipeline().run()

        return {
            "status": "ok",
            "uploaded_file": str(uploads_path),
            "ingested_dataset": str(ingested_dataset_path),
            "preprocessed": str(preprocessing_path),
            "transformed": str(transformation_path),
            "model": str(model_path),
            "metrics": str(metrics_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur lors du pipeline ingestion+train")
        raise HTTPException(status_code=500, detail="Erreur pipeline ingestion+train") from e
    

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)