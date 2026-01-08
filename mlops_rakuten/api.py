from __future__ import annotations

import json
import yaml
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
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
    PROJ_ROOT
)
from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.utils import create_directories

REQUIRED_COLUMNS = ["designation", "prdtypecode"]


app = FastAPI(
    title="Rakuten Product Classification API",
    description="API d'inférence pour la classification de produits Rakuten.",
    version="0.1.0",
)

def get_dvc_lock_info() -> Dict[str, Any]:
    """Récupère les infos du dvc.lock"""
    dvc_lock_path = PROJ_ROOT / "dvc.lock"
    
    try:
        if dvc_lock_path.exists():
            with open(dvc_lock_path, 'r') as f:
                dvc_lock = yaml.safe_load(f)
            
            # Extraire les hashes des outputs
            stages_info = {}
            for stage_name, stage_data in dvc_lock.items():
                if 'outs' in stage_data:
                    stages_info[stage_name] = {
                        'outputs': stage_data['outs'],
                        'timestamp': stage_data.get('cmd', '').split()[-1] if stage_data.get('cmd') else None
                    }
            
            return {
                'dvc_lock_path': str(dvc_lock_path),
                'stages': stages_info,
                'last_modified': dvc_lock_path.stat().st_mtime
            }
    except Exception as e:
        logger.error(f"Erreur lecture dvc.lock: {e}")
    
    return {}


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


# Endpoints
@app.get("/health")
async def healthcheck(_=Depends(require_user)):
    return {"status": "ok"}


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
        latest_model_dir = MODELS_DIR
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
        latest_report_dir = REPORTS_DIR
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
        latest_report_dir = REPORTS_DIR
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

@app.get("/dvc/lock")
async def get_dvc_lock(_=Depends(require_admin)):
    """
    Retourne le contenu du dvc.lock (hashes de tous les outputs)
    
    Utile pour:
      - Vérifier la reproductibilité
      - Voir les hashes des versions
      - Déboguer les cachés DVC
    """
    logger.info("Requête /dvc/lock reçue")
    
    dvc_lock_path = PROJ_ROOT / "dvc.lock"
    
    try:
        if not dvc_lock_path.exists():
            raise FileNotFoundError("dvc.lock not found")
        
        with open(dvc_lock_path, 'r') as f:
            dvc_lock = yaml.safe_load(f)
        
        return {
            'path': str(dvc_lock_path),
            'content': dvc_lock
        }
    
    except Exception as e:
        logger.error(f"Erreur lecture dvc.lock: {e}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/dvc/dag")
async def get_dvc_dag(_=Depends(require_user)):
    """
    Retourne le DAG du pipeline DVC
    """
    logger.info("Requête /dvc/dag reçue")
    
    try:
        result = subprocess.run(
            ["dvc", "dag"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0: 
            return {"dag": result.stdout} 
        else: raise Exception(result.stderr or "dvc dag failed") 
    except Exception as e: 
        logger.error(f"Erreur dvc dag: {e}") 
        raise HTTPException(status_code=500, detail=str(e))

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

    # 3) Ingestion + DVC repro (détecte changements)
    try:
        # Ingérer les données
        ingestion_pipeline = DataIngestionPipeline()
        ingested_dataset_path = ingestion_pipeline.run(uploaded_csv_path=uploads_path)
        logger.success(f"Ingestion complète: {ingested_dataset_path}")

        # Relancer le pipeline DVC (détecte changement rakuten_train_current.csv)
        logger.info("Lancement dvc repro...")
        result = subprocess.run(
            ["dvc", "repro"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"dvc repro failed: {result.stderr}")
            raise Exception(f"dvc repro failed: {result.stderr}")
        
        logger.success("dvc repro complète")


        return {
            "status": "ok",
            "message": "Ingestion + retraining complète via dvc repro",
            "uploaded_file": str(uploads_path),
            "ingested_dataset": str(ingested_dataset_path),
            "pipeline_status": "completed via dvc repro",
            "dvc_lock": str(PROJ_ROOT / "dvc.lock")
        }

    except Exception as e:
        logger.exception("Erreur ingestion + dvc repro")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}") from e