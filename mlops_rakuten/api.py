from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field, field_validator

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
from mlops_rakuten.utils import create_directories

REQUIRED_COLUMNS = ["designation", "prdtypecode"]


app = FastAPI(
    title="Rakuten Product Classification API",
    description="API d'inférence pour la classification de produits Rakuten.",
    version="0.1.0",
)


def get_latest_run_dir(parent_dir: Path) -> Path:
    """
    Retourne le sous-répertoire le plus récent via tri lexical.
    Suppose un naming ISO du type: YYYY-MM-DDTHH-MM-SS (ou similaire).
    """
    if not parent_dir.exists():
        raise FileNotFoundError(f"Répertoire inexistant: {parent_dir}")

    run_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Aucun run trouvé dans: {parent_dir}")

    return sorted(run_dirs)[-1]


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
async def healthcheck():
    return {"status": "ok"}


@app.get("/model/config", response_model=ModelConfigResponse)
async def get_model_config():
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
async def get_model_metrics():
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
async def get_model_classification_report():
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
async def ingest_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
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
