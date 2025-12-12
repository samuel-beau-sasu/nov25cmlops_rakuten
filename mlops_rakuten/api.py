from typing import List, Optional

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from mlops_rakuten.pipelines.prediction import PredictionPipeline

app = FastAPI(
    title="Rakuten Product Classification API",
    description="API d'inférence pour la classification de produits Rakuten.",
    version="0.1.0",
)


# Initialisation Pipeline

# On charge le pipeline une seule fois au démarrage du serveur.
logger.info("Initialisation globale du PredictionPipeline FastAPI")
prediction_pipeline = PredictionPipeline()
logger.success("PredictionPipeline initialisé")


# Schemas Pydantic


class PredictionRequest(BaseModel):
    """
    Requête d'inférence pour un produit Rakuten.
    """

    designation: str = Field(
        min_length=10, description="Designation produit (au moins 10 caractères)."
    )
    top_k: int | None = 5  # nombre de catégories à retourner (None = toutes)

    @field_validator("designation")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("La description doit contenir au moins 10 caractères non vides.")
        return v


class CategoryScore(BaseModel):
    """
    Une catégorie prédite avec son code, son nom et sa probabilité.
    """

    prdtypecode: int
    category_name: Optional[str]
    proba: float


class PredictionResponse(BaseModel):
    """
    Réponse d'inférence pour un texte.
    """

    designation: str
    predictions: List[CategoryScore]


# Endpoints


@app.get("/health")
async def healthcheck():
    """
    Endpoint de healthcheck simple.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prend une description produit en entrée, retourne les top_k catégories
    avec leurs probabilités.
    """
    logger.info("Requête /predict reçue")

    # PredictionPipeline.run attend une liste de textes
    results_per_text = prediction_pipeline.run(
        texts=[request.designation],
        top_k=request.top_k,
    )

    # Comme on n'a qu'un texte, on récupère le premier élément
    preds_raw = results_per_text[0]

    # On adapte en objets Pydantic
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
