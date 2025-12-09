from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import pandas as pd
import io
import logging
from datetime import datetime
import os
import traceback
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Data Loading API',
    description='API pour charger des données de training (admin) et de test (utilisateurs)',
    version='1.0.0'
)

# Sécurité HTTP Basic
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

class LoadResponse(BaseModel):
    status: str
    message: str
    file_info: Optional[FileInfo] = None
    
# NOUVEAU : Modèle étendu avec prédictions
class LoadAndTrainResponse(LoadResponse):
    training_completed: bool = Field(default=False)
    prediction_results: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    next_steps: Optional[List[str]] = None

class FileStatus(BaseModel):
    has_training_data: bool
    has_test_data: bool
    training_files: List[str]
    test_files: List[str]

# Stockage des fichiers (en mémoire - à remplacer par une base de données en production)
training_data = {
    "X_train": None,
    "Y_train": None
}

# Dictionnaire pour stocker les fichiers de test par utilisateur
user_test_files: Dict[str, Dict] = {}

# Fonctions d'authentification
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    if username not in USERS_DB:
        raise HTTPException(
            status_code=401,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if USERS_DB[username] != password:
        raise HTTPException(
            status_code=401,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    if username not in ADMIN_CREDENTIALS:
        raise HTTPException(
            status_code=401,
            detail="Accès administrateur requis",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if ADMIN_CREDENTIALS[username] != password:
        raise HTTPException(
            status_code=401,
            detail="Accès administrateur requis",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

# Fonction utilitaire pour valider les fichiers CSV
def validate_csv(file: UploadFile, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Lit et valide un fichier CSV"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")
    
    # Lire le contenu
    content = file.file.read()
    
    try:
        # Lire le CSV
        df = pd.read_csv(io.BytesIO(content))
        
        # Vérifier que le fichier n'est pas vide
        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier CSV est vide")
        
        # Vérifier les colonnes requises si spécifiées
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
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est mal formaté")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture du CSV: {str(e)}")

@app.get('/')
def get_root():
    return {
        "message": "API de chargement de données",
        "endpoints": {
            "admin": {
                "load_training": "POST /admin/load-training",
                "training_status": "GET /admin/training-status"
            },
            "user": {
                "upload_test": "POST /upload-test",
                "list_files": "GET /user/files"
            },
            "public": {
                "status": "GET /status",
                "health": "GET /health"
            }
        }
    }

"""
Outil utilisant le health
Prometheus/Grafana : Métriques et alertes
Kubernetes : Vérifie si le conteneur est prêt (liveness/readiness probes)
AWS ELB/ALB : Health checks pour les load balancers
Datadog, New Relic : Monitoring d'application
"""
@app.get("/health")
async def health_check():
    """Endpoint de santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "data_loading_api"
    }


# Indique si des fichiers de training on ete chargés
@app.get("/status")
async def system_status():
    """Statut général du système"""
    has_x_train = training_data["X_train"] is not None
    has_y_train = training_data["Y_train"] is not None
    
    return {
        "training_data_loaded": has_x_train and has_y_train,
        "x_train_loaded": has_x_train,
        "y_train_loaded": has_y_train,
        "x_train_info": {
            "rows": len(training_data["X_train"]) if has_x_train else 0,
            "columns": list(training_data["X_train"].columns) if has_x_train else []
        } if has_x_train else None,
        "total_test_files": len(user_test_files),
        "active_users": list(user_test_files.keys())
    }

# ==================== ENDPOINTS ADMIN ====================

@app.post("/admin/load-training", response_model=LoadResponse)
async def load_training_data(
    x_train_file: UploadFile = File(..., description="Fichier X_train.csv avec les features"),
    y_train_file: UploadFile = File(..., description="Fichier Y_train.csv avec les labels"),
    admin: str = Depends(authenticate_admin)
    ):
    """
    Charge les données d'entraînement (admin uniquement).
    
    - X_train.csv : doit contenir les colonnes 'designation', 'description', 'productid', 'imageid'
    - Y_train.csv : doit contenir la colonne 'prdtypecode'
    """
    try:
        logger.info(f"Chargement des données d'entraînement par l'admin: {admin}")
        
        # Valider et lire X_train.csv
        logger.info(f"Validation de {x_train_file.filename}...")
        x_train_df = validate_csv(
            x_train_file, 
            required_columns=['designation', 'description', 'productid', 'imageid']
        )
        
        # Valider et lire Y_train.csv
        logger.info(f"Validation de {y_train_file.filename}...")
        y_train_df = validate_csv(
            y_train_file,
            required_columns=['prdtypecode']
        )
        
        # Vérifier la correspondance des lignes
        if len(x_train_df) != len(y_train_df):
            raise HTTPException(
                status_code=400,
                detail=f"Les fichiers n'ont pas le même nombre de lignes. X_train: {len(x_train_df)}, Y_train: {len(y_train_df)}"
            )
        
        # Stocker les données
        training_data["X_train"] = x_train_df
        training_data["Y_train"] = y_train_df
        
        # Calculer la taille des fichiers
        x_train_size = len(x_train_file.file.read()) / 1024 if hasattr(x_train_file.file, 'read') else 0
        
        file_info = FileInfo(
            filename=f"{x_train_file.filename}, {y_train_file.filename}",
            upload_time=datetime.now().isoformat(),
            rows=len(x_train_df),
            columns=list(x_train_df.columns) + list(y_train_df.columns),
            file_size_kb=round(x_train_size, 2)
        )
        
        logger.info(f"Données d'entraînement chargées avec succès: {len(x_train_df)} lignes")
        
        
        return LoadResponse(
            status="success",
            message=f"Données d'entraînement chargées avec succès. {len(x_train_df)} échantillons.",
            file_info=file_info
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

#@app.post("/admin/load-and-train", response_model=LoadAndTrainResponse)
async def load_and_train_data_new(
    x_train_file: UploadFile = File(..., description="Fichier X_train.csv"),
    y_train_file: UploadFile = File(..., description="Fichier Y_train.csv"),
    admin: str = Depends(authenticate_admin)
    ):
    """
    Charge les données ET entraîne le modèle en une seule opération.
    """
    # 1. Chargement des données (même code que avant)
    logger.info(f"Validation de {x_train_file.filename}...")
    x_train_df = validate_csv(x_train_file, required_columns=['designation', 'description', 'productid', 'imageid'])
    logger.info(f"Validation de {y_train_file.filename}...")
    y_train_df = validate_csv(y_train_file, required_columns=['prdtypecode'])
    
    # Vérifier la correspondance des lignes
        if len(x_train_df) != len(y_train_df):
            raise HTTPException(
                status_code=400,
                detail=f"Les fichiers n'ont pas le même nombre de lignes. X_train: {len(x_train_df)}, Y_train: {len(y_train_df)}"
            )

    # Stocker les données
    training_data["X_train"] = x_train_df
    training_data["Y_train"] = y_train_df
    
    # Calculer la taille des fichiers
    x_train_size = len(x_train_file.file.read()) / 1024 if hasattr(x_train_file.file, 'read') else 0
    
    # 2. Entraînement du modèle
    #obj = PredictionPipeline()
    #prediction = obj.predict(data)
    
    
    # 4. Retour combiné
    return LoadAndTrainResponse(
        status="success",
        message=f"✅ Données chargées ({len(x_train_df)} échantillons) et modèle entraîné avec succès",
        file_info=FileInfo(
            filename=f"{x_train_file.filename}, {y_train_file.filename}",
            upload_time=datetime.now().isoformat(),
            rows=len(x_train_df),
            columns=list(x_train_df.columns) + list(y_train_df.columns),
            file_size_kb=round(x_train_size, 2)
        ),
        training_completed=True,
        prediction_results={
            "samples_processed": len(x_train_df),
            "classes_predicted": 10,  # exemple
            "distribution": {"class_1": 100, "class_2": 50}  # exemple
        },
        model_info={
            "type": "RandomForestClassifier",
            "features_used": 1000,
            "training_date": datetime.now().isoformat()
        },
        next_steps=[
            "Le modèle est prêt pour les prédictions",
            "Utilisez /upload-test pour charger des données de test",
            "Utilisez /predict pour obtenir des prédictions"
        ]
    )

@app.get("/admin/training-status", response_model=FileStatus)
async def get_training_status(admin: str = Depends(authenticate_admin)):
    """
    Affiche le statut des données d'entraînement chargées.
    """
    has_x_train = training_data["X_train"] is not None
    has_y_train = training_data["Y_train"] is not None
    
    training_files = []
    if has_x_train:
        training_files.append("X_train.csv")
    if has_y_train:
        training_files.append("Y_train.csv")
    
    # Compter les fichiers de test par utilisateur
    test_files = []
    for user, data in user_test_files.items():
        test_files.append(f"{data['filename']} (user: {user})")
    
    return FileStatus(
        has_training_data=has_x_train and has_y_train,
        has_test_data=len(user_test_files) > 0,
        training_files=training_files,
        test_files=test_files
    )

@app.get("/admin/training-stats")
async def get_training_stats(admin: str = Depends(authenticate_admin)):
    """
    Statistiques détaillées sur les données d'entraînement.
    """
    if training_data["X_train"] is None or training_data["Y_train"] is None:
        raise HTTPException(
            status_code=404,
            detail="Aucune donnée d'entraînement chargée"
        )
    
    x_df = training_data["X_train"]
    y_df = training_data["Y_train"]
    
    # Statistiques de base
    stats = {
        "x_train_stats": {
            "rows": len(x_df),
            "columns": len(x_df.columns),
            "column_names": list(x_df.columns),
            "missing_values": x_df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in x_df.dtypes.items()}
        },
        "y_train_stats": {
            "rows": len(y_df),
            "unique_classes": y_df['prdtypecode'].nunique(),
            "class_distribution": y_df['prdtypecode'].value_counts().head(10).to_dict(),
            "dtype": str(y_df['prdtypecode'].dtype)
        }
    }
    
    # Statistiques textuelles (si colonnes texte présentes)
    if 'designation' in x_df.columns:
        stats["text_stats"] = {
            "designation": {
                "unique_count": x_df['designation'].nunique(),
                "avg_length": x_df['designation'].str.len().mean(),
                "max_length": x_df['designation'].str.len().max(),
                "min_length": x_df['designation'].str.len().min()
            }
        }
    
    return stats

# ==================== ENDPOINTS UTILISATEUR ====================

@app.post("/upload-test", response_model=LoadResponse)
async def upload_test_file(
    test_file: UploadFile = File(..., description="Fichier X_test.csv pour les prédictions"),
    user: str = Depends(authenticate_user)
    ):
    """
    Charge un fichier de test (utilisateurs authentifiés).
    
    - Le fichier doit contenir au minimum les colonnes 'designation' et 'productid'
    """
    try:
        logger.info(f"Chargement d'un fichier de test par l'utilisateur: {user}")
        
        # Valider et lire le fichier
        test_df = validate_csv(
            test_file,
            required_columns=['designation', 'productid']
        )
        
        # Vérifier la présence optionnelle d'autres colonnes
        optional_columns = ['description', 'imageid']
        missing_optional = [col for col in optional_columns if col not in test_df.columns]
        if missing_optional:
            logger.warning(f"Colonnes optionnelles manquantes pour {user}: {missing_optional}")
        
        # Stocker le fichier pour cet utilisateur
        file_content = test_file.file.read()
        file_size_kb = len(file_content) / 1024
        
        user_test_files[user] = {
            "filename": test_file.filename,
            "data": test_df,
            "upload_time": datetime.now().isoformat(),
            "file_size_bytes": len(file_content)
        }
        
        file_info = FileInfo(
            filename=test_file.filename,
            upload_time=datetime.now().isoformat(),
            rows=len(test_df),
            columns=list(test_df.columns),
            file_size_kb=round(file_size_kb, 2)
        )
        
        logger.info(f"Fichier de test chargé avec succès par {user}: {test_file.filename}")
        
        return LoadResponse(
            status="success",
            message=f"Fichier '{test_file.filename}' chargé avec succès. {len(test_df)} produits.",
            file_info=file_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

 
#@app.get("/user/files")     
async def get_user_files_old(user: str = Depends(authenticate_user)):
    """
    Liste les fichiers de test chargés par l'utilisateur.
    """
    if user not in user_test_files:
        return {
            "user": user,
            "files": [],
            "message": "Aucun fichier chargé"
        }
    
    user_file = user_test_files[user]
    
    # Information détaillée sur le fichier
    file_info = {
        "filename": user_file["filename"],
        "upload_time": user_file["upload_time"],
        "rows": len(user_file["data"]),
        "columns": list(user_file["data"].columns),
        "file_size_kb": round(user_file["file_size_bytes"] / 1024, 2),
        "sample_data": user_file["data"].head(3).to_dict(orient='records')
    }
    
    return {
        "user": user,
        "files": [file_info],
        "total_files": 1
    }

#@app.get("/user/files")
async def get_user_files_old2(user: str = Depends(authenticate_user)):
    """
    Liste les fichiers de test chargés par l'utilisateur.
    """
    try:
        logger.info(f"Accès à /user/files par l'utilisateur: {user}")
        
        if user not in user_test_files:
            logger.info(f"Utilisateur {user} n'a pas de fichiers")
            return {
                "user": user,
                "files": [],
                "message": "Aucun fichier chargé"
            }
        
        user_file = user_test_files[user]
        logger.info(f"Fichier trouvé pour {user}: {user_file['filename']}")
        
        # Information détaillée sur le fichier
        file_info = {
            "filename": user_file["filename"],
            "upload_time": user_file["upload_time"],
            "rows": len(user_file["data"]),
            "columns": list(user_file["data"].columns),
            "file_size_kb": round(user_file["file_size_bytes"] / 1024, 2),
            "sample_data": user_file["data"].head(3).to_dict(orient='records')
        }
        
        logger.info(f"Réponse préparée pour {user}")
        return {
            "user": user,
            "files": [file_info],
            "total_files": 1
        }
        
    except Exception as e:
        logger.error(f"Erreur dans /user/files pour {user}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne: {str(e)}"
        )

@app.get("/user/files")
async def get_user_files(user: str = Depends(authenticate_user)):
    """Liste les fichiers de test chargés par l'utilisateur."""
    if user not in user_test_files:
        return {
            "user": user,
            "files": [],
            "message": "Aucun fichier chargé"
        }
    
    user_file = user_test_files[user]
    df = user_file["data"]
    
    # Conversion safe des données
    sample_data = df.head(3).replace({np.nan: None}).to_dict(orient='records')
    
    # Information détaillée sur le fichier
    file_info = {
        "filename": user_file["filename"],
        "upload_time": user_file["upload_time"],
        "rows": len(df),
        "columns": list(df.columns),
        "file_size_kb": round(user_file["file_size_bytes"] / 1024, 2),
        "sample_data": sample_data,
        "missing_values_summary": {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        }
    }
    
    return {
        "user": user,
        "files": [file_info],
        "total_files": 1
    }

#@app.get("/user/test-stats")
async def get_test_stats_old(user: str = Depends(authenticate_user)):
    """
    Statistiques sur le fichier de test de l'utilisateur.
    """
    if user not in user_test_files:
        raise HTTPException(
            status_code=404,
            detail="Aucun fichier de test chargé"
        )
    
    test_df = user_test_files[user]["data"]
    
    stats = {
        "filename": user_test_files[user]["filename"],
        "basic_stats": {
            "total_rows": len(test_df),
            "total_columns": len(test_df.columns),
            "column_names": list(test_df.columns)
        },
        "column_info": {},
        "sample_records": test_df.head(5).to_dict(orient='records')
    }
    
    # Statistiques par colonne
    for column in test_df.columns:
        col_stats = {
            "dtype": str(test_df[column].dtype),
            "unique_values": test_df[column].nunique(),
            "missing_values": test_df[column].isnull().sum(),
            "missing_percentage": round((test_df[column].isnull().sum() / len(test_df)) * 100, 2)
        }
        
        # Stats spécifiques pour les colonnes numériques
        if pd.api.types.is_numeric_dtype(test_df[column]):
            col_stats.update({
                "min": float(test_df[column].min()),
                "max": float(test_df[column].max()),
                "mean": float(test_df[column].mean()),
                "std": float(test_df[column].std())
            })
        
        # Stats spécifiques pour les colonnes textuelles
        if pd.api.types.is_string_dtype(test_df[column]):
            col_stats.update({
                "avg_length": float(test_df[column].str.len().mean()),
                "max_length": int(test_df[column].str.len().max()),
                "min_length": int(test_df[column].str.len().min())
            })
        
        stats["column_info"][column] = col_stats
    
    return stats

@app.get("/user/test-stats")
async def get_test_stats(user: str = Depends(authenticate_user)):
    """
    Statistiques sur le fichier de test de l'utilisateur.
    """
    if user not in user_test_files:
        raise HTTPException(
            status_code=404,
            detail="Aucun fichier de test chargé"
        )
    
    test_df = user_test_files[user]["data"]
    
    stats = {
        "filename": user_test_files[user]["filename"],
        "basic_stats": {
            "total_rows": int(len(test_df)),  # ← Conversion explicite
            "total_columns": int(len(test_df.columns)),  # ← Conversion
            "column_names": list(test_df.columns)
        },
        "column_info": {},
        "sample_records": test_df.head(5).replace({np.nan: None}).to_dict(orient='records')
    }
    
    # Statistiques par colonne avec conversion des types numpy
    for column in test_df.columns:
        col_stats = {
            "dtype": str(test_df[column].dtype),
            "unique_values": int(test_df[column].nunique()),  # ← Conversion
            "missing_values": int(test_df[column].isnull().sum()),  # ← Conversion
            "missing_percentage": float(round((test_df[column].isnull().sum() / len(test_df)) * 100, 2))  # ← Conversion
        }
        
        # Stats spécifiques pour les colonnes numériques
        if pd.api.types.is_numeric_dtype(test_df[column]):
            # CONVERSION EXPLICITE de tous les types numpy
            col_stats.update({
                "min": float(test_df[column].min()) if not pd.isna(test_df[column].min()) else None,
                "max": float(test_df[column].max()) if not pd.isna(test_df[column].max()) else None,
                "mean": float(test_df[column].mean()) if not pd.isna(test_df[column].mean()) else None,
                "std": float(test_df[column].std()) if not pd.isna(test_df[column].std()) else None
            })
        
        # Stats spécifiques pour les colonnes textuelles
        if pd.api.types.is_string_dtype(test_df[column]):
            # CONVERSION EXPLICITE
            col_stats.update({
                "avg_length": float(test_df[column].str.len().mean()) if not test_df[column].empty else None,
                "max_length": int(test_df[column].str.len().max()) if not test_df[column].empty else None,
                "min_length": int(test_df[column].str.len().min()) if not test_df[column].empty else None
            })
        
        stats["column_info"][column] = col_stats
    
    return stats


# ==================== ENDPOINTS DE NETTOYAGE ====================

@app.delete("/admin/clear-training")
async def clear_training_data(admin: str = Depends(authenticate_admin)):
    """
    Supprime toutes les données d'entraînement chargées.
    """
    training_data["X_train"] = None
    training_data["Y_train"] = None
    
    logger.info(f"Données d'entraînement effacées par l'admin: {admin}")
    
    return {
        "status": "success",
        "message": "Données d'entraînement effacées avec succès"
    }

@app.delete("/user/clear-test")
async def clear_test_files(user: str = Depends(authenticate_user)):
    """
    Supprime les fichiers de test de l'utilisateur.
    """
    if user in user_test_files:
        filename = user_test_files[user]["filename"]
        del user_test_files[user]
        
        logger.info(f"Fichier de test effacé pour l'utilisateur: {user}")
        
        return {
            "status": "success",
            "message": f"Fichier '{filename}' effacé avec succès"
        }
    
    return {
        "status": "info",
        "message": "Aucun fichier à effacer"
    }