"""
Component pour créer les versions de données
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from mlops_rakuten.entities import DataVersioningConfig
from mlops_rakuten.utils import create_directories


class DataVersioning:
    """Crée une version du dataset avec drift simulé"""
    
    def __init__(self, config: DataVersioningConfig) -> None:
        self.config = config
    
    def run(self) -> Path:
        """Crée la version et retourne le répertoire de sortie"""
        cfg = self.config
        logger.info(f" Création de la version {cfg.version_name}")
        
        # 1. Charger les données sources
        logger.info(" Chargement du dataset source")
        X = pd.read_csv(cfg.source_x_train_path)
        Y = pd.read_csv(cfg.source_y_train_path)
        df_full = X.merge(Y, left_index=True, right_index=True)
        
        # Mélanger
        df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"   Total: {len(df_full)} samples, "
                   f"{df_full['prdtypecode'].nunique()} catégories")
        
        # 2. Extraire la portion pour cette version
        n_version = int(len(df_full) * cfg.split_ratio)
        df_version = df_full.iloc[:n_version].copy()
        
        logger.info(f"   Extraction: {cfg.split_ratio*100:.0f}% → {n_version} samples")
        
        # 3. Appliquer le drift
        if cfg.apply_drift:
            logger.info(f"Application du drift: {cfg.apply_drift}")
            if cfg.apply_drift == "light":
                df_version = self._apply_light_drift(df_version)
            elif cfg.apply_drift == "strong":
                df_version = self._apply_strong_drift(df_version)
        
        # 5. Sauvegarder
        self._save_version(df_version)
        
        logger.success(f"Version {cfg.version_name} créée: {len(df_version)} samples")
        
        return cfg.output_x_path.parent
    
    def _apply_light_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drift léger: +15% sur top 3 catégories"""
        top_cats = df['prdtypecode'].value_counts().head(3).index
        
        dfs = []
        for cat in df['prdtypecode'].unique():
            df_cat = df[df['prdtypecode'] == cat].copy()
            
            if cat in top_cats:
                n_samples = int(len(df_cat) * 1.15)
                df_cat = df_cat.sample(
                    n=min(n_samples, len(df_cat)),
                    replace=True,
                    random_state=42
                )
            dfs.append(df_cat)
        
        return pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    def _apply_strong_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drift fort: -40% populaires, +50% rares"""
        cat_counts = df['prdtypecode'].value_counts()
        popular = cat_counts.head(5).index
        rare = cat_counts.tail(5).index
        
        dfs = []
        for cat in df['prdtypecode'].unique():
            df_cat = df[df['prdtypecode'] == cat].copy()
            
            if cat in popular:
                n_samples = int(len(df_cat) * 0.6)
                df_cat = df_cat.sample(n=n_samples, random_state=42)
            elif cat in rare:
                n_samples = int(len(df_cat) * 1.5)
                df_cat = df_cat.sample(
                    n=min(n_samples, len(df_cat)),
                    replace=True,
                    random_state=42
                )
            dfs.append(df_cat)
        
        return pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    def _save_version(self, df: pd.DataFrame):
        """Sauvegarde la version avec métadonnées"""
        cfg = self.config
        
        # Créer le répertoire
        version_dir = cfg.output_x_path.parent
        create_directories([version_dir])
        
        # Séparer X et Y
        X_cols = ['designation', 'description', 'productid', 'imageid']
        X_version = df[X_cols].copy()
        Y_version = df[['prdtypecode']].copy()
        
        # Sauvegarder CSV
        X_version.to_csv(cfg.output_x_path)
        Y_version.to_csv(cfg.output_y_path)
        
        # Métadonnées
        metadata = {
            'version': cfg.version_name,
            'created_at': datetime.now().isoformat(),
            'description': cfg.description,
            'split_ratio': cfg.split_ratio,
            'drift_applied': cfg.apply_drift,
            'n_samples': len(df),
            'n_categories': df['prdtypecode'].nunique(),
            'category_distribution': df['prdtypecode'].value_counts().to_dict(),
            'top_5_categories': df['prdtypecode'].value_counts().head(5).to_dict(),
            'text_stats': {
                'designation_length_mean': float(df['designation'].str.len().mean()),
                'missing_description_rate': float(df['description'].isna().mean()),
            }
        }
        
        with open(cfg.output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # README
        readme = f"""# Data Version {cfg.version_name}

## Description
{cfg.description}

## Statistics
- **Samples**: {len(df):,}
- **Categories**: {df['prdtypecode'].nunique()}
- **Split ratio**: {cfg.split_ratio*100:.0f}%
- **Drift applied**: {cfg.apply_drift}

## Top 5 Categories
{df['prdtypecode'].value_counts().head(5).to_string()}

## Files
- `X_train.csv`: Features (designation, description, IDs)
- `Y_train.csv`: Target (prdtypecode)
- `metadata.json`: Detailed statistics

## Usage
```python
import pandas as pd

X = pd.read_csv('X_train.csv')
Y = pd.read_csv('Y_train.csv')
df = X.merge(Y, left_index=True, right_index=True)
```
"""
        
        with open(version_dir / 'README.md', 'w') as f:
            f.write(readme)
