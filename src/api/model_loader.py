import os
import sys
from pathlib import Path

# Permite importar walmart_sales_forecast estando en src/api/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow

from walmart_sales_forecast.config import MODELS_DIR
from walmart_sales_forecast.features import Preprocesador
from walmart_sales_forecast.tracking import MLflowRegistryManager

MODEL_NAME = "Walmart_RandomForest"
ALIAS_PRODUCCION = "produccion"
TRACKING_URI = "sqlite:///mlflow.db"

# el Model Registry.
IS_DOCKER = os.environ.get("ENVIRONMENT") == "docker"

if not IS_DOCKER:
    # mlflow.sklearn.load_model() usa esta URI global
    mlflow.set_tracking_uri(TRACKING_URI)
    _registryMgr = MLflowRegistryManager(trackingUri=TRACKING_URI)


def cargarModelo():
    """Trae el modelo en 'produccion': del Registry en local, del .pkl en Docker."""
    if IS_DOCKER:
        try:
            return joblib.load(MODELS_DIR / "modelo_random_forest.pkl")
        except Exception:  # noqa: BLE001
            return None
    try:
        return _registryMgr.cargarModeloPorAlias(
            MODEL_NAME, alias=ALIAS_PRODUCCION, tipoModelo="sklearn"
        )
    except Exception:  # noqa: BLE001
        return None


def obtenerMetadataModelo() -> dict:
    """Version y run_id del modelo actual, para trazabilidad."""
    if IS_DOCKER:
        return {"version": "docker-pkl", "run_id": "n/a"}
    try:
        mv = _registryMgr.client.get_model_version_by_alias(MODEL_NAME, ALIAS_PRODUCCION)
        return {"version": str(mv.version), "run_id": mv.run_id}
    except Exception:  # noqa: BLE001
        return {"version": "Desconocida", "run_id": "Desconocido"}


def obtenerPreprocesadorEntrenado() -> Preprocesador:
    """Carga el scaler ya ajustado en entrenamiento"""
    preprocesador = Preprocesador()
    preprocesador.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return preprocesador
