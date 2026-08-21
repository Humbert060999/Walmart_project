"""
UBICACIÓN: walmart_sales_forecast/tracking.py
(archivo NUEVO, tarea de implementación de MLflow)

Contiene MLflowTracker (registra corridas: params, metricas, modelo)
y MLflowRegistryManager (promueve el modelo ganador con el sistema
de alias actual de MLflow -- las "stages" viejas estan deprecadas).
"""

import subprocess

import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Registra corridas de entrenamiento (parametros, metricas, modelo) en MLflow."""

    def __init__(self, experimentName: str, trackingUri: str = "sqlite:///mlflow.db"):
        self.trackingUri = trackingUri
        mlflow.set_tracking_uri(trackingUri)
        mlflow.set_experiment(experimentName)

    def obtenerVersionDvc(self) -> str:
        """Hash corto de git, para trazabilidad de con que version de datos se entreno."""
        try:
            resultado = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return resultado.stdout.strip() or "sin-commit"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "version-desconocida"

    def loguearCorrida(
        self,
        modelo,
        params: dict,
        metricas: dict,
        nombreCorrida: str,
        tipoModelo: str,
        registrarComo: str | None = None,
    ) -> str:
        """Abre un run de MLflow y loguea todo: tags, params, metricas, modelo.

        tipoModelo: "sklearn" (RandomForest) o "statsmodels" (ARIMA) -- cada
        uno se loguea con el flavor correcto de MLflow.
        """
        with mlflow.start_run(run_name=nombreCorrida) as run:
            mlflow.set_tags({
                "proyecto": "Walmart_Sales_Forecast",
                "tipo_modelo": tipoModelo,
                "version_datos": self.obtenerVersionDvc(),
            })

            mlflow.log_params(params)
            mlflow.log_metrics(metricas)

            self._loguearModelo(modelo, tipoModelo, registrarComo)

            print(f"[MLflow] Run registrado: {run.info.run_id} | {metricas}")
            return run.info.run_id

    def _loguearModelo(self, modelo, tipoModelo: str, registrarComo: str | None) -> None:
        if tipoModelo == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=modelo.modelo,
                name="modelo",
                registered_model_name=registrarComo,
            )
        elif tipoModelo == "statsmodels":
            mlflow.statsmodels.log_model(
                statsmodels_model=modelo._resultado_ajuste,
                name="modelo",
                registered_model_name=registrarComo,
            )
        else:
            raise ValueError(f"tipoModelo desconocido: {tipoModelo}")


class MLflowRegistryManager:
    """Administra el ciclo de vida de los modelos en el Model Registry (API de alias)."""

    def __init__(self, trackingUri: str = "sqlite:///mlflow.db"):
        self.client = MlflowClient(tracking_uri=trackingUri)

    def obtenerUltimaVersion(self, nombreModelo: str) -> int:
        """Numero de la version mas reciente registrada para ese modelo."""
        versiones = self.client.search_model_versions(
            filter_string=f"name='{nombreModelo}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if not versiones:
            raise ValueError(f"No hay versiones registradas para '{nombreModelo}'")
        return int(versiones[0].version)

    def promoverModelo(self, nombreModelo: str, version: int, alias: str = "produccion") -> None:
        """Asigna un alias (ej. 'produccion') a una version especifica del modelo."""
        self.client.set_registered_model_alias(name=nombreModelo, alias=alias, version=version)
        print(f"[Registry] '{nombreModelo}' version {version} -> alias '{alias}'")

    def cargarModeloPorAlias(self, nombreModelo: str, alias: str = "produccion", tipoModelo: str = "sklearn"):
        """Carga el modelo marcado con ese alias, listo para inferencia."""
        uri = f"models:/{nombreModelo}@{alias}"
        if tipoModelo == "sklearn":
            return mlflow.sklearn.load_model(uri)
        if tipoModelo == "statsmodels":
            return mlflow.statsmodels.load_model(uri)
        raise ValueError(f"tipoModelo desconocido: {tipoModelo}")

    def compararCorridas(self, nombreExperimento: str):
        """Tabla de todas las corridas del experimento, para comparar metricas."""
        return mlflow.search_runs(experiment_names=[nombreExperimento])
