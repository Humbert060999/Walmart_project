"""
UBICACIÓN: walmart_forecast/pipeline.py
(archivo NUEVO en la raíz del paquete, al mismo nivel que dataset.py,
 features.py, metrics.py)

Clase Pipeline: junta todas las piezas anteriores y expone un único
punto de entrada. Es la misma clase "Pipeline" del diagrama.
"""

import pandas as pd

from walmart_sales_forecast.config import config
from walmart_sales_forecast.dataset import DataIntegrador, DataLoader
from walmart_sales_forecast.features import Preprocesador
from walmart_sales_forecast.metrics import Evaluador
from walmart_sales_forecast.modeling.models import ModeloARIMA, ModeloRandomForest
from walmart_sales_forecast.tracking import MLflowRegistryManager, MLflowTracker


class Pipeline:
    """Orquesta todo el flujo: carga -> integra -> preprocesa -> entrena -> evalúa."""

    def __init__(self):
        self.cargador = DataLoader()
        self.integrador = DataIntegrador()
        self.preprocesador = Preprocesador()
        self.evaluador = Evaluador()
        self.config = config
        self.tracker = MLflowTracker(experimentName="Walmart_Sales_Forecast")
        self.registryMgr = MLflowRegistryManager()

    def preparar_datos(self) -> pd.DataFrame:
        train = self.cargador.cargar_train()
        features = self.cargador.cargar_features()
        stores = self.cargador.cargar_stores()

        dataset = self.integrador.combinar_datasets(train, features, stores)
        dataset_procesado = self.preprocesador.preprocesar(dataset, ajustar_scaler=True)
        return dataset_procesado.dropna()

    def entrenar_y_evaluar(self, modelo, columnas_excluidas=None):
        if columnas_excluidas is None:
            columnas_excluidas = ["Weekly_Sales", "Date", "Peso_WMAE"]

        dataset = self.preparar_datos()
        train_split, val_split = self.preprocesador.dividir_train_val(dataset)

        X_train = train_split.drop(columns=columnas_excluidas)
        y_train = train_split["Weekly_Sales"]
        X_val = val_split.drop(columns=columnas_excluidas)
        y_val = val_split["Weekly_Sales"]
        pesos_val = val_split["Peso_WMAE"]

        modelo.entrenar(X_train, y_train)
        predicciones = modelo.predecir(X_val)

        wmae = self.evaluador.calcular_wmae(y_val, predicciones, pesos_val)
        return modelo, wmae

    def comparar_modelos(self, registrarEnMlflow: bool = True):
        resultados = []

        # --- Random Forest ---
        params_rf = {
            "n_estimadores": self.config.obtener("n_estimadores_rf"),
            "profundidad_maxima": self.config.obtener("profundidad_maxima_rf"),
        }
        modelo_rf = ModeloRandomForest(**params_rf)
        modelo_rf, wmae_rf = self.entrenar_y_evaluar(modelo_rf)
        resultados.append(("RandomForest", wmae_rf))

        if registrarEnMlflow:
            self.tracker.loguearCorrida(
                modelo=modelo_rf,
                params=params_rf,
                metricas={"wmae": wmae_rf},
                nombreCorrida="RandomForest",
                tipoModelo="sklearn",
                registrarComo="Walmart_RandomForest",
            )

        # ARIMA se evalúa distinto (serie temporal por Store-Dept);
        # aquí un ejemplo simplificado con una sola serie agregada.
        dataset = self.preparar_datos()
        serie_agregada = dataset.groupby("Date")["Weekly_Sales"].sum().sort_index()
        train_serie = serie_agregada.iloc[:-10]
        val_serie = serie_agregada.iloc[-10:]

        orden_arima = self.config.obtener("orden_arima")
        modelo_arima = ModeloARIMA(orden_pdq=orden_arima)
        modelo_arima.entrenar(X=None, y=train_serie)
        pred_arima = modelo_arima.predecir(val_serie)
        pesos_arima = [1] * len(val_serie)  # simplificado; usa Peso_WMAE real si lo tienes por fecha

        wmae_arima = self.evaluador.calcular_wmae(val_serie.values, pred_arima, pesos_arima)
        resultados.append(("ARIMA", wmae_arima))

        if registrarEnMlflow:
            self.tracker.loguearCorrida(
                modelo=modelo_arima,
                params={"orden_pdq": str(orden_arima)},
                metricas={"wmae": wmae_arima},
                nombreCorrida="ARIMA",
                tipoModelo="statsmodels",
                registrarComo="Walmart_ARIMA",
            )

        tabla = self.evaluador.comparar_modelos(resultados)

        if registrarEnMlflow:
            self._promoverGanador(tabla)

        return tabla

    def _promoverGanador(self, tabla) -> None:
        """Marca con el alias 'produccion' la version del modelo con menor WMAE."""
        nombreGanador = tabla.iloc[0]["Modelo"]
        nombreRegistrado = {
            "RandomForest": "Walmart_RandomForest",
            "ARIMA": "Walmart_ARIMA",
        }[nombreGanador]

        version = self.registryMgr.obtenerUltimaVersion(nombreRegistrado)
        self.registryMgr.promoverModelo(nombreRegistrado, version, alias="produccion")


if __name__ == "__main__":
    # Ejecuta con: python -m walmart_forecast.pipeline
    pipeline = Pipeline()
    tabla_comparacion = pipeline.comparar_modelos()
    print(tabla_comparacion)
