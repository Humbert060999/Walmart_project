"""
UBICACIÓN: walmart_sales_forecast/pipeline.py

Pipeline principal del proyecto Walmart.

Flujo Random Forest:

    carga
      ↓
    integración
      ↓
    feature engineering
      ↓
    split temporal
      ↓
    outliers calculados SOLO con train
      ↓
    scaler fit SOLO con train
      ↓
    scaler transform en validation
      ↓
    entrenamiento
      ↓
    predicción
      ↓
    WMAE
      ↓
    logueo en MLflow (params, metricas, modelo)

Flujo ARIMA:

    carga
      ↓
    agregación temporal
      ↓
    split temporal
      ↓
    entrenamiento ARIMA
      ↓
    predicción
      ↓
    Peso_WMAE real por fecha
      ↓
    WMAE
      ↓
    logueo en MLflow (params, metricas, modelo)

Al final de comparar_modelos(), el modelo con menor WMAE se
promueve automáticamente al alias 'produccion' en el Model Registry
de MLflow.
"""

import joblib
import pandas as pd

from walmart_sales_forecast.config import MODELS_DIR, config
from walmart_sales_forecast.dataset import (
    DataIntegrador,
    DataLoader,
)
from walmart_sales_forecast.features import Preprocesador
from walmart_sales_forecast.metrics import Evaluador
from walmart_sales_forecast.modeling.models import (
    ModeloARIMA,
    ModeloRandomForest,
)
from walmart_sales_forecast.tracking import (
    MLflowRegistryManager,
    MLflowTracker,
)
from walmart_sales_forecast.visualizador_entrenamiento import (
    VisualizadorEntrenamiento,
)

# Nombres con los que cada modelo queda registrado en el Model Registry
NOMBRES_REGISTRO_MLFLOW = {
    "RandomForest": "Walmart_RandomForest",
    "ARIMA": "Walmart_ARIMA",
}


class Pipeline:
    """
    Orquesta todo el flujo del proyecto.
    """

    def __init__(self):
        self.cargador = DataLoader()
        self.integrador = DataIntegrador()
        self.preprocesador = Preprocesador()
        self.evaluador = Evaluador()
        self.config = config
        self.tracker = MLflowTracker(experimentName="Walmart_Sales_Forecast")
        self.registryMgr = MLflowRegistryManager()
        self.visualizador = VisualizadorEntrenamiento()

    # =========================================================
    # CARGA E INTEGRACIÓN
    # =========================================================

    def cargar_datos(self) -> pd.DataFrame:
        """
        Carga e integra train, features y stores.
        """

        train = (
            self.cargador.cargar_train()
        )

        features = (
            self.cargador.cargar_features()
        )

        stores = (
            self.cargador.cargar_stores()
        )

        dataset = (
            self.integrador.combinar_datasets(
                train,
                features,
                stores,
            )
        )

        return dataset

    # =========================================================
    # PREPARACIÓN RANDOM FOREST
    # =========================================================

    def preparar_datos(self):
        """
        Prepara train y validation para Random Forest.

        Orden:

        1. Cargar datos.
        2. Crear features.
        3. Split temporal.
        4. Calcular outliers SOLO con train.
        5. Aplicar esos límites a train.
        6. Ajustar scaler SOLO con train.
        7. Aplicar transform a validation.
        """

        dataset = self.cargar_datos()

        # -----------------------------------------------------
        # Feature engineering
        # -----------------------------------------------------

        dataset = (
            self.preprocesador.crear_features(
                dataset
            )
        )

        # -----------------------------------------------------
        # Split temporal
        # -----------------------------------------------------

        train_split, val_split = (
            self.preprocesador.dividir_train_val(
                dataset
            )
        )

        # -----------------------------------------------------
        # Outliers
        #
        # Los límites se calculan EXCLUSIVAMENTE sobre train.
        # -----------------------------------------------------

        limites_outliers = (
            self.preprocesador.obtener_limites_outliers(
                train_split
            )
        )

        train_split = (
            self.preprocesador.tratar_outliers(
                train_split,
                limites=limites_outliers,
            )
        )

        # -----------------------------------------------------
        # Lags
        #
        # El primer registro de cada Store/Dept puede no tener
        # información histórica.
        # -----------------------------------------------------

        columnas_lag = [
            c
            for c in train_split.columns
            if c.startswith(
                "Ventas_Lag_"
            )
        ]

        if columnas_lag:
            train_split = (
                train_split.dropna(
                    subset=columnas_lag
                )
            )

            val_split = (
                val_split.dropna(
                    subset=columnas_lag
                )
            )

        # -----------------------------------------------------
        # El target tampoco puede ser nulo.
        # -----------------------------------------------------

        if "Weekly_Sales" in train_split.columns:
            train_split = (
                train_split.dropna(
                    subset=[
                        "Weekly_Sales"
                    ]
                )
            )

        if "Weekly_Sales" in val_split.columns:
            val_split = (
                val_split.dropna(
                    subset=[
                        "Weekly_Sales"
                    ]
                )
            )

        # -----------------------------------------------------
        # StandardScaler
        #
        # FIT SOLO EN TRAIN.
        # -----------------------------------------------------

        train_split = (
            self.preprocesador.ajustar_scaler(
                train_split
            )
        )

        # -----------------------------------------------------
        # VALIDATION:
        # solamente transform().
        # -----------------------------------------------------

        val_split = (
            self.preprocesador.transformar_scaler(
                val_split
            )
        )

        return train_split, val_split

    # =========================================================
    # ENTRENAR Y EVALUAR RANDOM FOREST
    # =========================================================

    def entrenar_y_evaluar(
        self,
        modelo,
        columnas_excluidas=None,
    ):
        """
        Entrena un modelo supervisado y calcula WMAE
        sobre validation.
        """

        if columnas_excluidas is None:
            columnas_excluidas = [
                "Weekly_Sales",
                "Date",
                "Peso_WMAE",
            ]

        train_split, val_split = (
            self.preparar_datos()
        )

        # -----------------------------------------------------
        # X / y TRAIN
        # -----------------------------------------------------

        X_train = (
            train_split.drop(
                columns=columnas_excluidas,
                errors="ignore",
            )
        )

        y_train = (
            train_split["Weekly_Sales"]
        )

        # -----------------------------------------------------
        # X / y VALIDATION
        # -----------------------------------------------------

        X_val = (
            val_split.drop(
                columns=columnas_excluidas,
                errors="ignore",
            )
        )

        y_val = (
            val_split["Weekly_Sales"]
        )

        pesos_val = (
            val_split["Peso_WMAE"]
        )

        # -----------------------------------------------------
        # Asegurar mismas columnas
        # -----------------------------------------------------

        X_val = X_val.reindex(
            columns=X_train.columns,
            fill_value=0,
        )

        # -----------------------------------------------------
        # Entrenamiento
        # -----------------------------------------------------

        modelo.entrenar(
            X_train,
            y_train,
        )

        # -----------------------------------------------------
        # Predicción
        # -----------------------------------------------------

        predicciones = (
            modelo.predecir(
                X_val
            )
        )

        # -----------------------------------------------------
        # WMAE
        # -----------------------------------------------------

        wmae = (
            self.evaluador.calcular_wmae(
                y_val,
                predicciones,
                pesos_val,
            )
        )

        return (
                modelo,
                wmae,
                val_split,
                predicciones,
        )

    # =========================================================
    # PESOS WMAE PARA ARIMA
    # =========================================================

    def obtener_pesos_por_fecha(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series:
        """
        Obtiene un Peso_WMAE por cada fecha.

        ARIMA trabaja con una serie agregada por fecha, por lo que
        necesitamos un único peso para cada fecha.

        Se toma el primer peso disponible para cada fecha porque
        IsHoliday es una característica de la semana y debe ser
        consistente entre las observaciones de esa fecha.
        """

        datos = dataset.copy()

        datos["Date"] = pd.to_datetime(
            datos["Date"]
        )

        pesos = (
            datos[
                [
                    "Date",
                    "Peso_WMAE",
                ]
            ]
            .drop_duplicates(
                subset=["Date"]
            )
            .set_index("Date")[
                "Peso_WMAE"
            ]
            .sort_index()
        )

        return pesos

    # =========================================================
    # ARIMA
    # =========================================================

    def evaluar_arima(self):
        """
        Entrena ARIMA sobre la serie agregada y calcula WMAE
        utilizando el Peso_WMAE real de las fechas de validation.
        """

        dataset = self.cargar_datos()

        # -----------------------------------------------------
        # Crear Peso_WMAE
        # -----------------------------------------------------

        dataset = (
            self.preprocesador.crear_features(
                dataset
            )
        )

        # -----------------------------------------------------
        # Serie agregada de ventas
        # -----------------------------------------------------

        serie_agregada = (
            dataset
            .groupby("Date")[
                "Weekly_Sales"
            ]
            .sum()
            .sort_index()
        )

        # -----------------------------------------------------
        # Pesos reales por fecha
        # -----------------------------------------------------

        pesos_por_fecha = (
            self.obtener_pesos_por_fecha(
                dataset
            )
        )

        # -----------------------------------------------------
        # Mantener únicamente fechas disponibles en ambas
        # estructuras.
        # -----------------------------------------------------

        fechas_comunes = (
            serie_agregada.index.intersection(
                pesos_por_fecha.index
            )
        )

        serie_agregada = (
            serie_agregada.loc[
                fechas_comunes
            ]
        )

        pesos_por_fecha = (
            pesos_por_fecha.loc[
                fechas_comunes
            ]
        )

        # -----------------------------------------------------
        # Split temporal ARIMA
        #
        # Se mantienen las últimas 10 semanas como validation,
        # igual que en el código original.
        # -----------------------------------------------------

        train_serie = (
            serie_agregada.iloc[:-10]
        )

        val_serie = (
            serie_agregada.iloc[-10:]
        )

        pesos_val = (
            pesos_por_fecha.iloc[-10:]
        )

        # -----------------------------------------------------
        # Entrenar ARIMA
        # -----------------------------------------------------

        orden_arima = self.config.obtener(
            "orden_arima"
        )

        modelo_arima = ModeloARIMA(
            orden_pdq=orden_arima
        )

        modelo_arima.entrenar(
            X=None,
            y=train_serie,
        )

        # -----------------------------------------------------
        # Predicción
        # -----------------------------------------------------

        pred_arima = (
            modelo_arima.predecir(
                val_serie
            )
        )

        # -----------------------------------------------------
        # WMAE con pesos reales
        # -----------------------------------------------------

        wmae_arima = (
            self.evaluador.calcular_wmae(
                val_serie.values,
                pred_arima,
                pesos_val.values,
            )
        )

        return (
                modelo_arima,
                wmae_arima,
                val_serie,
                pred_arima,
        )

    # =========================================================
    # MLFLOW: LOGUEO POR MODELO
    # =========================================================

    def _loguearRandomForest(self, modeloRf, wmaeRf: float) -> None:
        paramsRf = {
            "n_estimadores": self.config.obtener("n_estimadores_rf"),
            "profundidad_maxima": self.config.obtener("profundidad_maxima_rf"),
            "semilla_aleatoria": self.config.obtener("semilla_aleatoria"),
        }
        self.tracker.loguearCorrida(
            modelo=modeloRf,
            params=paramsRf,
            metricas={"wmae": wmaeRf},
            nombreCorrida="RandomForest",
            tipoModelo="sklearn",
            registrarComo=NOMBRES_REGISTRO_MLFLOW["RandomForest"],
        )

    def _loguearArima(self, modeloArima, wmaeArima: float) -> None:
        p, d, q = self.config.obtener("orden_arima")
        self.tracker.loguearCorrida(
            modelo=modeloArima,
            params={"p": p, "d": d, "q": q},
            metricas={"wmae": wmaeArima},
            nombreCorrida="ARIMA",
            tipoModelo="statsmodels",
            registrarComo=NOMBRES_REGISTRO_MLFLOW["ARIMA"],
        )

    def _promoverGanador(self, tabla) -> None:
        """Marca con el alias 'produccion' la version del modelo con menor WMAE."""
        nombreGanador = tabla.iloc[0]["Modelo"]
        nombreRegistrado = NOMBRES_REGISTRO_MLFLOW[nombreGanador]

        version = self.registryMgr.obtenerUltimaVersion(nombreRegistrado)
        self.registryMgr.promoverModelo(nombreRegistrado, version, alias="produccion")

    def _guardarModelosEnDisco(self, modeloRf, modeloArima) -> None:
        # Copia en models/ (para DVC); la API en produccion lee de MLflow,
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        modeloRf.guardar_modelo(MODELS_DIR / "modelo_random_forest.pkl")
        modeloArima.guardar_modelo(MODELS_DIR / "modelo_arima.pkl")
        joblib.dump(self.preprocesador.scaler, MODELS_DIR / "scaler.pkl")

    # =========================================================
    # COMPARACIÓN DE MODELOS
    # =========================================================

    def comparar_modelos(
        self,
        generar_graficos: bool = True,
        registrarEnMlflow: bool = True,
    ):
        """
        Compara Random Forest y ARIMA utilizando WMAE.

        Si generar_graficos=True, genera las visualizaciones
        correspondientes en reports/figures/.

        Si registrarEnMlflow=True, loguea cada corrida en MLflow y
        promueve automáticamente al alias 'produccion' el modelo
        con menor WMAE.
        """

        resultados = []

    # =====================================================
    # RANDOM FOREST
    # =====================================================

        modelo_rf = ModeloRandomForest(
            n_estimadores=self.config.obtener(
                "n_estimadores_rf"
            ),
            profundidad_maxima=self.config.obtener(
                "profundidad_maxima_rf"
            ),
            semilla_aleatoria=self.config.obtener(
                "semilla_aleatoria"
            ),
        )

        (
            modelo_rf,
            wmae_rf,
            val_rf,
            pred_rf,
        ) = self.entrenar_y_evaluar(
            modelo_rf
        )

        resultados.append(
            (
                "RandomForest",
                wmae_rf,
            )
        )

        if registrarEnMlflow:
            self._loguearRandomForest(modelo_rf, wmae_rf)

    # =====================================================
    # ARIMA
    # =====================================================

        (
            modelo_arima,
            wmae_arima,
            val_arima,
            pred_arima,
        ) = self.evaluar_arima()

        resultados.append(
            (
                "ARIMA",
                wmae_arima,
            )
        )

        if registrarEnMlflow:
            self._loguearArima(modelo_arima, wmae_arima)

    # =====================================================
    # TABLA DE RESULTADOS
    # =====================================================

        tabla_comparacion = (
            self.evaluador.comparar_modelos(
                resultados
            )
        )

        if registrarEnMlflow:
            self._promoverGanador(tabla_comparacion)

        self._guardarModelosEnDisco(modelo_rf, modelo_arima)

    # =====================================================
    # VISUALIZACIONES
    # =====================================================

        if generar_graficos:

        # -------------------------------------------------
        # Reporte completo (real vs. predicho, residuos,
        # importancia de variables, error por feriado,
        # diagnóstico de ARIMA y comparación WMAE)
        # -------------------------------------------------

            resultadosRandomForest = {
                "y_real": val_rf["Weekly_Sales"],
                "y_pred": pred_rf,
                "es_feriado": val_rf["IsHoliday"],
            }

            resultadosArima = {
                "y_real": val_arima.values,
                "y_pred": pred_arima,
            }

            self.visualizador.generar_reporte_completo(
                resultadosRandomForest,
                resultadosArima,
                tabla_comparacion,
                modelo_rf,
                modelo_arima,
            )

        return tabla_comparacion


# =============================================================
# PUNTO DE ENTRADA
# =============================================================

if __name__ == "__main__":
    """
    Ejecutar desde la raíz del proyecto:

        python -m walmart_sales_forecast.pipeline
    """

    pipeline = Pipeline()

    tabla_comparacion = (
        pipeline.comparar_modelos()
    )

    print(tabla_comparacion)
