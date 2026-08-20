"""
UBICACIÓN: walmart_forecast/modeling/models.py
(archivo NUEVO dentro de la carpeta modeling/ que ya crea ccds,
 junto a train.py y predict.py)

Contiene ModeloBase (clase abstracta) y las implementaciones
ModeloRandomForest y ModeloARIMA. Cualquier modelo nuevo que agregues
después (ExponentialSmoothing, ARCH, etc.) hereda de ModeloBase igual
que estos dos.
"""

from abc import ABC, abstractmethod
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


class ModeloBase(ABC):
    """Contrato que todo modelo del proyecto debe cumplir."""

    def __init__(self, nombre_modelo: str):
        self.nombre_modelo = nombre_modelo
        self.esta_entrenado = False
        self.modelo = None

    @abstractmethod
    def entrenar(self, X, y):
        ...

    @abstractmethod
    def predecir(self, X):
        ...

    def guardar_modelo(self, ruta):
        joblib.dump(self.modelo, ruta)

    def cargar_modelo(self, ruta):
        self.modelo = joblib.load(ruta)
        self.esta_entrenado = True


class ModeloRandomForest(ModeloBase):
    """Random Forest: usa las variables ya preprocesadas por Preprocesador."""

    def __init__(self, n_estimadores=200, profundidad_maxima=12, semilla_aleatoria=42):
        super().__init__(nombre_modelo="RandomForest")
        self.n_estimadores = n_estimadores
        self.profundidad_maxima = profundidad_maxima
        self.semilla_aleatoria = semilla_aleatoria
        self.modelo = RandomForestRegressor(
            n_estimators=self.n_estimadores,
            max_depth=self.profundidad_maxima,
            random_state=self.semilla_aleatoria,
            n_jobs=-1,
        )

    def entrenar(self, X, y):
        self.modelo.fit(X, y)
        self.esta_entrenado = True

    def predecir(self, X):
        return self.modelo.predict(X)

    def obtener_importancia_variables(self) -> pd.Series:
        return pd.Series(
            self.modelo.feature_importances_, index=self.modelo.feature_names_in_
        ).sort_values(ascending=False)


class ModeloARIMA(ModeloBase):
    """
    ARIMA trabaja distinto: es univariado y se entrena por serie temporal
    (normalmente por combinación Store-Dept), no con una matriz X de
    variables como Random Forest.
    """

    def __init__(self, orden_pdq=(1, 1, 1)):
        super().__init__(nombre_modelo="ARIMA")
        self.orden_pdq = orden_pdq
        self.modelo = None
        self._resultado_ajuste = None

    def entrenar(self, X, y):
        # Para ARIMA, "y" es la serie de Weekly_Sales ordenada por fecha
        self.modelo = ARIMA(y, order=self.orden_pdq)
        self._resultado_ajuste = self.modelo.fit()
        self.esta_entrenado = True

    def predecir(self, X):
        # "X" aquí representa la cantidad de periodos futuros a predecir
        n_periodos = len(X) if hasattr(X, "__len__") else X
        return self._resultado_ajuste.forecast(steps=n_periodos)

    def guardar_modelo(self, ruta):
        self._resultado_ajuste.save(ruta)

    def cargar_modelo(self, ruta):
        from statsmodels.tsa.arima.model import ARIMAResults
        self._resultado_ajuste = ARIMAResults.load(ruta)
        self.esta_entrenado = True
