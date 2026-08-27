import numpy as np
import pandas as pd

from walmart_sales_forecast.modeling.models import ModeloARIMA, ModeloRandomForest


def test_modeloRandomForestUsaHiperparametrosPorDefecto():
    modelo = ModeloRandomForest()

    assert modelo.modelo.n_estimators == 200
    assert modelo.modelo.max_depth == 12


def test_modeloRandomForestNoEstaEntrenadoAlCrear():
    modelo = ModeloRandomForest()

    assert modelo.esta_entrenado is False


def test_modeloRandomForestQuedaEntrenadoTrasEntrenar():
    modelo = ModeloRandomForest(n_estimadores=10, profundidad_maxima=3)
    X = pd.DataFrame({"Temperature": np.arange(20.0), "Size": np.arange(20.0) * 100})
    y = pd.Series(np.arange(20.0) * 10)

    modelo.entrenar(X, y)

    assert modelo.esta_entrenado is True
    predicciones = modelo.predecir(X)
    assert len(predicciones) == len(X)


def test_modeloRandomForestObtenerImportanciaVariables():
    modelo = ModeloRandomForest(n_estimadores=10, profundidad_maxima=3)
    X = pd.DataFrame({"Temperature": np.arange(20.0), "Size": np.arange(20.0) * 100})
    y = pd.Series(np.arange(20.0) * 10)
    modelo.entrenar(X, y)

    importancias = modelo.obtener_importancia_variables()

    assert set(importancias.index) == {"Temperature", "Size"}


def test_modeloArimaNoEstaEntrenadoAlCrear():
    modelo = ModeloARIMA()

    assert modelo.esta_entrenado is False


def test_modeloArimaQuedaEntrenadoYPredicePeriodosFuturos():
    modelo = ModeloARIMA(orden_pdq=(1, 1, 1))
    rng = np.random.default_rng(42)
    serie = pd.Series(np.linspace(100, 200, 30) + rng.normal(0, 2, 30))

    modelo.entrenar(X=None, y=serie)

    assert modelo.esta_entrenado is True
    predicciones = modelo.predecir(5)
    assert len(predicciones) == 5
