import pytest

from walmart_sales_forecast.metrics import Evaluador


@pytest.fixture
def evaluador():
    return Evaluador()


def test_calcularWmaeValorConocido(evaluador):
    yReal = [100, 200]
    yPred = [90, 180]
    pesos = [1, 1]

    wmae = evaluador.calcular_wmae(yReal, yPred, pesos)

    assert wmae == pytest.approx(15.0)


def test_calcularWmaePrediccionPerfectaEsCero(evaluador):
    yReal = [100, 200, 300]
    yPred = [100, 200, 300]
    pesos = [1, 5, 1]

    wmae = evaluador.calcular_wmae(yReal, yPred, pesos)

    assert wmae == pytest.approx(0.0)


def test_calcularWmaePonderaMasSemanaFeriado(evaluador):
    yReal = [100, 100]
    yPred = [110, 100]  # error solo en la primera semana

    wmaeConFeriado = evaluador.calcular_wmae(yReal, yPred, pesos=[5, 1])
    wmaeSinPonderar = evaluador.calcular_wmae(yReal, yPred, pesos=[1, 1])

    assert wmaeConFeriado > wmaeSinPonderar


def test_compararModelosOrdenaPorMenorWmae(evaluador):
    resultados = [("ARIMA", 1_500_000.0), ("RandomForest", 1800.0)]

    tabla = evaluador.comparar_modelos(resultados)

    assert tabla.iloc[0]["Modelo"] == "RandomForest"
    assert tabla.iloc[-1]["Modelo"] == "ARIMA"
