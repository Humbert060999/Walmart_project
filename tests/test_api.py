import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as appModule
from walmart_sales_forecast.features import Preprocesador


class ModeloFalso:
    feature_names_in_ = ["Temperature", "Fuel_Price"]

    def predict(self, X):
        return [12345.67] * len(X)


def preprocesadorFalso():
    preprocesador = Preprocesador()
    dfAjuste = pd.DataFrame({
        "Temperature": [10.0, 20.0],
        "Fuel_Price": [2.0, 3.0],
        "CPI": [200.0, 210.0],
        "Unemployment": [7.0, 8.0],
        "Size": [100000, 150000],
    })
    preprocesador.ajustar_scaler(dfAjuste)
    return preprocesador


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appModule, "cargarModelo", lambda: ModeloFalso())
    monkeypatch.setattr(appModule, "obtenerPreprocesadorEntrenado", preprocesadorFalso)
    monkeypatch.setattr(appModule, "obtenerMetadataModelo", lambda: {"version": "1", "run_id": "abc123"})

    with TestClient(appModule.app) as testClient:
        yield testClient


def ventaValida():
    return {
        "tienda": 5,
        "departamento": 12,
        "fecha": "2013-03-15",
        "temperatura": 45.3,
        "precio_combustible": 3.6,
        "cpi": 220.5,
        "desempleo": 7.2,
        "tipo_tienda": "A",
        "tamano": 151315,
        "es_feriado": False,
        "ventas_semana_anterior": 23000.50,
    }


def test_raizRespondeEstadoEnLinea(client):
    respuesta = client.get("/")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "En linea"
    assert cuerpo["version_produccion"] == "1"


def test_predecirConPayloadValidoRespondeResultados(client):
    respuesta = client.post("/predecir", json={"ventas": [ventaValida()]})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total_predicciones"] == 1
    assert cuerpo["resultados"][0]["prediccion_ventas_semanales"] == pytest.approx(12345.67)


def test_predecirConCampoFaltanteRespondeErrorValidacion(client):
    ventaIncompleta = ventaValida()
    del ventaIncompleta["tienda"]

    respuesta = client.post("/predecir", json={"ventas": [ventaIncompleta]})

    assert respuesta.status_code == 422


def test_predecirConFechaInvalidaRespondeErrorControlado(client):
    ventaConFechaInvalida = ventaValida()
    ventaConFechaInvalida["fecha"] = "no-es-una-fecha"

    respuesta = client.post("/predecir", json={"ventas": [ventaConFechaInvalida]})

    assert respuesta.status_code == 400
