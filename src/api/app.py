import sys
from pathlib import Path

# Permite importar walmart_sales_forecast estando en src/api/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.model_loader import (
    MODEL_NAME,
    cargarModelo,
    obtenerMetadataModelo,
    obtenerPreprocesadorEntrenado,
)

app = FastAPI(
    title="API de Prediccion de Ventas - Walmart",
    version="1.0.0",
)

modelo = None
preprocesador = None
metadataModelo = {"version": "Desconocida", "run_id": "Desconocido"}


@app.on_event("startup")
def eventoInicio():
    # Carga el modelo y el scaler UNA sola vez al arrancar, no en cada request
    global modelo, preprocesador, metadataModelo
    modelo = cargarModelo()
    preprocesador = obtenerPreprocesadorEntrenado()
    metadataModelo = obtenerMetadataModelo()


class VentaEntrada(BaseModel):
    tienda: int
    departamento: int
    fecha: str = Field(..., description="YYYY-MM-DD")
    temperatura: float
    precio_combustible: float
    cpi: float
    desempleo: float
    tipo_tienda: str
    tamano: int
    es_feriado: bool
    ventas_semana_anterior: float


class PrediccionRequest(BaseModel):
    ventas: list[VentaEntrada]


class EstadoResponse(BaseModel):
    estado: str
    modelo: str
    version_produccion: str
    run_id: str


class ModeloMetadata(BaseModel):
    nombre: str
    version: str
    run_id: str


class PrediccionResultado(BaseModel):
    tienda: int
    departamento: int
    fecha: str
    prediccion_ventas_semanales: float


class PrediccionResponse(BaseModel):
    modelo_metadata: ModeloMetadata
    total_predicciones: int
    resultados: list[PrediccionResultado]


@app.get("/", response_model=EstadoResponse)
def raiz():
    return {
        "estado": "En linea",
        "modelo": MODEL_NAME,
        "version_produccion": metadataModelo["version"],
        "run_id": metadataModelo["run_id"],
    }


@app.post("/predecir", response_model=PrediccionResponse)
def predecir(payload: PrediccionRequest):
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no esta cargado.")

    try:
        filas = [
            {
                "Store": venta.tienda,
                "Dept": venta.departamento,
                "Date": venta.fecha,
                "Temperature": venta.temperatura,
                "Fuel_Price": venta.precio_combustible,
                "CPI": venta.cpi,
                "Unemployment": venta.desempleo,
                "Type": venta.tipo_tienda,
                "Size": venta.tamano,
                "IsHoliday": venta.es_feriado,
                # En entrenamiento este valor se calcula con .shift(1) sobre
                # el historial; aqui no hay historial, lo manda el cliente.
                "Ventas_Lag_1": venta.ventas_semana_anterior,
            }
            for venta in payload.ventas
        ]
        df = pd.DataFrame(filas)

        # Mismas transformaciones que en entrenamiento, pero solo transform
        # (el scaler ya viene ajustado, no se vuelve a entrenar aqui)
        df = preprocesador.extraer_variables_fecha(df)
        df = preprocesador.codificar_categoricas(df)
        df = preprocesador.transformar_scaler(df)

        # Si falta alguna columna que el modelo si vio en entrenamiento
        # (ej. Type_C si nadie mando una tienda tipo C), se rellena con 0
        df = df.reindex(columns=modelo.feature_names_in_, fill_value=0)

        predicciones = modelo.predict(df)

        resultados = [
            {
                "tienda": venta.tienda,
                "departamento": venta.departamento,
                "fecha": venta.fecha,
                "prediccion_ventas_semanales": round(float(pred), 2),
            }
            for venta, pred in zip(payload.ventas, predicciones)
        ]

        return {
            "modelo_metadata": {
                "nombre": MODEL_NAME,
                "version": metadataModelo["version"],
                "run_id": metadataModelo["run_id"],
            },
            "total_predicciones": len(resultados),
            "resultados": resultados,
        }

    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
