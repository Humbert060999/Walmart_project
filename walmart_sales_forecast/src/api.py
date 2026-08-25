import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. INICIALIZACIÓN DE FASTAPI Y CARGA
# ==========================================
app = FastAPI(
    title="API de Predicción de Ventas de Walmart",
    description="API MLOps robusta para predecir las ventas semanales por tienda y departamento.",
    version="1.0.0"
)

model = None
model_version_info = {"version": "1.0.0", "run_id": "produccion_local"}

@app.on_event("startup")
def startup_event():
    global model
    try:
        # Ruta dinámica basada en la estructura del proyecto
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "../../models/modelo_random_forest.pkl")
        
        model = joblib.load(MODEL_PATH)
        print(f"[FastAPI] ¡Modelo de Walmart cargado exitosamente desde: {MODEL_PATH}!")
    except Exception as e:
        print(f"[FastAPI ERROR] No se pudo cargar el modelo: {e}")

# ==========================================
# 2. ESQUEMAS DE ENTRADA (Alineados con el entrenamiento)
# ==========================================
class WalmartFeatures(BaseModel):
    Store: int
    Dept: int
    IsHoliday: bool
    Temperature: float
    Fuel_Price: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    CPI: float
    Unemployment: float
    Size: int
    Anio: int
    Mes: int
    Semana: int
    Ventas_Lag_1: float
    TipoTienda_A: bool
    TipoTienda_B: bool
    TipoTienda_C: bool

    class Config:
        validate_by_name = True

class PredictionRequest(BaseModel):
    data: List[WalmartFeatures]

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "project": "Walmart Sales Forecast",
        "production_version": model_version_info["version"],
    }

@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo no está cargado en memoria."
        )
    
    try:
        # Convertir los datos de entrada a DataFrame de Pandas
        input_data = pd.DataFrame([item.model_dump(by_alias=True) for item in payload.data])
        
        # Realizar la predicción con el modelo
        predictions = model.predict(input_data)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "store": int(payload.data[i].Store),
                "dept": int(payload.data[i].Dept),
                "predicted_weekly_sales": round(float(pred), 2)
            })

        return {
            "model_metadata": {
                "version": model_version_info["version"],
            },
            "total_predictions": len(predictions),
            "results": results,
            "message": "Predicción de ventas completada con éxito."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")