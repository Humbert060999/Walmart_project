"""
UBICACIÓN: walmart_forecast/modeling/train.py
(ccds ya te crea este archivo; reemplaza su contenido)

Script que orquesta: cargar -> combinar -> preprocesar -> entrenar
Random Forest -> guardar el modelo entrenado en models/.

Se ejecuta con: python -m walmart_forecast.modeling.train
"""

import pandas as pd

from walmart_sales_forecast.config import MODELS_DIR
from walmart_sales_forecast.dataset import DataLoader, DataIntegrador
from walmart_sales_forecast.features import Preprocesador
from walmart_sales_forecast.modeling.models import ModeloRandomForest


def main():
    cargador = DataLoader()
    integrador = DataIntegrador()
    preprocesador = Preprocesador()

    train = cargador.cargar_train()
    features = cargador.cargar_features()
    stores = cargador.cargar_stores()

    dataset = integrador.combinar_datasets(train, features, stores)
    dataset_procesado = preprocesador.preprocesar(dataset, ajustar_scaler=True)
    dataset_procesado = dataset_procesado.dropna()  # por los lags del inicio de cada serie

    columnas_excluidas = ["Weekly_Sales", "Date", "Peso_WMAE"]
    X = dataset_procesado.drop(columns=columnas_excluidas)
    y = dataset_procesado["Weekly_Sales"]

    train_split, val_split = preprocesador.dividir_train_val(dataset_procesado)

    modelo_rf = ModeloRandomForest()
    modelo_rf.entrenar(
        X.loc[train_split.index],
        y.loc[train_split.index],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo_rf.guardar_modelo(MODELS_DIR / "modelo_random_forest.pkl")
    print("Modelo Random Forest entrenado y guardado en models/modelo_random_forest.pkl")


if __name__ == "__main__":
    main()
