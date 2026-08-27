"""
UBICACIÓN: walmart_forecast/dataset.py
(ccds ya te crea este archivo; reemplaza su contenido)

Contiene las clases DataLoader (lee los CSV crudos) y DataIntegrador
(los combina en un solo DataFrame). Corresponden a las mismas clases
del diagrama de clases.
"""

import pandas as pd

from walmart_sales_forecast.config import (
    RUTA_DATASET_COMBINADO,
    RUTA_FEATURES,
    RUTA_STORES,
    RUTA_TEST,
    RUTA_TRAIN,
)


class DataLoader:
    """Solo lee los 4 CSV desde data/raw/. No transforma nada."""

    def __init__(self, ruta_train=RUTA_TRAIN, ruta_test=RUTA_TEST,
                 ruta_features=RUTA_FEATURES, ruta_stores=RUTA_STORES):
        self.ruta_train = ruta_train
        self.ruta_test = ruta_test
        self.ruta_features = ruta_features
        self.ruta_stores = ruta_stores

    def cargar_train(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta_train, parse_dates=["Date"])

    def cargar_test(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta_test, parse_dates=["Date"])

    def cargar_features(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta_features, parse_dates=["Date"])

    def cargar_stores(self) -> pd.DataFrame:
        return pd.read_csv(self.ruta_stores)


class DataIntegrador:
    """Combina train/test + features + stores en un solo DataFrame."""

    def combinar_datasets(self, df_principal: pd.DataFrame,
                           df_features: pd.DataFrame,
                           df_stores: pd.DataFrame) -> pd.DataFrame:
        # features.csv trae su propio IsHoliday duplicado -> lo quitamos
        # antes del merge para no generar columnas repetidas (_x, _y)
        df_features = df_features.drop(columns=["IsHoliday"])

        df = df_principal.merge(df_features, on=["Store", "Date"], how="left")
        df = df.merge(df_stores, on="Store", how="left")
        return df

    def validar_integridad(self, df_original: pd.DataFrame,
                            df_combinado: pd.DataFrame) -> bool:
        """Chequea que el merge no haya perdido ni duplicado filas."""
        return len(df_original) == len(df_combinado)


if __name__ == "__main__":
    # Prueba rápida: ejecuta "python -m walmart_forecast.dataset"
    cargador = DataLoader()
    integrador = DataIntegrador()

    train = cargador.cargar_train()
    features = cargador.cargar_features()
    stores = cargador.cargar_stores()

    dataset = integrador.combinar_datasets(train, features, stores)
    assert integrador.validar_integridad(train, dataset), "¡Se perdieron filas en el merge!"

    dataset.to_csv(RUTA_DATASET_COMBINADO, index=False)
    print(f"Dataset combinado guardado en {RUTA_DATASET_COMBINADO} con {len(dataset)} filas")
