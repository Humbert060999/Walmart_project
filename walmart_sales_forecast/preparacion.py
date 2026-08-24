"""
UBICACIÓN: walmart_sales_forecast/preparacion.py
Contiene la lógica específica para la validación y preparación del EDA,
importando las capacidades base de features.py.
"""

import pandas as pd
from walmart_sales_forecast.features import Preprocesador


class PreparadorEDA(Preprocesador):
    """
    Hereda de Preprocesador para reutilizar la limpieza y añade 
    métodos exclusivos de validación de formatos y control de duplicados para el EDA.
    """

    def __init__(self):
        super().__init__()

    def detectarDuplicados(self, df: pd.DataFrame) -> int:
        """Retorna la cantidad de filas duplicadas exactas."""
        return df.duplicated().sum()

    def eliminarDuplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas exactamente iguales en el DataFrame."""
        df = df.copy()
        filas_antes = len(df)
        df = df.drop_duplicates()
        filas_despues = len(df)
        if filas_antes != filas_despues:
            print(f"[PreparadorEDA] Se eliminaron {filas_antes - filas_despues} filas duplicadas.")
        return df

    def validarFormatos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y corrige los tipos de datos y formatos críticos 
        necesarios antes de hacer el análisis exploratorio.
        """
        df = df.copy()
        
        # 1. Validar y convertir la columna Date a formato datetime
        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            print("[PreparadorEDA] Columna 'Date' convertida a datetime.")

        # 2. Asegurar que las columnas clave sean numéricas enteras
        columnas_enteras = ["Store", "Dept"]
        for col in columnas_enteras:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # 3. Validar booleanos para IsHoliday
        if "IsHoliday" in df.columns:
            df["IsHoliday"] = df["IsHoliday"].astype(bool)

        return df

    def prepararParaEda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline exclusivo para el EDA: valida formatos, quita duplicados 
        y trata nulos usando los métodos heredados de Preprocesador.
        """
        df = self.validarFormatos(df)
        df = self.eliminarDuplicados(df)
        df = self.tratarValoresNulos(df)  # Método heredado de features.py
        return df