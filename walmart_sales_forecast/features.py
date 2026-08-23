"""
UBICACIÓN: walmart_forecast/features.py
(ccds ya te crea este archivo; reemplaza su contenido)

Contiene la clase Preprocesador: limpieza + creación de variables +
escalado. Es la misma clase "Preprocesador" del diagrama.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from walmart_sales_forecast.config import config


class Preprocesador:
    """
    Limpia el DataFrame combinado y crea las variables necesarias
    para modelar (fecha, semana feriada, lags, codificación, escalado).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.columnas_numericas = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]

    # --- Limpieza ---
    def tratarValoresNulos(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Los MarkDown vienen como NA cuando no hubo promoción -> 0 tiene sentido
        columnas_markdown = [c for c in df.columns if "MarkDown" in c]
        df[columnas_markdown] = df[columnas_markdown].fillna(0)
        # CPI/Unemployment: relleno hacia adelante por tienda (siguen tendencia)
        df[["CPI", "Unemployment"]] = df.groupby("Store")[["CPI", "Unemployment"]].transform(
            lambda serie: serie.ffill().bfill()
        )
        return df

    def tratarOutliers(self, df: pd.DataFrame, columna: str = "Weekly_Sales") -> pd.DataFrame:
        df = df.copy()
        if columna not in df.columns:
            return df
        q1, q3 = df[columna].quantile([0.01, 0.99])
        df[columna] = df[columna].clip(lower=q1, upper=q3)
        return df

    # --- Feature engineering ---
    def extraerVariablesFecha(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Anio"] = df["Date"].dt.year
        df["Mes"] = df["Date"].dt.month
        df["Semana"] = df["Date"].dt.isocalendar().week.astype(int)
        return df

    def crearVariableSemanaFestiva(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Peso_WMAE"] = df["IsHoliday"].apply(
            lambda es_feriado: config.obtener("peso_semana_feriado")
            if es_feriado else config.obtener("peso_semana_normal")
        )
        return df

    def crearVariablesLag(self, df: pd.DataFrame, n_periodos: int = 1) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["Store", "Dept", "Date"])
        df[f"Ventas_Lag_{n_periodos}"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(n_periodos)
        return df

    def codificar_categoricas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["IsHoliday"] = df["IsHoliday"].astype(int)
        df = pd.get_dummies(df, columns=["Type"], prefix="TipoTienda")
        return df

    def escalar_numericas(self, df: pd.DataFrame, ajustar: bool = True) -> pd.DataFrame:
        df = df.copy()
        columnas_presentes = [c for c in self.columnas_numericas if c in df.columns]
        if ajustar:
            df[columnas_presentes] = self.scaler.fit_transform(df[columnas_presentes])
        else:
            df[columnas_presentes] = self.scaler.transform(df[columnas_presentes])
        return df

    def dividir_train_val(self, df: pd.DataFrame, pct_validacion: float = 0.2):
        """División temporal (no aleatoria): las últimas semanas van a validación."""
        df = df.sort_values("Date")
        punto_corte = int(len(df) * (1 - pct_validacion))
        return df.iloc[:punto_corte], df.iloc[punto_corte:]

    # --- Método orquestador ---
    def preprocesar(self, df: pd.DataFrame, ajustar_scaler: bool = True) -> pd.DataFrame:
        df = self.tratarValoresNulos(df)
        if "Weekly_Sales" in df.columns:
            df = self.tratarOutliers(df)
        df = self.extraerVariablesFecha(df)
        df = self.crearVariableSemanaFestiva(df)
        if "Weekly_Sales" in df.columns:
            df = self.crearVariablesLag(df)
        df = self.codificar_categoricas(df)
        df = self.escalar_numericas(df, ajustar=ajustar_scaler)
        return df
