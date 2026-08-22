
"""
UBICACIÓN: walmart_sales_forecast/features.py

Preprocesamiento y creación de variables para el proyecto Walmart.

Responsabilidades:
- Limpieza de datos.
- Creación de variables de fecha.
- Creación de variable Peso_WMAE.
- Creación de variables lag.
- Codificación de variables categóricas.
- División temporal train/validation.
- Ajuste y transformación del StandardScaler.

IMPORTANTE:
El scaler y los límites para tratamiento de outliers se ajustan
ÚNICAMENTE con los datos de entrenamiento para evitar data leakage.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from walmart_sales_forecast.config import config


class Preprocesador:
    """
    Limpia el DataFrame y crea las variables necesarias para modelar.

    Las transformaciones que aprenden parámetros de los datos deben
    ajustarse únicamente utilizando train.
    """

    def __init__(self):
        self.scaler = StandardScaler()

        self.columnas_numericas = [
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "Size",
        ]

    # =========================================================
    # LIMPIEZA
    # =========================================================

    def tratar_valores_nulos(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Trata valores nulos conocidos del dataset.

        MarkDown:
            NA significa que no hubo promoción, por lo que se
            reemplaza por 0.

        CPI / Unemployment:
            Se completan por tienda respetando el orden temporal.
        """

        df = df.copy()

        # -----------------------------------------------------
        # MarkDown
        # -----------------------------------------------------

        columnas_markdown = [
            c
            for c in df.columns
            if "MarkDown" in c
        ]

        if columnas_markdown:
            df[columnas_markdown] = (
                df[columnas_markdown]
                .fillna(0)
            )

        # -----------------------------------------------------
        # CPI / Unemployment
        # -----------------------------------------------------

        if (
            "Store" in df.columns
            and "Date" in df.columns
        ):
            df["Date"] = pd.to_datetime(
                df["Date"]
            )

            df = df.sort_values(
                ["Store", "Date"]
            )

            columnas_rellenar = [
                c
                for c in [
                    "CPI",
                    "Unemployment",
                ]
                if c in df.columns
            ]

            if columnas_rellenar:
                df[columnas_rellenar] = (
                    df.groupby("Store")[
                        columnas_rellenar
                    ]
                    .transform(
                        lambda serie:
                        serie.ffill().bfill()
                    )
                )

        return df

    # =========================================================
    # OUTLIERS
    # =========================================================

    def obtener_limites_outliers(
        self,
        df: pd.DataFrame,
        columna: str = "Weekly_Sales",
    ):
        """
        Calcula los límites de outliers.

        IMPORTANTE:
        Este método debe recibir SOLO el conjunto de entrenamiento.

        Los límites no se calculan sobre validation.
        """

        if columna not in df.columns:
            return None, None

        q1, q3 = df[columna].quantile(
            [0.01, 0.99]
        )

        return q1, q3

    def tratar_outliers(
        self,
        df: pd.DataFrame,
        columna: str = "Weekly_Sales",
        limites=None,
    ) -> pd.DataFrame:
        """
        Trata outliers utilizando límites previamente calculados.

        Si 'limites' no se proporciona, los calcula sobre el DataFrame
        recibido. Por eso, en el pipeline se deben calcular utilizando
        únicamente train.
        """

        df = df.copy()

        if columna not in df.columns:
            return df

        if limites is None:
            q1, q3 = self.obtener_limites_outliers(
                df,
                columna,
            )
        else:
            q1, q3 = limites

        if q1 is None or q3 is None:
            return df

        df[columna] = df[columna].clip(
            lower=q1,
            upper=q3,
        )

        return df

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    def extraer_variables_fecha(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extrae variables temporales de Date.
        """

        df = df.copy()

        if "Date" not in df.columns:
            return df

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df["Anio"] = (
            df["Date"].dt.year
        )

        df["Mes"] = (
            df["Date"].dt.month
        )

        df["Semana"] = (
            df["Date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        return df

    def crear_variable_semana_festiva(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Crea el peso utilizado por WMAE.

        Las semanas festivas reciben el peso configurado para
        feriados y las normales el peso correspondiente.
        """

        df = df.copy()

        if "IsHoliday" not in df.columns:
            return df

        df["Peso_WMAE"] = (
            df["IsHoliday"]
            .apply(
                lambda es_feriado:
                config.obtener(
                    "peso_semana_feriado"
                )
                if es_feriado
                else config.obtener(
                    "peso_semana_normal"
                )
            )
        )

        return df

    def crear_variables_lag(
        self,
        df: pd.DataFrame,
        n_periodos: int = 1,
    ) -> pd.DataFrame:
        """
        Crea una variable lag de Weekly_Sales.

        El desplazamiento se realiza por Store y Dept respetando
        el orden temporal.
        """

        df = df.copy()

        columnas_necesarias = {
            "Store",
            "Dept",
            "Date",
            "Weekly_Sales",
        }

        if not columnas_necesarias.issubset(
            df.columns
        ):
            return df

        df = df.sort_values(
            [
                "Store",
                "Dept",
                "Date",
            ]
        )

        nombre_lag = (
            f"Ventas_Lag_{n_periodos}"
        )

        df[nombre_lag] = (
            df.groupby(
                ["Store", "Dept"]
            )["Weekly_Sales"]
            .shift(n_periodos)
        )

        return df

    def codificar_categoricas(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Codifica variables categóricas.
        """

        df = df.copy()

        if "IsHoliday" in df.columns:
            df["IsHoliday"] = (
                df["IsHoliday"]
                .astype(int)
            )

        if "Type" in df.columns:
            df = pd.get_dummies(
                df,
                columns=["Type"],
                prefix="TipoTienda",
                dtype=int,
            )

        return df

    # =========================================================
    # PREPARACIÓN DE FEATURES
    # =========================================================

    def crear_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ejecuta las transformaciones que no necesitan ajustar
        parámetros estadísticos.

        NO realiza fit del scaler.
        NO calcula outliers.
        """

        df = self.tratar_valores_nulos(
            df
        )

        df = self.extraer_variables_fecha(
            df
        )

        df = self.crear_variable_semana_festiva(
            df
        )

        if "Weekly_Sales" in df.columns:
            df = self.crear_variables_lag(
                df
            )

        df = self.codificar_categoricas(
            df
        )

        return df

    # =========================================================
    # SCALER
    # =========================================================

    def ajustar_scaler(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ajusta StandardScaler utilizando ÚNICAMENTE train.
        """

        df = df.copy()

        columnas_presentes = [
            c
            for c in self.columnas_numericas
            if c in df.columns
        ]

        if columnas_presentes:
            df[columnas_presentes] = (
                self.scaler.fit_transform(
                    df[columnas_presentes]
                )
            )

        return df

    def transformar_scaler(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aplica un scaler previamente ajustado.

        NO ejecuta fit().
        """

        df = df.copy()

        columnas_presentes = [
            c
            for c in self.columnas_numericas
            if c in df.columns
        ]

        if columnas_presentes:
            df[columnas_presentes] = (
                self.scaler.transform(
                    df[columnas_presentes]
                )
            )

        return df

    # =========================================================
    # DIVISIÓN TEMPORAL
    # =========================================================

    def dividir_train_val(
        self,
        df: pd.DataFrame,
        pct_validacion: float = 0.2,
    ):
        """
        División temporal.

        Las observaciones más antiguas van a train y las últimas
        observaciones van a validation.

        No utiliza división aleatoria.
        """

        df = df.copy()

        if "Date" in df.columns:
            df = df.sort_values(
                "Date"
            )

        punto_corte = int(
            len(df)
            * (1 - pct_validacion)
        )

        train = df.iloc[
            :punto_corte
        ].copy()

        validation = df.iloc[
            punto_corte:
        ].copy()

        return train, validation

    # =========================================================
    # MÉTODO DE COMPATIBILIDAD
    # =========================================================

    def preprocesar(
        self,
        df: pd.DataFrame,
        ajustar_scaler: bool = True,
    ) -> pd.DataFrame:
        """
        Método de compatibilidad con versiones anteriores.

        Para el pipeline principal se recomienda utilizar
        explícitamente:

            crear_features()
            dividir_train_val()
            ajustar_scaler()
            transformar_scaler()

        De esta manera se controla correctamente el data leakage.
        """

        df = self.crear_features(
            df
        )

        if ajustar_scaler:
            df = self.ajustar_scaler(
                df
            )
        else:
            df = self.transformar_scaler(
                df
            )

        return df

