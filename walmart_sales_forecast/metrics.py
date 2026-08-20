"""
UBICACIÓN: walmart_forecast/metrics.py
(archivo NUEVO que tú agregas junto a config.py, dataset.py, features.py)

Contiene la clase Evaluador, que calcula el WMAE exacto de tu PDF:
WMAE = (1 / sum(w_i)) * sum(w_i * |y_i - y_hat_i|)
donde w_i = 5 si es semana feriado, 1 en caso contrario.
"""

import numpy as np
import pandas as pd


class Evaluador:
    """Calcula y compara métricas de evaluación entre modelos."""

    def calcular_wmae(self, y_real, y_pred, pesos) -> float:
        y_real = np.asarray(y_real)
        y_pred = np.asarray(y_pred)
        pesos = np.asarray(pesos)

        numerador = np.sum(pesos * np.abs(y_real - y_pred))
        denominador = np.sum(pesos)
        return numerador / denominador

    def generar_reporte(self, nombre_modelo: str, wmae: float) -> pd.DataFrame:
        return pd.DataFrame([{"Modelo": nombre_modelo, "WMAE": wmae}])

    def comparar_modelos(self, lista_resultados: list) -> pd.DataFrame:
        """
        lista_resultados: lista de tuplas (nombre_modelo, wmae)
        Devuelve una tabla ordenada de mejor (menor WMAE) a peor.
        """
        df = pd.DataFrame(lista_resultados, columns=["Modelo", "WMAE"])
        return df.sort_values("WMAE").reset_index(drop=True)
