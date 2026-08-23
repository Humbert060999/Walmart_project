"""

CLASE PARA MOSTRAR GRÁFICAS DEL ENTRENAMIENTO
- Visualizar valores reales vs predichos.
- Comparar el WMAE de Random Forest vs ARIMA.
- Visualizar la importancia de variables de Random Forest.
- Visualizar residuos (dispersión y distribución).
- Visualizar el error absoluto según semana feriado vs normal,relevante porque el WMAE del caso pondera x5 las semanas feriado.
- Visualizar el diagnóstico nativo de statsmodels para ARIMA.
- Orquestar la generación de todas las figuras de una corrida de entrenamiento mediante generar_reporte_completo().
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from walmart_sales_forecast.config import FIGURES_DIR
from walmart_sales_forecast.modeling.models import ModeloARIMA, ModeloRandomForest
from walmart_sales_forecast.pipeline import Pipeline

COLOR_PRIMARIO = "#0f766e"
COLOR_SECUNDARIO = "#d97706"

 
DPI_FIGURAS = 600
FORMATO_FIGURAS = "png"
 
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
 

class VisualizadorEntrenamiento:
    """Genera los graficos del entrenamiento de los modelos"""

    def __init__(self, directorio_salida: Path = FIGURES_DIR):
        self.directorio_salida = Path(directorio_salida)
        self.directorio_salida.mkdir(parents=True, exist_ok=True)

    def graficar_real_vs_predicho(self, y_real, y_pred, nombre_modelo: str,) -> Path:
        """Dispersion de valores reales vs predichos"""
        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        minimo = min(y_real.min(), y_pred.min())
        maximo = max(y_real.max(), y_pred.max())

        plt.figure(figsize=(7, 7))
        plt.scatter(
            y_real,
            y_pred,
            alpha=0.4,
            color=COLOR_PRIMARIO,
            edgecolor="white",
            linewidth=0.3,
            label="Observaciones",
        )
        plt.plot(
            [minimo, maximo],
            [minimo, maximo],
            linestyle="--",
            color=COLOR_SECUNDARIO,
            label="Predicción perfecta",
        )
        plt.title(f"Real vs predicho — {nombre_modelo}")
        plt.xlabel("Venta semanales real")
        plt.ylabel("Venta semanales predicho")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()

        nombre_archivo = f"real_vs_predicho_{nombre_modelo.lower()}.{FORMATO_FIGURAS}"

        ruta_salida = self.directorio_salida / f"real_vs_predicho_{nombre_modelo.lower()}.png"
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def graficar_comparacion_modelos(self,tabla_wmae: pd.DataFrame,) -> Path:
        """WMAE de RandomForest vs ARIMA"""
        datos_ordenados = tabla_wmae.sort_values("WMAE")

        plt.figure(figsize=(7, 5))
        barras = plt.bar(
            datos_ordenados["Modelo"], datos_ordenados["WMAE"], color=COLOR_PRIMARIO
        )

        for barra, valor in zip(barras, datos_ordenados["WMAE"]):
            plt.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height(),
                f"{valor:,.2f}",
                ha="center",
                va="bottom",
            )

        plt.title("Comparación de modelos — WMAE")
        plt.xlabel("Modelo")
        plt.ylabel("WMAE")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        ruta_salida = self.directorio_salida / "comparacion_modelos_wmae.png"
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def graficar_importancia_variables(self, modelo_rf: ModeloRandomForest,top_n: int = 15,) -> Path:
        """Grafico de barra de caracteristicas del modelo Random Forest"""
        importancias = modelo_rf.obtener_importancia_variables()
        top_importancias = importancias.head(top_n).sort_values()

        plt.figure(figsize=(8, max(4, 0.35 * len(top_importancias))))
        top_importancias.plot(kind="barh", color=COLOR_PRIMARIO)

        plt.title("Importancia de variables en RandomForest")
        plt.xlabel("Importancia relativa")
        plt.ylabel("Variable")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        ruta_salida = self.directorio_salida / f"importancia_variables_rf.{FORMATO_FIGURAS}"
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def graficar_residuos(self, y_real,y_pred,nombre_modelo: str,) -> Path:
        """Residuos vs predicho + histograma de residuos
        Permite detectar sesgo sistemático y heterocedasticidad """
        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuos = y_real - y_pred

        figura, (eje_dispersion, eje_histograma) = plt.subplots(1, 2, figsize=(12, 5))

        eje_dispersion.scatter(
            y_pred,
            residuos,
            alpha=0.4,
            color=COLOR_PRIMARIO,
            edgecolor="white",
            linewidth=0.3,
        )
        eje_dispersion.axhline(0, linestyle="--", color=COLOR_SECUNDARIO)
        eje_dispersion.set_title(f"Residuos vs predicho — {nombre_modelo}")
        eje_dispersion.set_xlabel("Venta semanales predicho")
        eje_dispersion.set_ylabel("Residuo (real - predicho)")
        eje_dispersion.grid(True, alpha=0.3)

        eje_histograma.hist(residuos, bins=40, color=COLOR_PRIMARIO, edgecolor="white")
        eje_histograma.axvline(0, linestyle="--", color=COLOR_SECUNDARIO)
        eje_histograma.set_title("Distribución de residuos")
        eje_histograma.set_xlabel("Residuo")
        eje_histograma.set_ylabel("Frecuencia")
        eje_histograma.grid(True, alpha=0.3)

        plt.tight_layout()

        ruta_salida = self.directorio_salida / f"residuos_{nombre_modelo.lower()}.png"
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def graficar_error_por_feriado(
        self,
        y_real,
        y_pred,
        es_feriado,
        nombre_modelo: str,
    ) -> Path:
        """Boxplot del error absoluto para semanas feriado y normales
        El WMAE del caso pondera cinco veces las semanas feriado, por lo que interesa saber si el modelo falla más justo ahí."""

        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        es_feriado = np.asarray(es_feriado).astype(bool)

        error_absoluto = np.abs(y_real - y_pred)
        etiquetas = np.where(es_feriado, "Feriado", "Normal")
        datos_grafico = pd.DataFrame({"Error": error_absoluto, "Tipo": etiquetas})

        plt.figure(figsize=(6, 5))
        datos_grafico.boxplot(
            column="Error",
            by="Tipo",
            grid=False,
            boxprops={"color": COLOR_PRIMARIO},
            medianprops={"color": COLOR_SECUNDARIO},
            whiskerprops={"color": COLOR_PRIMARIO},
        )
        plt.title(f"Error absoluto por tipo de semana — {nombre_modelo}")
        plt.suptitle("")
        plt.xlabel("Tipo de semana")
        plt.ylabel("Error absoluto")
        plt.tight_layout()

        ruta_salida = self.directorio_salida / f"error_feriado_{nombre_modelo.lower()}.png"
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def graficar_diagnostico_arima(
        self,
        modelo_arima: ModeloARIMA,
    ) -> Path:
        """Genera el diagnóstico nativo de statsmodels (libreria para hacer analisis estadistico) para el ajuste de ARIMA
        Incluye residuos estandarizados, histograma con densidad estimada, Q-Q plot y correlograma (ACF) de los residuos"""
        figura = modelo_arima._resultado_ajuste.plot_diagnostics(figsize=(10, 8))
        plt.tight_layout()

        ruta_salida = self.directorio_salida / "diagnostico_arima.png"
        figura.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        plt.close(figura)

        logger.info(f"Figura guardada: {ruta_salida}")
        return ruta_salida

    def generar_reporte_completo(
        self,
        resultados_random_forest: dict,
        resultados_arima: dict,
        tabla_wmae: pd.DataFrame,
        modelo_rf: ModeloRandomForest,
        modelo_arima: ModeloARIMA | None = None,
    ) -> dict:
        """Genera todas las figuras de una corrida de entrenamiento"""
        rutas_generadas = {}

        rutas_generadas["real_vs_predicho_rf"] = self.graficar_real_vs_predicho(
            resultados_random_forest["y_real"],
            resultados_random_forest["y_pred"],
            "RandomForest",
        )

        rutas_generadas["importancia_variables"] = self.graficar_importancia_variables(
            modelo_rf
        )

        rutas_generadas["residuos_rf"] = self.graficar_residuos(
            resultados_random_forest["y_real"],
            resultados_random_forest["y_pred"],
            "RandomForest",
        )
        if "es_feriado" in resultados_random_forest:
            rutas_generadas["error_feriado_rf"] = self.graficar_error_por_feriado(
                resultados_random_forest["y_real"],
                resultados_random_forest["y_pred"],
                resultados_random_forest["es_feriado"],
                "RandomForest",
            )
     
        rutas_generadas["real_vs_predicho_arima"] = self.graficar_real_vs_predicho(
            resultados_arima["y_real"],
            resultados_arima["y_pred"],
            "ARIMA",
        )
        rutas_generadas["residuos_arima"] = self.graficar_residuos(
            resultados_arima["y_real"],
            resultados_arima["y_pred"],
            "ARIMA",
        )
        if modelo_arima is not None:
            rutas_generadas["diagnostico_arima"] = self.graficar_diagnostico_arima(modelo_arima)

        logger.info("Generando comparación de modelos")
        rutas_generadas["comparacion_modelos"] = self.graficar_comparacion_modelos(tabla_wmae)

        logger.success(f"Reporte completo generado: {len(rutas_generadas)} figuras.")
        return rutas_generadas


def main() -> None:
    """Entrena los modelos y genera todas las figuras del reporte."""
    pipeline = Pipeline()
    visualizador = VisualizadorEntrenamiento()

    modelo_rf = ModeloRandomForest(
        n_estimadores=pipeline.config.obtener("n_estimadores_rf"),
        profundidad_maxima=pipeline.config.obtener("profundidad_maxima_rf"),
    )
    modelo_rf, wmae_rf, val_rf, pred_rf = pipeline.entrenar_y_evaluar(modelo_rf)

    modelo_arima, wmae_arima, val_arima, pred_arima = pipeline.evaluar_arima()

    tabla_wmae = pipeline.evaluador.comparar_modelos(
        [("RandomForest", wmae_rf), ("ARIMA", wmae_arima)]
    )
    resultados_random_forest = {
        "y_real": val_rf["Weekly_Sales"],
        "y_pred": pred_rf,
        "es_feriado": val_rf["IsHoliday"],
    }
    resultados_arima = {
        "y_real": val_arima.values,
        "y_pred": pred_arima,
    }

    rutas_generadas = visualizador.generar_reporte_completo(
        resultados_random_forest,
        resultados_arima,
        tabla_wmae,
        modelo_rf,
        modelo_arima,
    )

    print("\nResultados WMAE:")
    print(tabla_wmae.to_string(index=False))
    print("\nFiguras generadas:")
    for nombre, ruta in rutas_generadas.items():
        print(f"- {nombre}: {ruta}")


if __name__ == "__main__":
    main()
