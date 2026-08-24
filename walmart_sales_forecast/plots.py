"""
UBICACIÓN:
walmart_sales_forecast/plots.py

Módulo de visualización del proyecto Walmart.

Responsabilidades:

- Visualizar la evolución temporal de las ventas.
- Comparar valores reales contra predicciones.
- Comparar el WMAE de los modelos.

El módulo NO entrena modelos.
El módulo NO realiza preprocessing.
El módulo NO calcula métricas.

Recibe los resultados ya calculados por el pipeline
y se encarga únicamente de generar las figuras.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from walmart_sales_forecast.config import FIGURES_DIR


# =============================================================
# CONFIGURACIÓN
# =============================================================

def preparar_directorio_salida(
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Crea el directorio de salida de las figuras si no existe.

    Parameters
    ----------
    output_dir : Path
        Directorio donde se guardarán las figuras.

    Returns
    -------
    Path
        Directorio de salida.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =============================================================
# 1. VENTAS TEMPORALES
# =============================================================

def plot_ventas_temporales(
    datos: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    nombre_archivo: str = "ventas_temporales.png",
):
    """
    Genera un gráfico de la evolución temporal de las ventas.

    Parameters
    ----------
    datos : pd.DataFrame
        DataFrame que debe contener:
            - Date
            - Weekly_Sales

    output_dir : Path
        Directorio donde se guardará la figura.

    nombre_archivo : str
        Nombre del archivo de salida.

    Returns
    -------
    Path
        Ruta de la figura generada.
    """

    columnas_requeridas = {
        "Date",
        "Weekly_Sales",
    }

    faltantes = (
        columnas_requeridas
        - set(datos.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{faltantes}"
        )

    df = datos.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # ---------------------------------------------------------
    # Agregar ventas por fecha
    # ---------------------------------------------------------

    ventas = (
        df.groupby("Date")[
            "Weekly_Sales"
        ]
        .sum()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Crear directorio
    # ---------------------------------------------------------

    output_dir = preparar_directorio_salida(
        output_dir
    )

    ruta_salida = (
        output_dir / nombre_archivo
    )

    # ---------------------------------------------------------
    # Gráfico
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        ventas.index,
        ventas.values,
    )

    plt.title(
        "Evolución temporal de las ventas"
    )

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Ventas semanales"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return ruta_salida


# =============================================================
# 2. REAL VS PREDICHO
# =============================================================

def plot_real_vs_predicho(
    fechas,
    valores_reales,
    predicciones,
    nombre_modelo: str,
    output_dir: Path = FIGURES_DIR,
    nombre_archivo: str | None = None,
):
    """
    Genera un gráfico comparando valores reales y predicciones.

    Parameters
    ----------
    fechas : array-like
        Fechas correspondientes a validation.

    valores_reales : array-like
        Valores reales.

    predicciones : array-like
        Valores predichos por el modelo.

    nombre_modelo : str
        Nombre del modelo.

    output_dir : Path
        Directorio donde se guardará la figura.

    nombre_archivo : str, optional
        Nombre personalizado del archivo.

    Returns
    -------
    Path
        Ruta de la figura generada.
    """

    if nombre_archivo is None:
        nombre_archivo = (
            f"real_vs_predicho_"
            f"{nombre_modelo.lower()}.png"
        )

    # ---------------------------------------------------------
    # Convertir a Series para manejar correctamente las fechas
    # ---------------------------------------------------------

    fechas = pd.to_datetime(
        fechas
    )

    valores_reales = pd.Series(
        valores_reales
    )

    predicciones = pd.Series(
        predicciones
    )

    if not (
        len(fechas)
        == len(valores_reales)
        == len(predicciones)
    ):
        raise ValueError(
            "Las fechas, valores reales y predicciones "
            "deben tener la misma longitud."
        )

    # ---------------------------------------------------------
    # Directorio
    # ---------------------------------------------------------

    output_dir = preparar_directorio_salida(
        output_dir
    )

    ruta_salida = (
        output_dir / nombre_archivo
    )

    # ---------------------------------------------------------
    # Gráfico
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        fechas,
        valores_reales,
        marker="o",
        label="Real",
    )

    plt.plot(
        fechas,
        predicciones,
        marker="o",
        label="Predicción",
    )

    plt.title(
        f"Valores reales vs predicción - {nombre_modelo}"
    )

    plt.xlabel(
        "Fecha"
    )

    plt.ylabel(
        "Weekly Sales"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return ruta_salida


# =============================================================
# 3. COMPARACIÓN WMAE
# =============================================================

def plot_comparacion_wmae(
    resultados: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    nombre_archivo: str = "comparacion_wmae.png",
):
    """
    Genera un gráfico de comparación del WMAE entre modelos.

    Parameters
    ----------
    resultados : pd.DataFrame
        DataFrame con las columnas:
            - Modelo
            - WMAE

    output_dir : Path
        Directorio donde se guardará la figura.

    nombre_archivo : str
        Nombre del archivo de salida.

    Returns
    -------
    Path
        Ruta de la figura generada.
    """

    columnas_requeridas = {
        "Modelo",
        "WMAE",
    }

    faltantes = (
        columnas_requeridas
        - set(resultados.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{faltantes}"
        )

    df = resultados.copy()

    df = df.sort_values(
        "WMAE"
    )

    # ---------------------------------------------------------
    # Directorio
    # ---------------------------------------------------------

    output_dir = preparar_directorio_salida(
        output_dir
    )

    ruta_salida = (
        output_dir / nombre_archivo
    )

    # ---------------------------------------------------------
    # Gráfico
    # ---------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        df["Modelo"],
        df["WMAE"],
    )

    plt.title(
        "Comparación de modelos mediante WMAE"
    )

    plt.xlabel(
        "Modelo"
    )

    plt.ylabel(
        "WMAE"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return ruta_salida