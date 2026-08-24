"""
UBICACIÓN: walmart_sales_forecast/eda.py
VisualizadorEDA: consumo de datos preparados para generar 
reportes de calidad y gráficas estadísticas en formato PNG.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from walmart_sales_forecast.preparacion import PreparadorEDA


class VisualizadorEDA:
    """
    Clase encargada de ejecutar un Análisis Exploratorio de Datos (EDA) completo,
    evaluando calidad de datos, sesgos, distribuciones y guardando gráficos en PNG.
    """

    def __init__(self, df: pd.DataFrame, directorio_salida: str = "reports"):
        self.df_original = df.copy()
        self.directorio_salida = directorio_salida
        
        # Crear la carpeta reports si no existe
        os.makedirs(self.directorio_salida, exist_ok=True)
        
        # Instanciamos el preparador específico de EDA
        self.preparador = PreparadorEDA()
        
        # Preparamos los datos (formatos, duplicados y nulos)
        self.df_analisis = self.preparador.prepararParaEda(self.df_original)

    def reporteCalidadDatos(self):
        """Muestra un reporte preliminar de duplicados, nulos y tipos de datos."""
        print("==============================")
        print("   REPORTE DE CALIDAD DE DATOS")
        print("==============================")
        
        duplicados = self.preparador.detectarDuplicados(self.df_original)
        print(f"• Filas duplicadas en datos crudos: {duplicados}")
        
        print("\n• Valores nulos restantes por columna:")
        nulos = self.df_analisis.isnull().sum()
        print(nulos[nulos > 0] if nulos.sum() > 0 else "  No se encontraron valores nulos críticos.")
        
        print("\n• Dimensiones del dataset limpio para EDA:", self.df_analisis.shape)
        print("==============================\n")

    def plotDistribucionVentas(self):
        """
        Grafica la distribución y el histograma de Weekly_Sales 
        para evidenciar el sesgo (skewness) y los outliers, guardándolo en PNG.
        """
        if "Weekly_Sales" not in self.df_analisis.columns:
            print("La columna 'Weekly_Sales' no está presente en el dataset.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Histograma con KDE (Sesgo)
        sns.histplot(
            self.df_analisis["Weekly_Sales"], 
            kde=True, 
            ax=axes[0], 
            color="teal", 
            bins=50
        )
        axes[0].set_title("Histograma y Densidad de Weekly_Sales (Sesgo)")
        axes[0].set_xlabel("Ventas Semanales")
        axes[0].set_ylabel("Frecuencia")

        # 2. Boxplot (Outliers)
        sns.boxplot(
            x=self.df_analisis["Weekly_Sales"], 
            ax=axes[1], 
            color="orange"
        )
        axes[1].set_title("Boxplot de Weekly_Sales (Outliers)")
        axes[1].set_xlabel("Ventas Semanales")

        plt.tight_layout()
        
        ruta_archivo = os.path.join(self.directorio_salida, "distribucion_weekly_sales.png")
        plt.savefig(ruta_archivo, dpi=300)
        plt.close()
        print(f"[EDA] Gráfico guardado con éxito en: {ruta_archivo}")

    def plotVentasPorTiempo(self):
        """Grafica la evolución temporal global de las ventas y la guarda en PNG."""
        if "Date" not in self.df_analisis.columns or "Weekly_Sales" not in self.df_analisis.columns:
            return

        plt.figure(figsize=(12, 5))
        ventas_temporales = self.df_analisis.groupby("Date")["Weekly_Sales"].sum().reset_index()
        sns.lineplot(data=ventas_temporales, x="Date", y="Weekly_Sales", color="dodgerblue")
        plt.title("Tendencia Histórica Global de Ventas Semanales")
        plt.xlabel("Fecha")
        plt.ylabel("Ventas Totales")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        ruta_archivo = os.path.join(self.directorio_salida, "tendencia_ventas_tiempo.png")
        plt.savefig(ruta_archivo, dpi=300)
        plt.close()
        print(f"[EDA] Gráfico guardado con éxito en: {ruta_archivo}")

    def plotCorrelacionNumericas(self):
        """Mapa de calor de correlaciones y lo guarda en PNG."""
        plt.figure(figsize=(8, 6))
        
        columnas = [c for c in self.preparador.columnas_numericas if c in self.df_analisis.columns]
        if "Weekly_Sales" in self.df_analisis.columns:
            columnas.append("Weekly_Sales")
            
        matriz_corr = self.df_analisis[columnas].corr()
        sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
        plt.title("Matriz de Correlación de Variables Numéricas")
        plt.tight_layout()
        
        ruta_archivo = os.path.join(self.directorio_salida, "matriz_correlacion.png")
        plt.savefig(ruta_archivo, dpi=300)
        plt.close()
        print(f"[EDA] Gráfico guardado con éxito en: {ruta_archivo}")