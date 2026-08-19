import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .dataset import WalmartDataLoader

def main():
    print("--- Iniciando Análisis Exploratorio de Datos (EDA) ---")
    loader = WalmartDataLoader(ruta_carpeta="data/raw/")
    df_minable = loader.obtener_tabla_limpia()
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Gráfica 1: Distribución
    plt.subplot(2, 2, 1)
    sns.histplot(df_minable['Weekly_Sales'], bins=40, kde=True, color='teal')
    plt.title('Distribución de Ventas Semanales')
    plt.xlabel('Ventas ($)')
    plt.ylabel('Frecuencia')
    
    # Gráfica 2: Tendencia temporal
    plt.subplot(2, 2, 2)
    df_tiempo = df_minable.groupby('Date')['Weekly_Sales'].sum().reset_index()
    plt.plot(df_tiempo['Date'], df_tiempo['Weekly_Sales'], color='darkorange', linewidth=1.5)
    plt.title('Ventas Totales de Walmart por Semana')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Totales ($)')
    plt.xticks(rotation=30)
    
    # Gráfica 3: Correlación
    plt.subplot(2, 2, 3)
    columnas_numericas = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    columnas_validas = [c for c in columnas_numericas if c in df_minable.columns]
    matriz_corr = df_minable[columnas_validas].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='Blues', fmt=".2f", cbar=False)
    plt.title('Matriz de Correlación')
    
    plt.tight_layout()
    archivo_imagen = 'eda_reporte_walmart.png'
    plt.savefig(archivo_imagen, dpi=300)
    plt.close()
    print(f"¡EDA completado! Gráfica guardada como: '{archivo_imagen}'")

if __name__ == "__main__":
    main()