import pandas as pd
from walmart_sales_forecast.dataset import DataLoader, DataIntegrador
from walmart_sales_forecast.eda import VisualizadorEDA

def probarEda():
    print("1. Cargando datos con DataLoader...")
    cargador = DataLoader()
    
    try:
        train = cargador.cargar_train()
        features = cargador.cargar_features()
        stores = cargador.cargar_stores()
    except Exception as e:
        print(f"Aviso: No se encontraron los archivos CSV reales ({e}). Usando DataFrame sintético de prueba...")
        # DataFrame sintético por si no tienes los CSVs a la mano
        train = pd.DataFrame({
            "Store": [1, 1, 2, 2, 1], "Dept": [1, 1, 1, 1, 2],
            "Date": ["2010-02-05", "2010-02-05", "2010-02-12", "2010-02-19", "2010-02-26"],
            "Weekly_Sales": [24924.50, 24924.50, 46039.49, 41595.55, 19403.54],
            "IsHoliday": [False, False, True, False, False]
        })
        features = pd.DataFrame({
            "Store": [1, 2], "Date": ["2010-02-05", "2010-02-12"],
            "Temperature": [42.31, 38.51], "Fuel_Price": [2.572, 2.548],
            "CPI": [211.096, 211.242], "Unemployment": [8.106, 8.106],
            "IsHoliday": [False, True]
        })
        stores = pd.DataFrame({
            "Store": [1, 2], "Type": ["A", "B"], "Size": [151315, 202307]
        })

    print("2. Integrando datasets...")
    integrador = DataIntegrador()
    dataset_combinado = integrador.combinar_datasets(train, features, stores)

    print("3. Ejecutando Visualizador EDA y guardando reportes...")
    eda = VisualizadorEDA(dataset_combinado)
    
    # Imprime reporte de calidad en consola
    eda.reporteCalidadDatos()
    
    # Genera y guarda los gráficos PNG en la carpeta 'reports/'
    eda.plotDistribucionVentas()
    eda.plotVentasPorTiempo()
    eda.plotCorrelacionNumericas()

    print("\n¡Prueba finalizada con éxito! Revisa la carpeta 'reports/' para ver tus gráficos .png")

if __name__ == "__main__":
    probarEda()