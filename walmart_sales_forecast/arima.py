import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def main():
    print("--- Iniciando modelo ARIMA (Serie Temporal Global) ---")
    
    try:
        print("[1/5] Cargando train.csv directamente para la serie temporal...")
        ruta_train = os.path.join("data", "raw", "train.csv")
        df_train = pd.read_csv(ruta_train)

        print("[2/5] Agrupando y preparando la serie temporal...")
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        df_global = df_train.groupby('Date')['Weekly_Sales'].sum().reset_index()
        df_global = df_global.sort_values('Date')
        df_global.set_index('Date', inplace=True)

        # Asegurar frecuencia semanal y rellenar nulos limpiamente
        serie_ventas = df_global['Weekly_Sales'].asfreq('W').ffill().bfill().fillna(0)
        
        # Escalar a millones para estabilidad numérica
        serie_ventas_millones = serie_ventas / 1e6

        # División train/test (80% / 20%)
        train_size = int(len(serie_ventas_millones) * 0.8)
        train, test = serie_ventas_millones.iloc[:train_size], serie_ventas_millones.iloc[train_size:]

        print(f"[3/5] Entrenando ARIMA con {len(train)} semanas de entrenamiento...")
        # Usamos order=(1, 0, 0) para evitar problemas de parámetros iniciales con pocos datos
        modelo = ARIMA(
            train, 
            order=(1, 0, 0), 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        modelo_ajustado = modelo.fit()

        print("[4/5] Generando pronósticos...")
        predicciones_millones = modelo_ajustado.forecast(steps=len(test))
        
        # Revertir escala a dólares reales
        test_real_vals = test.values * 1e6
        
        if hasattr(predicciones_millones, 'values'):
            predicciones_vals = predicciones_millones.values * 1e6
        else:
            predicciones_vals = np.array(predicciones_millones) * 1e6

        # Blindaje contra cualquier NaN residual
        test_real_vals = np.nan_to_num(test_real_vals, nan=0.0)
        predicciones_vals = np.nan_to_num(predicciones_vals, nan=0.0)

        print("[5/5] Calculando métricas finales...")
        mae_arima = mean_absolute_error(test_real_vals, predicciones_vals)

        print("\n==========================================")
        print("      RESULTADOS FINALES - ARIMA         ")
        print("==========================================")
        print(f"📉 MAE Global de ARIMA: ${mae_arima:,.2f}")
        print("==========================================")
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error detallado en ARIMA: {e}")

if __name__ == "__main__":
    main()