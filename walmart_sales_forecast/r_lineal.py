import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from .dataset import WalmartDataLoader

def main():
    print("--- Iniciando modelo Regresión Lineal (Multivariable) ---")
    
    try:
        # 1. Carga de datos usando tu DataLoader (aquí sí podemos usar las features)
        loader = WalmartDataLoader(ruta_carpeta="data/raw/")
        df = loader.obtener_tabla_limpia() # Esto ya une train + features

        # 2. Ingeniería de variables simple
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

        # Seleccionamos variables numéricas relevantes
        features = ['Store', 'Year', 'Month', 'Week', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']
        target = 'Weekly_Sales'

        # Limpiamos nulos (importante para Regresión Lineal)
        df = df[features + [target]].dropna()

        X = df[features]
        y = df[target]

        # 3. División
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Entrenamiento
        print("Entrenando Regresión Lineal...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. Predicción y Evaluación
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        print("\n==========================================")
        print("    RESULTADOS - REGRESIÓN LINEAL        ")
        print("==========================================")
        print(f"📉 MAE de Regresión Lineal: ${mae:,.2f}")
        print("==========================================")
        
    except Exception as e:
        print(f"❌ Error en Regresión Lineal: {e}")

if __name__ == "__main__":
    main()