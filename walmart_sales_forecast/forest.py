import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    print("--- Iniciando Random Forest con 'Lag Features' (Memoria Histórica) ---")
    
    try:
        # Carga
        df_train = pd.read_csv("data/raw/train.csv")
        df_features = pd.read_csv("data/raw/features.csv")
        df_stores = pd.read_csv("data/raw/stores.csv")

        # Merge
        df = pd.merge(df_train, df_features, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = pd.merge(df, df_stores, on='Store', how='left')
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Store', 'Date']) # Obligatorio para los lags

        # --- CREACIÓN DE LAG FEATURES ---
        print("[1/5] Creando variables de memoria (Lag 1 y 52 semanas)...")
        # Ventas de hace 1 semana y de hace 52 semanas por cada tienda
        df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
        df['Sales_Lag_52'] = df.groupby('Store')['Weekly_Sales'].shift(52)

        # Ingeniería extra
        df['Type'] = df['Type'].map({'A': 3, 'B': 2, 'C': 1})
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

        features = [
            'Store', 'Type', 'Size', 'Month', 'Week', 'Temperature', 
            'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 
            'Sales_Lag_1', 'Sales_Lag_52' # Nuestras nuevas armas
        ]
        target = 'Weekly_Sales'

        # Limpiamos nulos (aquí se eliminan las primeras 52 semanas por falta de datos previos)
        df = df[features + [target]].dropna()

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[3/5] Entrenando Random Forest con memoria histórica...")
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=20, 
            min_samples_split=10, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        print("[4/5] Prediciendo...")
        predictions = model.predict(X_test)

        # Evaluación
        weights = np.where(X_test['IsHoliday'] == True, 5, 1)
        wmae = np.sum(weights * np.abs(y_test - predictions)) / np.sum(weights)

        print("\n==========================================")
        print("    RESULTADOS - RANDOM FOREST AVANZADO   ")
        print("==========================================")
        print(f"📉 WMAE Final (con Lags): ${wmae:,.2f}")
        print("==========================================")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()