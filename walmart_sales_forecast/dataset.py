import pandas as pd
import os

class WalmartDataLoader:
    def __init__(self, ruta_carpeta="data/raw/"):
        self.ruta = ruta_carpeta

    def obtener_tabla_limpia(self):
        """Carga, limpia y fusiona train, features y stores en una sola tabla minable."""
        train = pd.read_csv(os.path.join(self.ruta, 'train.csv'))
        features = pd.read_csv(os.path.join(self.ruta, 'features.csv'))
        stores = pd.read_csv(os.path.join(self.ruta, 'stores.csv'))

        # Unir tablas
        df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores, on='Store', how='left')

        # Limpieza universal de nulos
        markdown_cols = [f'MarkDown{i}' for i in range(1, 6)]
        df[markdown_cols] = df[markdown_cols].fillna(0)
        df = df.dropna()

        # Formato de fecha
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df