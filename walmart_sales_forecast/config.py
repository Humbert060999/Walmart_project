"""
UBICACIÓN: walmart_forecast/config.py
(ccds ya te crea este archivo vacío/parcial en la raíz de tu paquete;
 reemplaza su contenido por esto, o agrégalo si no existe todavía)

Guarda en un solo lugar las rutas y parámetros del proyecto,
para no repetir "data/raw/train.csv" por todo el código.
"""

from pathlib import Path

# Raíz del proyecto (ccds ya suele definir PROJ_ROOT similar a esto)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# --- Rutas de datos ---
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Rutas de modelos entrenados ---
MODELS_DIR = PROJ_ROOT / "models"

# --- Nombres de archivo ---
RUTA_TRAIN = RAW_DATA_DIR / "train.csv"
RUTA_TEST = RAW_DATA_DIR / "test.csv"
RUTA_FEATURES = RAW_DATA_DIR / "features.csv"
RUTA_STORES = RAW_DATA_DIR / "stores.csv"

RUTA_DATASET_COMBINADO = INTERIM_DATA_DIR / "dataset_combinado.csv"
RUTA_TRAIN_PROCESADO = PROCESSED_DATA_DIR / "train_procesado.csv"
RUTA_TEST_PROCESADO = PROCESSED_DATA_DIR / "test_procesado.csv"

# --- Parámetros del proyecto ---
class ConfiguracionProyecto:
    """
    Clase simple para acceder a parámetros del proyecto desde cualquier
    módulo. Puedes ampliarla para leer un .yaml si más adelante quieres
    parametrizar hiperparámetros desde fuera del código.
    """

    def __init__(self):
        self.parametros = {
            "peso_semana_feriado": 5,
            "peso_semana_normal": 1,
            "n_estimadores_rf": 200,
            "profundidad_maxima_rf": 12,
            "semilla_aleatoria": 42,
            "orden_arima": (1, 1, 1),
        }

    def obtener(self, clave):
        return self.parametros.get(clave)


config = ConfiguracionProyecto()
