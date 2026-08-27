"""
UBICACIÓN: walmart_sales_forecast/config.py

Configuración general del proyecto Walmart.
"""

from pathlib import Path

# =============================================================
# RAÍZ DEL PROYECTO
# =============================================================

PROJ_ROOT = Path(__file__).resolve().parents[1]


# =============================================================
# RUTAS DE DATOS
# =============================================================

DATA_DIR = PROJ_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# =============================================================
# RUTAS DE MODELOS
# =============================================================

MODELS_DIR = PROJ_ROOT / "models"


# =============================================================
# RUTAS DE REPORTES Y FIGURAS
# =============================================================

REPORTS_DIR = PROJ_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"


# =============================================================
# NOMBRES DE ARCHIVOS
# =============================================================

RUTA_TRAIN = RAW_DATA_DIR / "train.csv"

RUTA_TEST = RAW_DATA_DIR / "test.csv"

RUTA_FEATURES = RAW_DATA_DIR / "features.csv"

RUTA_STORES = RAW_DATA_DIR / "stores.csv"

RUTA_DATASET_COMBINADO = (
    INTERIM_DATA_DIR / "dataset_combinado.csv"
)

RUTA_TRAIN_PROCESADO = (
    PROCESSED_DATA_DIR / "train_procesado.csv"
)

RUTA_TEST_PROCESADO = (
    PROCESSED_DATA_DIR / "test_procesado.csv"
)


# =============================================================
# PARÁMETROS DEL PROYECTO
# =============================================================

class ConfiguracionProyecto:
    """
    Clase para acceder a los parámetros del proyecto.
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
        """
        Obtiene un parámetro mediante su nombre.
        """

        return self.parametros.get(clave)


config = ConfiguracionProyecto()