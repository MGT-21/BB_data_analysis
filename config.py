from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

def get_path(*paths):
    return BASE_DIR.joinpath(*paths)

# Pastas principais
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"
EDA_DATA_DIR = DATA_DIR / "eda"

# Pasta de saida (gráficos e indicadores)
OUTPUT_DIR = BASE_DIR / "output"

# Arquivos específicos
DATASET_PATH = RAW_DATA_DIR / "databridge_squad19_sintetico.csv"
DIC_PATH = RAW_DATA_DIR / "databridge_squad19_dictionary.csv"
MODELS_DIR     = BASE_DIR / "models"
RF_MODELS_DIR  = MODELS_DIR / "rf"
GBM_MODELS_DIR = MODELS_DIR / "gbm"

BRONZE_DATASET = BRONZE_DATA_DIR / "bronze_databridge.csv"
SILVER_DATASET = SILVER_DATA_DIR / "silver_databridge.csv"
GOLD_DASHBOARD = GOLD_DATA_DIR / "gold_dashboard.csv"
GOLD_MODEL     = GOLD_DATA_DIR / "gold_model.csv"