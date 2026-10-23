"""Project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
DUCKDB_PATH = DATA_PROCESSED / "home_credit.duckdb"

KAGGLE_COMPETITION = "home-credit-default-risk"
TARGET = "TARGET"   # 1 = payment default, 0 = repaid
ID = "SK_ID_CURR"   # client identifier
SEED = 42

# Expected tables in data/raw after download
TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "POS_CASH_balance",
]
