from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "loan_eligibility.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "loan_eligibility_model.joblib"

LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

TEST_SIZE = 0.2
RANDOM_STATE = 42