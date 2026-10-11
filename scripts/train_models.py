import os
import sys
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.data.ingest import get_active_dataframe
from backend.app.ml.yield_model import CropYieldPredictor
from backend.app.ml.risk_model import CropRiskPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_all_models():
    logger.info("Loading active processed dataset for model training...")
    df = get_active_dataframe()
    logger.info(f"Loaded {len(df)} samples.")

    models_dir = settings.MODELS_DIR
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    logger.info("1. Training Crop Yield Regression Model (XGBoost)...")
    yield_pred = CropYieldPredictor()
    y_metrics = yield_pred.train(df, models_dir)
    logger.info(f"Yield Model metrics: {y_metrics}")

    logger.info("2. Training Crop Risk Classifier (XGBoost + Historical Envelopes)...")
    risk_pred = CropRiskPredictor()
    r_metrics = risk_pred.train(df, models_dir)
    logger.info(f"Risk Model metrics: {r_metrics}")

    logger.info("All classical ML models successfully trained and persisted.")

if __name__ == "__main__":
    train_all_models()
