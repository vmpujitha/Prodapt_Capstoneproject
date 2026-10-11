import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from backend.app.config import settings
from backend.app.data.schema import yield_unit
from backend.app.ml.features import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    GLOBAL_KEY,
    build_preprocessor,
    build_feature_row,
    compute_feature_defaults,
    time_aware_split,
)
from backend.app.schemas.prediction import YieldPredictionInterval

logger = logging.getLogger(__name__)

# Bumped whenever the persisted artifact layout changes; older files are retrained.
MODEL_FORMAT_VERSION = 2
# Minimum test residuals for a crop-specific prediction interval.
MIN_CROP_RESIDUALS = 15


def get_base_regressor():
    try:
        from xgboost import XGBRegressor
        reg = XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=6, subsample=0.9,
                           colsample_bytree=0.9, random_state=42, n_jobs=-1)
        return reg, "XGBoost"
    except Exception as e:
        logger.info(f"Using Scikit-Learn GradientBoostingRegressor (XGBoost unavailable: {e})")
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.06, max_depth=5,
                                         subsample=0.9, random_state=42), "GradientBoosting"


def _trained_on_label() -> str:
    from backend.app.data.ingest import is_synthetic_dataset
    return "synthetic" if is_synthetic_dataset() else "real_kaggle_dataset"


def _matches_active_dataset(predictor) -> bool:
    """False when a saved model was trained on synthetic data but real data is now active (or vice versa)."""
    if predictor.trained_on != _trained_on_label():
        logger.warning(f"Saved model was trained on '{predictor.trained_on}' data but the active dataset is "
                       f"'{_trained_on_label()}'; retraining.")
        return False
    return True


class CropYieldPredictor:
    """
    Gradient-boosted yield model trained on log1p(yield), so crops with very different
    yield scales (t/ha vs. nuts/ha) are fitted on comparable terms. Prediction intervals
    are the 10th-90th percentile of held-out log residuals, per crop when enough exist.
    """
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Dict[str, Any] = {}
        self.trained_on: str = "unknown"
        self.model_type: str = "GradientBoosting"
        self.feature_defaults: Dict[str, Dict[str, Any]] = {}
        self.residual_quantiles: Dict[str, Tuple[float, float]] = {}

    def train(self, df: pd.DataFrame, models_dir: str) -> Dict[str, Any]:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        df = df.reset_index(drop=True)

        self.feature_defaults = compute_feature_defaults(df)
        train_idx, test_idx, split_desc = time_aware_split(df)
        X_train, X_test = df.loc[train_idx, ALL_FEATURES], df.loc[test_idx, ALL_FEATURES]
        y_train, y_test = df.loc[train_idx, "yield"], df.loc[test_idx, "yield"]

        regressor, model_name = get_base_regressor()
        self.model_type = model_name
        self.pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("regressor", regressor)])

        logger.info(f"Training {model_name} yield model on {len(X_train)} rows ({split_desc})...")
        self.pipeline.fit(X_train, np.log1p(y_train))

        pred_log = self.pipeline.predict(X_test)
        preds = np.expm1(pred_log).clip(min=0)
        log_resid = np.log1p(y_test.to_numpy()) - pred_log

        self.residual_quantiles = {GLOBAL_KEY: self._quantiles(log_resid)}
        crops_test = df.loc[test_idx, "crop"].to_numpy()
        for crop in np.unique(crops_test):
            crop_resid = log_resid[crops_test == crop]
            if len(crop_resid) >= MIN_CROP_RESIDUALS:
                self.residual_quantiles[str(crop)] = self._quantiles(crop_resid)

        nonzero = y_test.to_numpy() > 0
        ape = np.abs(preds[nonzero] - y_test.to_numpy()[nonzero]) / y_test.to_numpy()[nonzero]
        self.trained_on = _trained_on_label()
        self.metrics = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
            "r2_log_scale": round(float(r2_score(np.log1p(y_test), pred_log)), 4),
            "median_abs_pct_error": round(float(np.median(ape)) * 100, 2) if len(ape) else None,
            "interval_coverage_10_90": round(float(np.mean(
                (log_resid >= self.residual_quantiles[GLOBAL_KEY][0]) & (log_resid <= self.residual_quantiles[GLOBAL_KEY][1])
            )), 4),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "split": split_desc,
            "target_transform": "log1p",
            "model_type": self.model_type,
            "trained_on": self.trained_on,
        }

        joblib.dump({
            "format_version": MODEL_FORMAT_VERSION,
            "pipeline": self.pipeline,
            "trained_on": self.trained_on,
            "model_type": self.model_type,
            "feature_defaults": self.feature_defaults,
            "residual_quantiles": self.residual_quantiles,
            "metrics": self.metrics,
        }, os.path.join(models_dir, "yield_model.joblib"))
        _update_metrics_file(models_dir, "yield_model", self.metrics)

        logger.info(f"Yield model trained: {self.metrics}")
        return self.metrics

    @staticmethod
    def _quantiles(residuals: np.ndarray) -> Tuple[float, float]:
        q10, q90 = np.quantile(residuals, [0.10, 0.90])
        return float(min(q10, 0.0)), float(max(q90, 0.0))

    def load(self, models_dir: str) -> bool:
        model_path = os.path.join(models_dir, "yield_model.joblib")
        if not os.path.exists(model_path):
            return False
        try:
            data = joblib.load(model_path)
            if data.get("format_version") != MODEL_FORMAT_VERSION:
                logger.info("Yield model artifact uses an older format; it will be retrained.")
                return False
            self.pipeline = data["pipeline"]
            self.trained_on = data.get("trained_on", "unknown")
            self.model_type = data.get("model_type", "GradientBoosting")
            self.feature_defaults = data["feature_defaults"]
            self.residual_quantiles = data["residual_quantiles"]
            self.metrics = data.get("metrics", {})
            return True
        except Exception as e:
            logger.warning(f"Failed to load yield model: {e}")
            return False

    def complete_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fills fields missing from `data` with the crop's historical medians/modes."""
        return build_feature_row(data, self.feature_defaults)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predicts yields (original scale) for a frame with all model features."""
        return np.expm1(self.pipeline.predict(frame[ALL_FEATURES])).clip(min=0)

    def predict_one(self, data: Dict[str, Any]) -> YieldPredictionInterval:
        """Predicts expected yield with a 10th-90th percentile interval."""
        row = self.complete_profile(data)
        pred_log = float(self.pipeline.predict(pd.DataFrame([row])[ALL_FEATURES])[0])
        q10, q90 = self.residual_quantiles.get(row["crop"], self.residual_quantiles[GLOBAL_KEY])
        value = max(0.0, float(np.expm1(pred_log)))
        low = max(0.0, float(np.expm1(pred_log + q10)))
        high = float(np.expm1(pred_log + q90))
        return YieldPredictionInterval(
            value=round(value, 2), low=round(low, 2), high=round(high, 2), unit=yield_unit(row["crop"])
        )


def _update_metrics_file(models_dir: str, key: str, metrics: Dict[str, Any]) -> None:
    metrics_file = os.path.join(models_dir, "metrics.json")
    existing: Dict[str, Any] = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning(f"Ignoring unreadable metrics file {metrics_file}: {e}")
    existing[key] = metrics
    with open(metrics_file, "w") as f:
        json.dump(existing, f, indent=2)


_GLOBAL_YIELD_PREDICTOR: Optional[CropYieldPredictor] = None


def get_yield_predictor() -> CropYieldPredictor:
    """Returns singleton CropYieldPredictor, auto-training if absent."""
    global _GLOBAL_YIELD_PREDICTOR
    if _GLOBAL_YIELD_PREDICTOR is None:
        predictor = CropYieldPredictor()
        if not predictor.load(settings.MODELS_DIR) or not _matches_active_dataset(predictor):
            from backend.app.data.ingest import get_active_dataframe
            predictor.train(get_active_dataframe(), settings.MODELS_DIR)
        _GLOBAL_YIELD_PREDICTOR = predictor
    return _GLOBAL_YIELD_PREDICTOR


def reset_predictor_cache() -> None:
    """Drops the cached predictor so the next call reloads from disk (used after retraining)."""
    global _GLOBAL_YIELD_PREDICTOR
    _GLOBAL_YIELD_PREDICTOR = None
