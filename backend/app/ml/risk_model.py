import os
import logging
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from backend.app.config import settings
from backend.app.ml.features import (
    ALL_FEATURES,
    build_preprocessor,
    build_feature_row,
    compute_feature_defaults,
    time_aware_split,
)
from backend.app.ml.yield_model import _update_metrics_file, _trained_on_label, _matches_active_dataset
from backend.app.schemas.prediction import RiskAssessment, RiskFactor

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 2
ENVELOPE_FIELDS = ["n", "p", "k", "ph", "temperature", "rainfall", "humidity"]


def get_base_classifier():
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, learning_rate=0.06, max_depth=5, random_state=42,
                            n_jobs=-1, eval_metric="logloss")
        return clf, "XGBoost"
    except Exception as e:
        logger.info(f"Using Scikit-Learn GradientBoostingClassifier (XGBoost unavailable: {e})")
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.06, max_depth=4,
                                          random_state=42), "GradientBoosting"


class CropRiskPredictor:
    """
    Calibrated classifier estimating P(yield < crop's 25th percentile), combined with a
    rule-based check against each crop's historical 10th-90th percentile soil/weather envelope.
    """
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.crop_envelopes: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.crop_q25: Dict[str, float] = {}
        self.feature_defaults: Dict[str, Dict[str, Any]] = {}
        self.model_type: str = "GradientBoosting"
        self.metrics: Dict[str, Any] = {}

    def train(self, df: pd.DataFrame, models_dir: str) -> Dict[str, Any]:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        df = df.reset_index(drop=True).copy()

        self.feature_defaults = compute_feature_defaults(df)
        self.crop_q25 = {str(k): float(v) for k, v in df.groupby("crop")["yield"].quantile(0.25).items()}
        self.crop_envelopes = {}
        for crop, group in df.groupby("crop"):
            self.crop_envelopes[str(crop)] = {
                col: (round(float(group[col].quantile(0.10)), 2), round(float(group[col].quantile(0.90)), 2))
                for col in ENVELOPE_FIELDS
            }

        df["is_low_yield"] = (df["yield"] < df["crop"].map(self.crop_q25)).astype(int)
        train_idx, test_idx, split_desc = time_aware_split(df)
        X_train, X_test = df.loc[train_idx, ALL_FEATURES], df.loc[test_idx, ALL_FEATURES]
        y_train, y_test = df.loc[train_idx, "is_low_yield"], df.loc[test_idx, "is_low_yield"]

        base_clf, model_name = get_base_classifier()
        self.model_type = f"{model_name} + isotonic calibration"
        self.pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", CalibratedClassifierCV(base_clf, method="isotonic", cv=3)),
        ])

        logger.info(f"Training calibrated {model_name} risk model on {len(X_train)} rows ({split_desc})...")
        self.pipeline.fit(X_train, y_train)

        proba = self.pipeline.predict_proba(X_test)[:, 1]
        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, proba >= 0.5)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4) if y_test.nunique() > 1 else None,
            "brier_score": round(float(brier_score_loss(y_test, proba)), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "split": split_desc,
            "model_type": self.model_type,
            "trained_on": _trained_on_label(),
        }

        joblib.dump({
            "format_version": MODEL_FORMAT_VERSION,
            "pipeline": self.pipeline,
            "crop_envelopes": self.crop_envelopes,
            "crop_q25": self.crop_q25,
            "feature_defaults": self.feature_defaults,
            "model_type": self.model_type,
            "metrics": self.metrics,
        }, os.path.join(models_dir, "risk_model.joblib"))
        _update_metrics_file(models_dir, "risk_model", self.metrics)

        logger.info(f"Risk model trained: {self.metrics}")
        return self.metrics

    @property
    def trained_on(self) -> str:
        return self.metrics.get("trained_on", "unknown")

    def load(self, models_dir: str) -> bool:
        model_path = os.path.join(models_dir, "risk_model.joblib")
        if not os.path.exists(model_path):
            return False
        try:
            data = joblib.load(model_path)
            if data.get("format_version") != MODEL_FORMAT_VERSION:
                logger.info("Risk model artifact uses an older format; it will be retrained.")
                return False
            self.pipeline = data["pipeline"]
            self.crop_envelopes = data["crop_envelopes"]
            self.crop_q25 = data["crop_q25"]
            self.feature_defaults = data["feature_defaults"]
            self.model_type = data.get("model_type", "GradientBoosting")
            self.metrics = data.get("metrics", {})
            return True
        except Exception as e:
            logger.warning(f"Failed to load risk model: {e}")
            return False

    def assess_risk(self, data: Dict[str, Any], check_fields: Optional[Iterable[str]] = None) -> RiskAssessment:
        """
        Estimates the low-yield probability, a risk level, and the top contributing factors.
        Envelope checks consider the supplied values, restricted to `check_fields` if given
        (e.g. only the conditions a user actually stated).
        """
        allowed = set(check_fields) if check_fields is not None else None
        row = build_feature_row(data, self.feature_defaults)
        crop = row["crop"]
        prob_low_yield = float(self.pipeline.predict_proba(pd.DataFrame([row])[ALL_FEATURES])[0][1])

        envelope = self.crop_envelopes.get(crop, {})
        field_labels = {
            "rainfall": ("Rainfall", "mm"),
            "temperature": ("Temperature", "°C"),
            "ph": ("Soil pH", ""),
            "n": ("Nitrogen (N)", "kg/ha"),
            "p": ("Phosphorus (P)", "kg/ha"),
            "k": ("Potassium (K)", "kg/ha"),
            "humidity": ("Humidity", "%"),
        }

        scored: List[Tuple[float, RiskFactor]] = []
        for field, (label, unit) in field_labels.items():
            if data.get(field) is None or field not in envelope or (allowed is not None and field not in allowed):
                continue
            val = float(data[field])
            low_b, high_b = envelope[field]
            if val < low_b:
                gap_pct = (low_b - val) / max(1e-6, abs(low_b)) * 100
                desc = (f"{label} ({val:g}{unit}) is below the typical {crop} range "
                        f"({low_b:g}-{high_b:g}{unit}); about {gap_pct:.0f}% short of the 10th percentile.")
            elif val > high_b:
                gap_pct = (val - high_b) / max(1e-6, abs(high_b)) * 100
                desc = (f"{label} ({val:g}{unit}) is above the typical {crop} range "
                        f"({low_b:g}-{high_b:g}{unit}); about {gap_pct:.0f}% over the 90th percentile.")
            else:
                continue
            impact = "high" if gap_pct > 30 else "medium"
            scored.append((gap_pct, RiskFactor(factor=label, impact=impact, description=desc)))

        scored.sort(key=lambda item: item[0], reverse=True)
        factors = [f for _, f in scored]
        high_factors = sum(1 for f in factors if f.impact == "high")

        if prob_low_yield < 0.25 and not factors:
            level = "low"
        elif prob_low_yield < 0.5 and high_factors == 0 and len(factors) <= 1:
            level = "medium"
        else:
            level = "high"

        return RiskAssessment(level=level, probability=round(prob_low_yield, 3), factors=factors[:3])


_GLOBAL_RISK_PREDICTOR: Optional[CropRiskPredictor] = None


def get_risk_predictor() -> CropRiskPredictor:
    """Returns singleton CropRiskPredictor, auto-training if absent."""
    global _GLOBAL_RISK_PREDICTOR
    if _GLOBAL_RISK_PREDICTOR is None:
        predictor = CropRiskPredictor()
        if not predictor.load(settings.MODELS_DIR) or not _matches_active_dataset(predictor):
            from backend.app.data.ingest import get_active_dataframe
            predictor.train(get_active_dataframe(), settings.MODELS_DIR)
        _GLOBAL_RISK_PREDICTOR = predictor
    return _GLOBAL_RISK_PREDICTOR


def reset_predictor_cache() -> None:
    """Drops the cached predictor so the next call reloads from disk (used after retraining)."""
    global _GLOBAL_RISK_PREDICTOR
    _GLOBAL_RISK_PREDICTOR = None
