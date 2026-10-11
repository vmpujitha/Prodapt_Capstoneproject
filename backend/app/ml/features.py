"""
Shared feature handling for the yield and risk models: feature lists, per-crop
default values for fields the user did not supply, and a time-aware train/test split.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = [
    "n", "p", "k", "ph", "temperature", "rainfall", "humidity",
    "fertilizer", "pesticide", "area", "crop_year"
]
CATEGORICAL_FEATURES = ["crop", "season", "state"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

GLOBAL_KEY = "__global__"


def build_preprocessor() -> ColumnTransformer:
    """One-hot encodes the categoricals and passes numeric features through."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def compute_feature_defaults(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Per-crop medians (numeric) and modes (season/state) used to fill fields a request
    leaves out, plus a dataset-wide entry under GLOBAL_KEY.
    """
    def summarize(frame: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col in NUMERICAL_FEATURES:
            if col in frame:
                out[col] = float(frame[col].median())
        out["crop_year"] = int(frame["crop_year"].max())
        for col in ("season", "state"):
            if col in frame and not frame[col].dropna().empty:
                out[col] = str(frame[col].mode().iloc[0])
        return out

    defaults = {GLOBAL_KEY: summarize(df)}
    for crop, group in df.groupby("crop"):
        defaults[str(crop)] = summarize(group)
    return defaults


def build_feature_row(data: Dict[str, Any], defaults: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a complete model input row: values present in `data` win, missing or None
    values come from the crop's defaults, then the dataset-wide defaults.
    """
    crop = str(data.get("crop") or "")
    crop_defaults = defaults.get(crop, {})
    global_defaults = defaults.get(GLOBAL_KEY, {})

    row: Dict[str, Any] = {"crop": crop}
    for col in ("season", "state"):
        value = data.get(col)
        row[col] = str(value) if value else str(crop_defaults.get(col, global_defaults.get(col, "")))
    for col in NUMERICAL_FEATURES:
        value = data.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = crop_defaults.get(col, global_defaults.get(col, 0.0))
        row[col] = float(value)
    row["crop_year"] = int(row["crop_year"])
    return row


def time_aware_split(df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.Index, pd.Index, str]:
    """
    Holds out the most recent crop years (about `test_fraction` of rows) as the test set,
    so metrics reflect forecasting later seasons from earlier ones. Falls back to a seeded
    random split when there are too few distinct years.
    """
    years = np.sort(df["crop_year"].unique())
    if len(years) >= 5:
        cutoff: Optional[int] = None
        for year in years[::-1]:
            if (df["crop_year"] >= year).mean() >= test_fraction:
                cutoff = int(year)
                break
        if cutoff is not None and cutoff > years[0]:
            test_mask = df["crop_year"] >= cutoff
            return df.index[~test_mask], df.index[test_mask], f"time-aware: train < {cutoff}, test >= {cutoff}"

    shuffled = df.sample(frac=1.0, random_state=42).index
    n_test = int(len(df) * test_fraction)
    return shuffled[n_test:], shuffled[:n_test], "random 80/20 (too few distinct years for a time split)"
