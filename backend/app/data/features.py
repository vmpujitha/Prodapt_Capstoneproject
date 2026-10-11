import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

DEFAULT_THRESHOLDS = {
    "ph": {"acidic_max": 6.0, "neutral_max": 7.5},
    "n": {"low_max": 60.0, "medium_max": 100.0},
    "p": {"low_max": 40.0, "medium_max": 60.0},
    "k": {"low_max": 35.0, "medium_max": 55.0},
    "temperature": {"cool_max": 22.0, "moderate_max": 28.0},
    "rainfall": {"low_max": 450.0, "moderate_max": 900.0},
    "humidity": {"low_max": 55.0, "moderate_max": 75.0}
}

def compute_and_save_thresholds(df: pd.DataFrame, output_path: str) -> Dict[str, Any]:
    """
    Computes tertile cut-offs for numerical conditions from the dataset
    and saves them as JSON for consistent classification across queries.
    """
    thresholds = {
        "ph": {"acidic_max": 6.0, "neutral_max": 7.5}
    }

    for col, name in [
        ("n", "n"),
        ("p", "p"),
        ("k", "k"),
        ("temperature", "temperature"),
        ("rainfall", "rainfall"),
        ("humidity", "humidity")
    ]:
        if col in df.columns:
            q33 = float(df[col].quantile(0.333))
            q66 = float(df[col].quantile(0.666))
            if name in ["temperature"]:
                thresholds[name] = {"cool_max": round(q33, 2), "moderate_max": round(q66, 2)}
            else:
                thresholds[name] = {"low_max": round(q33, 2), "medium_max": round(q66, 2)}
        else:
            thresholds[name] = DEFAULT_THRESHOLDS.get(name, {})

    if "crop_year" in df.columns:
        thresholds["crop_year"] = {"min": int(df["crop_year"].min()), "max": int(df["crop_year"].max())}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    return thresholds

def load_thresholds(thresholds_path: str) -> Dict[str, Any]:
    """Loads thresholds JSON if present, else defaults."""
    path = Path(thresholds_path)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_THRESHOLDS

def classify_value_by_tertile(val: float | None, low_max: float, med_max: float, labels: Tuple[str, str, str] = ("low", "medium", "high")) -> str | None:
    if val is None or np.isnan(val):
        return None
    if val < low_max:
        return labels[0]
    elif val <= med_max:
        return labels[1]
    else:
        return labels[2]

def classify_ph(val: float | None, acidic_max: float = 6.0, neutral_max: float = 7.5) -> str | None:
    if val is None or np.isnan(val):
        return None
    if val < acidic_max:
        return "acidic"
    elif val <= neutral_max:
        return "neutral"
    else:
        return "alkaline"

def add_derived_features(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds derived classes and within-crop yield percentiles to the DataFrame.
    """
    df = df.copy()

    # pH Class
    ph_cfg = thresholds.get("ph", DEFAULT_THRESHOLDS["ph"])
    df["ph_class"] = df["ph"].apply(lambda v: classify_ph(v, ph_cfg.get("acidic_max", 6.0), ph_cfg.get("neutral_max", 7.5)))

    # N, P, K levels
    n_cfg = thresholds.get("n", DEFAULT_THRESHOLDS["n"])
    df["n_level"] = df["n"].apply(lambda v: classify_value_by_tertile(v, n_cfg.get("low_max", 60.0), n_cfg.get("medium_max", 100.0)))

    p_cfg = thresholds.get("p", DEFAULT_THRESHOLDS["p"])
    df["p_level"] = df["p"].apply(lambda v: classify_value_by_tertile(v, p_cfg.get("low_max", 40.0), p_cfg.get("medium_max", 60.0)))

    k_cfg = thresholds.get("k", DEFAULT_THRESHOLDS["k"])
    df["k_level"] = df["k"].apply(lambda v: classify_value_by_tertile(v, k_cfg.get("low_max", 35.0), k_cfg.get("medium_max", 55.0)))

    # Weather classes
    temp_cfg = thresholds.get("temperature", DEFAULT_THRESHOLDS["temperature"])
    df["temp_class"] = df["temperature"].apply(
        lambda v: classify_value_by_tertile(v, temp_cfg.get("cool_max", 22.0), temp_cfg.get("moderate_max", 28.0), ("cool", "moderate", "hot"))
    )

    rain_cfg = thresholds.get("rainfall", DEFAULT_THRESHOLDS["rainfall"])
    df["rainfall_class"] = df["rainfall"].apply(
        lambda v: classify_value_by_tertile(v, rain_cfg.get("low_max", 450.0), rain_cfg.get("moderate_max", 900.0), ("low", "moderate", "high"))
    )

    hum_cfg = thresholds.get("humidity", DEFAULT_THRESHOLDS["humidity"])
    df["humidity_class"] = df["humidity"].apply(
        lambda v: classify_value_by_tertile(v, hum_cfg.get("low_max", 55.0), hum_cfg.get("moderate_max", 75.0), ("low", "moderate", "high"))
    )

    # Yield percentile within crop
    df["yield_percentile_within_crop"] = df.groupby("crop")["yield"].rank(pct=True) * 100.0
    df["yield_percentile_within_crop"] = df["yield_percentile_within_crop"].round(1)

    return df
