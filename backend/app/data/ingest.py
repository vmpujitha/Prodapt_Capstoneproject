import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from langchain_core.documents import Document

from backend.app.config import settings
from backend.app.data.schema import standardize_columns, CANONICAL_COLUMNS, yield_unit
from backend.app.data.features import compute_and_save_thresholds, add_derived_features

logger = logging.getLogger(__name__)

# Rows whose yield exceeds this multiple of their crop's median yield are treated as data errors.
IMPLAUSIBLE_YIELD_FACTOR = 10.0

def construct_case_narrative(row: pd.Series) -> str:
    """
    Constructs a rich natural language case narrative for vector embeddings and search.
    Example:
    'Rice grown in Punjab during Kharif 2019. Soil: N 90, P 42, K 43 (medium nitrogen), pH 6.5 (neutral).
    Weather: 26.5°C, 1200 mm rainfall (high), 80% humidity. Inputs: fertilizer 120 kg/ha, pesticide 2.1 kg/ha.
    Area 1500 ha, production 4200 t, yield 2.8 t/ha (top 30% for rice).'
    """
    crop = row.get("crop", "Unknown Crop")
    state = row.get("state", "Unknown Region")
    season = row.get("season", "Season")
    year = int(row.get("crop_year", 2020))
    
    n = row.get("n", 0)
    p = row.get("p", 0)
    k = row.get("k", 0)
    n_level = row.get("n_level", "medium")
    ph = row.get("ph", 7.0)
    ph_class = row.get("ph_class", "neutral")
    
    temp = row.get("temperature", 25.0)
    rain = row.get("rainfall", 800.0)
    rain_class = row.get("rainfall_class", "moderate")
    humidity = row.get("humidity", 70.0)
    
    fert = row.get("fertilizer", 100.0)
    pest = row.get("pesticide", 2.0)
    area = row.get("area", 1000.0)
    prod = row.get("production", 3000.0)
    yield_val = row.get("yield", 3.0)
    yield_pct = row.get("yield_percentile_within_crop", 50.0)
    
    # Calculate top percentile representation
    top_pct = round(100.0 - yield_pct, 0)
    top_str = f"top {int(max(1, top_pct))}%" if yield_pct >= 50 else f"lower {int(yield_pct)}%"
    
    narrative = (
        f"{crop} grown in {state} during {season} {year}. "
        f"Soil: N {n:.1f}, P {p:.1f}, K {k:.1f} ({n_level} nitrogen), pH {ph:.1f} ({ph_class}). "
        f"Weather: {temp:.1f}°C, {rain:.0f} mm rainfall ({rain_class}), {humidity:.0f}% humidity. "
        f"Inputs: fertilizer {fert:.1f} kg/ha, pesticide {pest:.2f} kg/ha. "
        f"Area {area:.1f} ha, production {prod:.1f}, yield {yield_val:.2f} {yield_unit(crop)} ({top_str} for {crop})."
    )
    return narrative

def clean_and_prepare_dataset(raw_csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw CSV data, imputes missing values, assigns stable case IDs,
    derives categorical features and persists thresholds and cleaned parquet.
    """
    logger.info(f"Reading raw dataset from: {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)
    initial_count = len(raw_df)
    logger.info(f"Initial raw row count: {initial_count}")

    # Standardize column names
    df = standardize_columns(raw_df)

    # Check required columns
    required = ["crop", "state", "yield"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column '{col}' in standardized dataset.")

    # Categorical normalization: strip, collapse inner whitespace and title-case
    for cat_col in ["crop", "season", "state"]:
        if cat_col in df.columns:
            df[cat_col] = (
                df[cat_col].where(df[cat_col].notna())
                .astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
            )

    # Drop rows missing crucial values, then exact duplicates
    df = df.dropna(subset=["crop", "state", "yield"])
    logger.info(f"Row count after dropping rows missing crop/state/yield: {len(df)}")
    df = df.drop_duplicates()
    logger.info(f"Row count after dropping duplicates: {len(df)}")

    # Numeric clipping & sanity bounds
    numeric_bounds = {
        "ph": (0.0, 14.0),
        "n": (0.0, 500.0),
        "p": (0.0, 500.0),
        "k": (0.0, 500.0),
        "temperature": (-10.0, 55.0),
        "rainfall": (0.0, 5000.0),
        "humidity": (0.0, 100.0),
        "fertilizer": (0.0, 2000.0),
        "pesticide": (0.0, 100.0),
        # Coconut yield is recorded in nuts/ha (up to ~21k); other crops in t/ha.
        "yield": (0.0, 25000.0),
        # Areas are state-level aggregates (up to ~5e7 ha) in the Kaggle data.
        "area": (0.1, 100000000.0),
        "production": (0.0, 1e10),
        "crop_year": (1950, 2030)
    }

    for col, (lower, upper) in numeric_bounds.items():
        if col in df.columns:
            # Impute missing values with crop-wise median or overall median
            if df[col].isnull().any():
                crop_medians = df.groupby("crop")[col].transform("median")
                df[col] = df[col].fillna(crop_medians).fillna(df[col].median())
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Drop implausible yields: the source "yield" column is occasionally inconsistent with
    # production/area (e.g. Maize at ~990 t/ha). Rows above 10x their crop's median are removed.
    crop_median = df.groupby("crop")["yield"].transform("median")
    implausible = (crop_median > 0) & (df["yield"] > IMPLAUSIBLE_YIELD_FACTOR * crop_median)
    if implausible.any():
        logger.info(f"Dropping {int(implausible.sum())} rows with yield > {IMPLAUSIBLE_YIELD_FACTOR}x crop median")
        df = df[~implausible]
    logger.info(f"Row count after implausible-yield filter: {len(df)}")

    # Ensure integer crop_year
    if "crop_year" in df.columns:
        df["crop_year"] = df["crop_year"].astype(int)
    else:
        df["crop_year"] = 2022

    # Compute thresholds and save
    thresholds = compute_and_save_thresholds(df, settings.THRESHOLDS_PATH)

    # Add derived features
    df = add_derived_features(df, thresholds)

    # Assign stable case IDs: CASE-000001, CASE-000002...
    df = df.reset_index(drop=True)
    df["case_id"] = [f"CASE-{i+1:06d}" for i in range(len(df))]

    # Generate narratives
    df["narrative"] = df.apply(construct_case_narrative, axis=1)

    # Save to parquet (and CSV fallback)
    out_parquet = Path(settings.PROCESSED_DATA_PATH)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    csv_fallback = out_parquet.with_suffix(".csv")
    df.to_csv(csv_fallback, index=False)

    write_dataset_source(raw_csv_path, len(df))
    refresh_known_values(df)

    logger.info(f"Cleaned dataset saved: {len(df)} rows -> {out_parquet}")
    return df, thresholds


def write_dataset_source(raw_csv_path: str, rows: int) -> None:
    """Records which raw file produced the processed dataset and whether it is synthetic."""
    info = {
        "raw_path": str(raw_csv_path),
        "is_synthetic": is_synthetic_path(raw_csv_path),
        "rows": rows,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(settings.DATASET_SOURCE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))


def get_dataset_info() -> Dict[str, Any]:
    """Returns provenance of the processed dataset (raw path, synthetic flag, row count)."""
    path = Path(settings.DATASET_SOURCE_PATH)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"Could not read dataset provenance file {path}: {e}")
    # Unknown provenance: assume synthetic unless the real raw CSV exists, so nothing is passed off as real.
    raw_exists = Path(settings.DATA_CSV_PATH).exists()
    return {"raw_path": settings.DATA_CSV_PATH if raw_exists else settings.SYNTHETIC_DATA_PATH,
            "is_synthetic": not raw_exists, "rows": None, "processed_at": None}


def is_synthetic_dataset() -> bool:
    """True when the active processed dataset was built from the synthetic fallback CSV."""
    return bool(get_dataset_info().get("is_synthetic", True))


def refresh_known_values(df: pd.DataFrame) -> None:
    """Updates the guardrail vocabularies (crops, seasons, states) from the active dataset."""
    from backend.app.guardrails.input_validation import set_known_values
    set_known_values(
        crops=sorted(df["crop"].dropna().astype(str).unique().tolist()),
        seasons=sorted(df["season"].dropna().astype(str).unique().tolist()) if "season" in df else [],
        states=sorted(df["state"].dropna().astype(str).unique().tolist()),
    )

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Converts DataFrame rows to LangChain Document objects with flat scalar metadata.
    """
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "case_id": str(row["case_id"]),
            "crop": str(row["crop"]),
            "season": str(row.get("season", "")),
            "state": str(row["state"]),
            "crop_year": int(row.get("crop_year", 2020)),
            "area": float(row.get("area", 0.0)),
            "production": float(row.get("production", 0.0)),
            "yield": float(row["yield"]),
            "fertilizer": float(row.get("fertilizer", 0.0)),
            "pesticide": float(row.get("pesticide", 0.0)),
            "n": float(row.get("n", 0.0)),
            "p": float(row.get("p", 0.0)),
            "k": float(row.get("k", 0.0)),
            "ph": float(row.get("ph", 7.0)),
            "temperature": float(row.get("temperature", 25.0)),
            "rainfall": float(row.get("rainfall", 500.0)),
            "humidity": float(row.get("humidity", 60.0)),
            "ph_class": str(row.get("ph_class", "neutral")),
            "n_level": str(row.get("n_level", "medium")),
            "p_level": str(row.get("p_level", "medium")),
            "k_level": str(row.get("k_level", "medium")),
            "temp_class": str(row.get("temp_class", "moderate")),
            "rainfall_class": str(row.get("rainfall_class", "moderate")),
            "humidity_class": str(row.get("humidity_class", "moderate")),
            "yield_percentile_within_crop": float(row.get("yield_percentile_within_crop", 50.0))
        }
        narrative = str(row.get("narrative") or construct_case_narrative(row))
        doc = Document(page_content=narrative, metadata=metadata)
        documents.append(doc)
    return documents

_ACTIVE_DF: Optional[pd.DataFrame] = None
_ACTIVE_DF_MTIME: Optional[float] = None


def get_active_dataframe() -> pd.DataFrame:
    """
    Loads the processed cases dataframe (cached in memory until the parquet file changes).
    If it is not present, resolves the raw dataset and runs the cleaning pipeline.
    """
    global _ACTIVE_DF, _ACTIVE_DF_MTIME
    processed_path = Path(settings.PROCESSED_DATA_PATH)
    if _ACTIVE_DF is None and processed_path.exists() and is_synthetic_dataset():
        from scripts.download_data import real_dataset_available
        if real_dataset_available():
            logger.warning("Real dataset found but the processed data is synthetic; re-processing from the real files.")
            processed_path.unlink()
    if processed_path.exists():
        mtime = processed_path.stat().st_mtime
        if _ACTIVE_DF is None or _ACTIVE_DF_MTIME != mtime:
            _ACTIVE_DF = pd.read_parquet(processed_path)
            _ACTIVE_DF_MTIME = mtime
            refresh_known_values(_ACTIVE_DF)
        return _ACTIVE_DF

    from scripts.download_data import ensure_dataset
    raw_path_str = ensure_dataset()
    if is_synthetic_path(raw_path_str):
        logger.warning("=" * 70)
        logger.warning("SYNTHETIC DATA IN USE: no real Kaggle dataset was found. Results are for demo only.")
        logger.warning("=" * 70)

    df, _ = clean_and_prepare_dataset(raw_path_str)
    _ACTIVE_DF = df
    _ACTIVE_DF_MTIME = processed_path.stat().st_mtime if processed_path.exists() else None
    return df


def is_synthetic_path(path: str) -> bool:
    return "SYNTHETIC" in Path(path).name.upper()
