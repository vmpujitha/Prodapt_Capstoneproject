from typing import Dict, Any, List, Optional
import pandas as pd

# Canonical feature names used throughout the entire application
CANONICAL_COLUMNS = [
    "crop",
    "crop_year",
    "season",
    "state",
    "area",
    "production",
    "yield",
    "fertilizer",
    "pesticide",
    "n",
    "p",
    "k",
    "ph",
    "temperature",
    "rainfall",
    "humidity"
]

# Alias map for standardizing column names from varying CSV sources
COLUMN_ALIASES: Dict[str, str] = {
    # Crop
    "crop": "crop",
    "crop_name": "crop",
    "commodity": "crop",
    # Crop Year
    "crop_year": "crop_year",
    "year": "crop_year",
    "cropyear": "crop_year",
    # Season
    "season": "season",
    "cropping_season": "season",
    # State
    "state": "state",
    "state_name": "state",
    "region": "state",
    # Area
    "area": "area",
    "area_ha": "area",
    "cultivated_area": "area",
    # Production
    "production": "production",
    "production_tonnes": "production",
    "total_production": "production",
    # Yield
    "yield": "yield",
    "crop_yield": "yield",
    "yield_ton_per_ha": "yield",
    "yield_t_ha": "yield",
    # Fertilizer
    "fertilizer": "fertilizer",
    "fertilizer_kg_ha": "fertilizer",
    "fertilizer_usage": "fertilizer",
    # Pesticide
    "pesticide": "pesticide",
    "pesticide_kg_ha": "pesticide",
    "pesticide_usage": "pesticide",
    # N, P, K
    "n": "n",
    "nitrogen": "n",
    "n_kg_ha": "n",
    "p": "p",
    "phosphorus": "p",
    "p_kg_ha": "p",
    "k": "k",
    "potassium": "k",
    "k_kg_ha": "k",
    # pH
    "ph": "ph",
    "soil_ph": "ph",
    # Temperature
    "temperature": "temperature",
    "temp": "temperature",
    "avg_temp": "temperature",
    "temperature_c": "temperature",
    "avg_temp_c": "temperature",
    # Rainfall
    "rainfall": "rainfall",
    "annual_rainfall": "rainfall",
    "rainfall_mm": "rainfall",
    "total_rainfall_mm": "rainfall",
    # Humidity
    "humidity": "humidity",
    "relative_humidity": "humidity",
    "humidity_pct": "humidity",
    "avg_humidity_percent": "humidity"
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames DataFrame columns using the canonical alias dictionary.
    Columns are normalized to lowercase before matching.
    """
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower().replace(" ", "_").replace("-", "_")
        if norm in COLUMN_ALIASES:
            col_map[col] = COLUMN_ALIASES[norm]
        else:
            col_map[col] = norm
    
    df_renamed = df.rename(columns=col_map)
    return df_renamed

# Yield is recorded in t/ha for every crop except Coconut, which the source data records in nuts/ha.
YIELD_UNITS: Dict[str, str] = {"Coconut": "nuts/ha"}


def yield_unit(crop: Optional[str]) -> str:
    """Returns the unit in which yield is recorded for `crop`."""
    return YIELD_UNITS.get(str(crop or "").strip(), "t/ha")
