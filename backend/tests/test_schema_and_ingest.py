import pytest
import pandas as pd
from backend.app.data.schema import standardize_columns, CANONICAL_COLUMNS
from backend.app.data.features import classify_ph, classify_value_by_tertile, add_derived_features, compute_and_save_thresholds
from backend.app.data.ingest import construct_case_narrative, df_to_documents

def test_standardize_columns():
    df = pd.DataFrame({
        "Crop_Name": ["Rice"],
        "Year": [2021],
        "Cropping_Season": ["Kharif"],
        "State_Name": ["Punjab"],
        "Area_ha": [1200.0],
        "Crop_Yield": [3.4],
        "Nitrogen": [90.0],
        "Phosphorus": [45.0],
        "Potassium": [40.0],
        "Soil_pH": [6.5],
        "Temp": [26.0],
        "Annual_Rainfall": [1100.0],
        "Relative_Humidity": [75.0]
    })
    std_df = standardize_columns(df)
    assert "crop" in std_df.columns
    assert "crop_year" in std_df.columns
    assert "season" in std_df.columns
    assert "yield" in std_df.columns
    assert "n" in std_df.columns
    assert "ph" in std_df.columns

def test_derived_features():
    assert classify_ph(5.5) == "acidic"
    assert classify_ph(6.8) == "neutral"
    assert classify_ph(8.2) == "alkaline"
    assert classify_value_by_tertile(30.0, 50.0, 100.0) == "low"
    assert classify_value_by_tertile(75.0, 50.0, 100.0) == "medium"
    assert classify_value_by_tertile(150.0, 50.0, 100.0) == "high"

def test_narrative_and_document_construction():
    row = pd.Series({
        "crop": "Wheat",
        "crop_year": 2022,
        "season": "Rabi",
        "state": "Haryana",
        "area": 500.0,
        "production": 2000.0,
        "yield": 4.0,
        "fertilizer": 120.0,
        "pesticide": 1.5,
        "n": 100.0,
        "p": 50.0,
        "k": 40.0,
        "ph": 6.8,
        "temperature": 18.0,
        "rainfall": 350.0,
        "humidity": 60.0,
        "ph_class": "neutral",
        "n_level": "medium",
        "yield_percentile_within_crop": 80.0
    })
    narrative = construct_case_narrative(row)
    assert "Wheat grown in Haryana" in narrative
    assert "Rabi 2022" in narrative
    assert "yield 4.00 t/ha" in narrative

    df = pd.DataFrame([row])
    df["case_id"] = "CASE-TEST01"
    docs = df_to_documents(df)
    assert len(docs) == 1
    assert docs[0].metadata["case_id"] == "CASE-TEST01"
    assert docs[0].metadata["crop"] == "Wheat"
