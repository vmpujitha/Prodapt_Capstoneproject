import pytest
from backend.app.ml.yield_model import get_yield_predictor
from backend.app.ml.risk_model import get_risk_predictor
from backend.app.ml.impact import compute_global_importance, compute_what_if_sensitivity, compare_farm_profiles
from backend.app.schemas.analysis import FarmProfile, CompareRequest

def test_yield_model_prediction_shape():
    predictor = get_yield_predictor()
    data = {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "n": 95.0,
        "p": 45.0,
        "k": 42.0,
        "ph": 6.5,
        "temperature": 27.0,
        "rainfall": 1300.0,
        "humidity": 80.0,
        "fertilizer": 130.0,
        "pesticide": 2.2,
        "area": 1000.0,
        "crop_year": 2023
    }
    pred = predictor.predict_one(data)
    assert pred.value > 0.5
    assert pred.low <= pred.value <= pred.high

def test_risk_model_assessment():
    risk_pred = get_risk_predictor()
    data = {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "n": 95.0,
        "p": 45.0,
        "k": 42.0,
        "ph": 6.5,
        "temperature": 27.0,
        "rainfall": 1300.0,
        "humidity": 80.0
    }
    risk = risk_pred.assess_risk(data)
    assert risk.level in ["low", "medium", "high"]
    assert 0.0 <= risk.probability <= 1.0

def test_impact_analysis_and_compare():
    data = {
        "crop": "Wheat",
        "season": "Rabi",
        "state": "Punjab",
        "n": 110.0,
        "p": 55.0,
        "k": 45.0,
        "ph": 6.8,
        "temperature": 18.5,
        "rainfall": 380.0,
        "humidity": 60.0,
        "fertilizer": 140.0,
        "pesticide": 1.8
    }
    importances = compute_global_importance("Wheat")
    assert len(importances) > 0

    sensitivities = compute_what_if_sensitivity(data)
    assert len(sensitivities) > 0

    p1 = FarmProfile(name="Farm A", crop="Wheat", season="Rabi", state="Punjab", n=110, p=55, k=45, ph=6.8, temperature=18.5, rainfall=380, humidity=60, actual_yield=4.2)
    p2 = FarmProfile(name="Farm B", crop="Wheat", season="Rabi", state="Punjab", n=60, p=30, k=25, ph=6.8, temperature=18.5, rainfall=200, humidity=60, actual_yield=2.8)
    req = CompareRequest(profiles=[p1, p2])
    comp = compare_farm_profiles(req)
    assert comp.yield_gap < 0
    assert len(comp.ranked_factors) > 0
