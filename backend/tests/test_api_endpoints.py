import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "dataset_rows" in data
    assert "providers" in data

def test_providers_endpoint():
    res = client.get("/api/v1/providers")
    assert res.status_code == 200
    providers = res.json()
    assert len(providers) >= 2
    prov_names = [p["provider"] for p in providers]
    assert "ollama" in prov_names
    assert "openai" in prov_names

def test_metadata_options():
    res = client.get("/api/v1/metadata/options")
    assert res.status_code == 200
    data = res.json()
    assert "crops" in data
    assert "Rice" in data["crops"]

def test_classify_endpoint():
    res = client.post("/api/v1/classify", json={"query": "Which crops grow in Punjab in Kharif with N 90?"})
    assert res.status_code == 200
    clf = res.json()["classification"]
    assert clf["state"] == "Punjab"
    assert clf["season"] == "Kharif"

def test_search_endpoint():
    res = client.post("/api/v1/search", json={
        "query": "Rice cultivation in Punjab with high rainfall",
        "top_k": 4,
        "mode": "hybrid"
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) > 0

def test_recommend_endpoint():
    res = client.post("/api/v1/recommend", json={
        "query": "Best crops for Punjab during Kharif",
        "top_k": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["recommendations"]) > 0

def test_predict_yield_endpoint():
    payload = {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "n": 95, "p": 45, "k": 40, "ph": 6.5,
        "temperature": 27, "rainfall": 1200, "humidity": 80
    }
    res = client.post("/api/v1/predict/yield", json=payload)
    assert res.status_code == 200
    pred = res.json()["prediction"]
    assert pred["value"] > 0

def test_predict_risk_endpoint():
    payload = {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "n": 95, "p": 45, "k": 40, "ph": 6.5,
        "temperature": 27, "rainfall": 1200, "humidity": 80
    }
    res = client.post("/api/v1/predict/risk", json=payload)
    assert res.status_code == 200
    risk = res.json()["risk"]
    assert risk["level"] in ["low", "medium", "high"]

def test_analyze_impact_endpoint():
    payload = {
        "crop": "Rice",
        "season": "Kharif",
        "state": "Punjab",
        "n": 95, "p": 45, "k": 40, "ph": 6.5,
        "temperature": 27, "rainfall": 1200, "humidity": 80
    }
    res = client.post("/api/v1/analyze/impact", json=payload)
    assert res.status_code == 200
    assert len(res.json()["what_if_sensitivities"]) > 0

def test_compare_endpoint():
    payload = {
        "profiles": [
            {"name": "Farm 1", "crop": "Wheat", "season": "Rabi", "state": "Punjab", "n": 100, "p": 50, "k": 40, "ph": 6.8, "temperature": 18, "rainfall": 350, "humidity": 60, "actual_yield": 4.0},
            {"name": "Farm 2", "crop": "Wheat", "season": "Rabi", "state": "Punjab", "n": 50, "p": 30, "k": 20, "ph": 6.8, "temperature": 18, "rainfall": 200, "humidity": 60, "actual_yield": 2.5}
        ]
    }
    res = client.post("/api/v1/compare", json=payload)
    assert res.status_code == 200
    assert len(res.json()["ranked_factors"]) > 0

def test_feedback_flow():
    post_res = client.post("/api/v1/feedback", json={
        "query": "Test query",
        "case_ids": ["CASE-000001"],
        "recommended_crop": "Rice",
        "rating": 1,
        "comment": "Accurate recommendation",
        "helpful_case_ids": ["CASE-000001"]
    })
    assert post_res.status_code == 200

    stats_res = client.get("/api/v1/feedback/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["total_feedback"] >= 1

def test_eval_endpoints():
    run_res = client.post("/api/v1/eval/run", json={"limit": 3})
    assert run_res.status_code == 200
    run_id = run_res.json()["run_id"]

    status_res = client.get(f"/api/v1/eval/status/{run_id}")
    assert status_res.status_code == 200

def test_graph_mermaid_endpoint():
    res = client.get("/api/v1/graph")
    assert res.status_code == 200
    assert "mermaid" in res.json()

def test_query_pipeline_endpoint():
    res = client.post("/api/v1/query", json={
        "query": "Which crops do well in Punjab during Kharif with N around 90, P 40, K 40, pH 6.5 and about 1200 mm rainfall?",
        "top_k": 5,
        "mode": "hybrid"
    })
    assert res.status_code == 200
    data = res.json()
    assert "classification" in data
    assert "similar_cases" in data
    assert "recommendations" in data
    assert "agent_trace" in data
