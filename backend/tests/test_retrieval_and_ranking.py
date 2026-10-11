import pytest
from langchain_core.documents import Document
from backend.app.schemas.search import MetadataFilters
from backend.app.retrieval.filters import build_chroma_filter, matches_python_filters
from backend.app.retrieval.hybrid import reciprocal_rank_fusion
from backend.app.retrieval.ranking import compute_condition_proximity, compute_recency_score, prioritize_cases

def test_filter_translation():
    filters = MetadataFilters(
        crop="Rice",
        season="Kharif",
        crop_year_min=2015,
        crop_year_max=2020,
        ph_min=6.0,
        ph_max=7.5
    )
    chroma_where = build_chroma_filter(filters)
    assert "$and" in chroma_where
    assert {"crop": {"$eq": "Rice"}} in chroma_where["$and"]

    meta = {"crop": "Rice", "season": "Kharif", "crop_year": 2018, "ph": 6.5}
    assert matches_python_filters(meta, filters) is True

    bad_meta = {"crop": "Wheat", "season": "Kharif", "crop_year": 2018, "ph": 6.5}
    assert matches_python_filters(bad_meta, filters) is False

def test_rrf_fusion():
    d1 = Document(page_content="Case 1", metadata={"case_id": "CASE-1"})
    d2 = Document(page_content="Case 2", metadata={"case_id": "CASE-2"})
    d3 = Document(page_content="Case 3", metadata={"case_id": "CASE-3"})

    vec_results = [(d1, 0.9), (d2, 0.7)]
    bm25_results = [(d2, 0.95), (d3, 0.6)]

    merged = reciprocal_rank_fusion(vec_results, bm25_results, alpha=0.5)
    assert len(merged) == 3
    # d2 is present in both top positions so should score high
    case_ids = [doc.metadata["case_id"] for doc, _ in merged]
    assert "CASE-2" in case_ids

def test_ranking_math():
    doc = Document(
        page_content="Test case narrative",
        metadata={
            "case_id": "CASE-0001",
            "n": 90.0,
            "p": 40.0,
            "k": 40.0,
            "ph": 6.5,
            "temperature": 26.0,
            "rainfall": 1200.0,
            "humidity": 80.0,
            "crop_year": 2022,
            "yield_percentile_within_crop": 85.0
        }
    )
    user_cond = {"n": 90.0, "p": 40.0, "k": 40.0, "ph": 6.5}
    prox = compute_condition_proximity(doc.metadata, user_cond)
    assert prox > 0.90

    recency = compute_recency_score(2022)
    assert 0.8 <= recency <= 1.0

    ranked = prioritize_cases([(doc, 0.85)], user_conditions=user_cond)
    assert len(ranked) == 1
    assert ranked[0].scores.final > 0.7
    assert ranked[0].scores.proximity > 0.9


def test_season_from_months():
    from backend.app.agents.classifier import season_from_months
    assert season_from_months("which crops yeild more in telangana during the moths of jun-september") == "Kharif"
    assert season_from_months("from June to Sept") == "Kharif"
    assert season_from_months("sowing in july") == "Kharif"
    assert season_from_months("crops for november to february") == "Rabi"
    assert season_from_months("december-march") == "Rabi"
    assert season_from_months("what may grow well here") is None   # 'may' the verb
    assert season_from_months("jan to dec") is None                # spans seasons
    assert season_from_months("best crop for Punjab") is None
