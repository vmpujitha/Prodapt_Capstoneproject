from typing import Dict, Any, List, Optional
from backend.app.schemas.search import MetadataFilters

def build_chroma_filter(filters: Optional[MetadataFilters]) -> Optional[Dict[str, Any]]:
    """
    Constructs a valid ChromaDB `$and` filter clause from MetadataFilters.
    Handles exact matching, list containment, and range queries ($gte, $lte).
    """
    if not filters:
        return None

    clauses: List[Dict[str, Any]] = []

    # Categorical filters
    if filters.crop:
        clauses.append({"crop": {"$eq": filters.crop.strip().title()}})
    if filters.season:
        clauses.append({"season": {"$eq": filters.season.strip().title()}})
    if filters.state:
        clauses.append({"state": {"$eq": filters.state.strip().title()}})

    # Classes
    if filters.ph_class:
        clauses.append({"ph_class": {"$eq": filters.ph_class.strip().lower()}})
    if filters.n_level:
        clauses.append({"n_level": {"$eq": filters.n_level.strip().lower()}})
    if filters.p_level:
        clauses.append({"p_level": {"$eq": filters.p_level.strip().lower()}})
    if filters.k_level:
        clauses.append({"k_level": {"$eq": filters.k_level.strip().lower()}})
    if filters.temp_class:
        clauses.append({"temp_class": {"$eq": filters.temp_class.strip().lower()}})
    if filters.rainfall_class:
        clauses.append({"rainfall_class": {"$eq": filters.rainfall_class.strip().lower()}})
    if filters.humidity_class:
        clauses.append({"humidity_class": {"$eq": filters.humidity_class.strip().lower()}})

    # Numeric ranges
    if filters.crop_year_min is not None and filters.crop_year_max is not None:
        clauses.append({"crop_year": {"$gte": int(filters.crop_year_min)}})
        clauses.append({"crop_year": {"$lte": int(filters.crop_year_max)}})
    elif filters.crop_year_min is not None:
        clauses.append({"crop_year": {"$gte": int(filters.crop_year_min)}})
    elif filters.crop_year_max is not None:
        clauses.append({"crop_year": {"$lte": int(filters.crop_year_max)}})

    if filters.ph_min is not None and filters.ph_max is not None:
        clauses.append({"ph": {"$gte": float(filters.ph_min)}})
        clauses.append({"ph": {"$lte": float(filters.ph_max)}})
    elif filters.ph_min is not None:
        clauses.append({"ph": {"$gte": float(filters.ph_min)}})
    elif filters.ph_max is not None:
        clauses.append({"ph": {"$lte": float(filters.ph_max)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def matches_python_filters(metadata: Dict[str, Any], filters: Optional[MetadataFilters]) -> bool:
    """
    Evaluates whether an in-memory document metadata dict matches MetadataFilters.
    Used for pre-filtering BM25 and fallback retrieval.
    """
    if not filters:
        return True

    if filters.crop and metadata.get("crop", "").lower() != filters.crop.lower():
        return False
    if filters.season and metadata.get("season", "").lower() != filters.season.lower():
        return False
    if filters.state and metadata.get("state", "").lower() != filters.state.lower():
        return False

    if filters.ph_class and metadata.get("ph_class", "").lower() != filters.ph_class.lower():
        return False
    if filters.n_level and metadata.get("n_level", "").lower() != filters.n_level.lower():
        return False
    if filters.p_level and metadata.get("p_level", "").lower() != filters.p_level.lower():
        return False
    if filters.k_level and metadata.get("k_level", "").lower() != filters.k_level.lower():
        return False
    if filters.temp_class and metadata.get("temp_class", "").lower() != filters.temp_class.lower():
        return False
    if filters.rainfall_class and metadata.get("rainfall_class", "").lower() != filters.rainfall_class.lower():
        return False
    if filters.humidity_class and metadata.get("humidity_class", "").lower() != filters.humidity_class.lower():
        return False

    year = metadata.get("crop_year")
    if year is not None:
        if filters.crop_year_min is not None and year < filters.crop_year_min:
            return False
        if filters.crop_year_max is not None and year > filters.crop_year_max:
            return False

    ph = metadata.get("ph")
    if ph is not None:
        if filters.ph_min is not None and ph < filters.ph_min:
            return False
        if filters.ph_max is not None and ph > filters.ph_max:
            return False

    return True
