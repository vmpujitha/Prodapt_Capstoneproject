import json
import logging
from typing import Dict, Optional
from sqlalchemy.orm import Session

from backend.app.feedback.store import SessionLocal, FeedbackRecord

logger = logging.getLogger(__name__)

def crop_feedback_key(crop: Optional[str]) -> str:
    """Key under which a crop-level boost is stored in the boost map."""
    return f"crop::{crop or ''}"


def _laplace_boost(pos: int, neg: int) -> float:
    """(pos - neg) / (pos + neg + 2) scaled by 0.1, i.e. bounded to [-0.1, +0.1]."""
    return max(-0.1, min(0.1, round((pos - neg) / (pos + neg + 2.0) * 0.1, 4)))


def get_feedback_boost_map() -> Dict[str, float]:
    """
    Laplace-smoothed feedback boosts, keyed by case_id and by crop (crop_feedback_key).
    Case-level counts come from the rating of every answer that cited the case, plus
    explicit "helpful" ticks; crop-level counts come from ratings of answers recommending
    that crop. Each boost is bounded to [-0.1, +0.1]. This is a re-ranking signal only;
    no model is retrained from feedback.
    """
    session: Session = SessionLocal()
    case_pos: Dict[str, int] = {}
    case_neg: Dict[str, int] = {}
    crop_pos: Dict[str, int] = {}
    crop_neg: Dict[str, int] = {}

    try:
        records = session.query(FeedbackRecord).all()
        for r in records:
            # Check helpful cases
            try:
                helpful = json.loads(r.helpful_case_ids) if r.helpful_case_ids else []
                for cid in helpful:
                    case_pos[cid] = case_pos.get(cid, 0) + 1
            except Exception:
                pass

            # Check general rating for cases in session
            try:
                cids = json.loads(r.case_ids) if r.case_ids else []
                for cid in cids:
                    if r.rating > 0:
                        case_pos[cid] = case_pos.get(cid, 0) + 1
                    elif r.rating < 0:
                        case_neg[cid] = case_neg.get(cid, 0) + 1
            except Exception:
                pass

            if r.recommended_crop:
                bucket = crop_pos if r.rating > 0 else crop_neg if r.rating < 0 else None
                if bucket is not None:
                    bucket[r.recommended_crop] = bucket.get(r.recommended_crop, 0) + 1

        boost_map: Dict[str, float] = {}
        for cid in set(case_pos) | set(case_neg):
            boost_map[cid] = _laplace_boost(case_pos.get(cid, 0), case_neg.get(cid, 0))
        for crop in set(crop_pos) | set(crop_neg):
            boost_map[crop_feedback_key(crop)] = _laplace_boost(crop_pos.get(crop, 0), crop_neg.get(crop, 0))
        return boost_map
    except Exception as e:
        logger.warning(f"Error computing feedback boost map: {e}")
        return {}
    finally:
        session.close()
