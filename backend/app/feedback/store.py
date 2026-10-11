import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.app.config import settings
from backend.app.schemas.feedback import FeedbackCreate, FeedbackStats

Base = declarative_base()

class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    case_ids = Column(Text, nullable=False)  # JSON array
    recommended_crop = Column(String(100), nullable=True)
    rating = Column(Integer, nullable=False)  # +1, -1, 0
    comment = Column(Text, nullable=True)
    helpful_case_ids = Column(Text, nullable=False)  # JSON array
    created_at = Column(DateTime, default=datetime.utcnow)

def get_engine():
    db_path = settings.FEEDBACK_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

# Initialize table on import
init_db()

def save_feedback(feedback: FeedbackCreate) -> int:
    """Saves farmer/agronomist feedback record into SQLite database."""
    session: Session = SessionLocal()
    try:
        rec = FeedbackRecord(
            query=feedback.query,
            case_ids=json.dumps(feedback.case_ids),
            recommended_crop=feedback.recommended_crop,
            rating=feedback.rating,
            comment=feedback.comment,
            helpful_case_ids=json.dumps(feedback.helpful_case_ids),
            created_at=datetime.now(timezone.utc)
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id
    finally:
        session.close()

def get_feedback_statistics() -> FeedbackStats:
    """Calculates summary statistics of feedback collected."""
    session: Session = SessionLocal()
    try:
        records: List[FeedbackRecord] = session.query(FeedbackRecord).all()
        total = len(records)
        pos = sum(1 for r in records if r.rating > 0)
        neg = sum(1 for r in records if r.rating < 0)
        neu = total - (pos + neg)

        helpful_counts: Dict[str, int] = {}
        crop_stats: Dict[str, Dict[str, int]] = {}

        for r in records:
            # Parse helpful cases
            try:
                cids = json.loads(r.helpful_case_ids) if r.helpful_case_ids else []
                for cid in cids:
                    helpful_counts[cid] = helpful_counts.get(cid, 0) + 1
            except Exception:
                pass

            # Crop ratings
            if r.recommended_crop:
                c = r.recommended_crop
                if c not in crop_stats:
                    crop_stats[c] = {"positive": 0, "negative": 0}
                if r.rating > 0:
                    crop_stats[c]["positive"] += 1
                elif r.rating < 0:
                    crop_stats[c]["negative"] += 1

        top_helpful = [
            {"case_id": cid, "helpful_count": count}
            for cid, count in sorted(helpful_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        return FeedbackStats(
            total_feedback=total,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            top_helpful_cases=top_helpful,
            crop_ratings=crop_stats
        )
    finally:
        session.close()
