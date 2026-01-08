"""
Intel Module Models
Manages dynamic keywords and source authority for the Intelligence Center.
"""

from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, String, Integer, Float, Boolean, JSON

class IntelKeyword(SQLModel, table=True):
    """
    Dynamic keyword for scoring articles.
    Replaces the hardcoded KEYWORDS dict in professional_scorer.py.
    """
    __tablename__ = "intel_keywords"

    id: Optional[int] = Field(default=None, primary_key=True)
    term: str = Field(index=True, max_length=255)
    category: str = Field(index=True, max_length=100)  # e.g., "business", "immigration"
    level: str = Field(default="medium", max_length=50) # "direct", "high", "medium"
    score_override: Optional[int] = Field(default=None) # If set, overrides standard level score
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class IntelSourceAuthority(SQLModel, table=True):
    """
    Source authority score.
    Replaces the hardcoded SOURCE_AUTHORITY dict.
    """
    __tablename__ = "intel_source_authority"

    id: Optional[int] = Field(default=None, primary_key=True)
    domain: str = Field(unique=True, index=True, max_length=255) # e.g., "reuters.com" or "reuters"
    name: str = Field(max_length=255)
    score: int = Field(default=50)
    category: str = Field(default="general", max_length=100) # "government", "major_media", etc.
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
