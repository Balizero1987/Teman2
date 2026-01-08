from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

# --- Keywords ---

class IntelKeywordBase(BaseModel):
    term: str = Field(..., min_length=2, max_length=255)
    category: str = Field(..., max_length=100)
    level: str = Field(default="medium", pattern="^(direct|high|medium)$")
    score_override: Optional[int] = Field(default=None, ge=0, le=100)
    is_active: bool = Field(default=True)

class IntelKeywordCreate(IntelKeywordBase):
    pass

class IntelKeywordUpdate(BaseModel):
    term: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None
    score_override: Optional[int] = None
    is_active: Optional[bool] = None

class IntelKeywordResponse(IntelKeywordBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Source Authority ---

class IntelSourceAuthorityBase(BaseModel):
    domain: str = Field(..., min_length=3, max_length=255)
    name: str = Field(..., max_length=255)
    score: int = Field(default=50, ge=0, le=100)
    category: str = Field(default="general", max_length=100)
    is_active: bool = Field(default=True)

class IntelSourceAuthorityCreate(IntelSourceAuthorityBase):
    pass

class IntelSourceAuthorityUpdate(BaseModel):
    domain: Optional[str] = None
    name: Optional[str] = None
    score: Optional[int] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None

class IntelSourceAuthorityResponse(IntelSourceAuthorityBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
