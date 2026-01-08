"""
Intel Module Router
API endpoints for managing Dynamic Intelligence (Keywords & Authority).
"""

import logging
from typing import List

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from app.dependencies import get_current_user, get_database_pool
from app.modules.intel.schemas import (
    IntelKeywordCreate,
    IntelKeywordResponse,
    IntelKeywordUpdate,
    IntelSourceAuthorityCreate,
    IntelSourceAuthorityResponse,
    IntelSourceAuthorityUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intel", tags=["intel"])

# ============================================================================
# KEYWORDS ENDPOINTS
# ============================================================================

@router.get("/keywords", response_model=List[IntelKeywordResponse])
async def list_keywords(
    category: str = Query(None, description="Filter by category"),
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """List all intelligence keywords"""
    query = "SELECT * FROM intel_keywords WHERE is_active = true"
    args = []
    
    if category:
        query += " AND category = $1"
        args.append(category)
    
    query += " ORDER BY category, term"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *args)
        return [dict(row) for row in rows]

@router.post("/keywords", response_model=IntelKeywordResponse)
async def create_keyword(
    keyword: IntelKeywordCreate,
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """Create a new intelligence keyword"""
    # Check if exists
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT id FROM intel_keywords WHERE term = $1 AND category = $2",
            keyword.term, keyword.category
        )
        if exists:
            raise HTTPException(status_code=400, detail="Keyword already exists in this category")

        query = """
            INSERT INTO intel_keywords (term, category, level, score_override, is_active, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
            RETURNING *
        """
        row = await conn.fetchrow(
            query,
            keyword.term,
            keyword.category,
            keyword.level,
            keyword.score_override,
            keyword.is_active
        )
        logger.info(
            f"Intel Config: Admin {current_user.get('email')} CREATED keyword '{keyword.term}' (Category: {keyword.category})"
        )
        return dict(row)

@router.put("/keywords/{keyword_id}", response_model=IntelKeywordResponse)
async def update_keyword(
    keyword_id: int,
    keyword_update: IntelKeywordUpdate,
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """Update a keyword"""
    async with pool.acquire() as conn:
        # Check existence
        existing = await conn.fetchrow("SELECT * FROM intel_keywords WHERE id = $1", keyword_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Keyword not found")

        # Build dynamic update query
        fields = []
        values = []
        idx = 1
        
        update_data = keyword_update.model_dump(exclude_unset=True)
        if not update_data:
            return dict(existing)

        for k, v in update_data.items():
            fields.append(f"{k} = ${idx}")
            values.append(v)
            idx += 1
        
        fields.append(f"updated_at = NOW()")
        
        query = f"""
            UPDATE intel_keywords
            SET {", ".join(fields)}
            WHERE id = ${idx}
            RETURNING *
        """
        values.append(keyword_id)
        
        row = await conn.fetchrow(query, *values)
        logger.info(
            f"Intel Config: Admin {current_user.get('email')} UPDATED keyword ID {keyword_id}"
        )
        return dict(row)

@router.delete("/keywords/{keyword_id}")
async def delete_keyword(
    keyword_id: int,
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """Delete (soft delete) a keyword"""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE intel_keywords SET is_active = false, updated_at = NOW() WHERE id = $1",
            keyword_id
        )
        if result == "DELETE 0": # Update returns "UPDATE count", usually
            # asyncpg execute returns string like "UPDATE 1"
            pass
        
        # Check if actually updated
        if "0" in result:
             raise HTTPException(status_code=404, detail="Keyword not found")
        
        logger.info(
            f"Intel Config: Admin {current_user.get('email')} DELETED keyword ID {keyword_id}"
        )
        return {"success": True, "message": "Keyword deactivated"}

# ============================================================================
# AUTHORITY ENDPOINTS
# ============================================================================

@router.get("/authority", response_model=List[IntelSourceAuthorityResponse])
async def list_authority(
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """List all source authority rules"""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM intel_source_authority WHERE is_active = true ORDER BY score DESC")
        return [dict(row) for row in rows]

@router.post("/authority", response_model=IntelSourceAuthorityResponse)
async def create_authority(
    authority: IntelSourceAuthorityCreate,
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """Create a new source authority rule"""
    async with pool.acquire() as conn:
        try:
            query = """
                INSERT INTO intel_source_authority (domain, name, score, category, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                RETURNING *
            """
            row = await conn.fetchrow(
                query,
                authority.domain,
                authority.name,
                authority.score,
                authority.category,
                authority.is_active
            )
            return dict(row)
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=400, detail="Domain already exists")

@router.delete("/authority/{auth_id}")
async def delete_authority(
    auth_id: int,
    pool: asyncpg.Pool = Depends(get_database_pool),
    current_user: dict = Depends(get_current_user),
):
    """Delete (soft delete) an authority rule"""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE intel_source_authority SET is_active = false, updated_at = NOW() WHERE id = $1",
            auth_id
        )
        return {"success": True}
