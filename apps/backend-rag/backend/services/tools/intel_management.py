"""
Intel Management Tool
Allows the Agent to manage Intelligence Center configuration (keywords, sources).
"""

import logging
from typing import ClassVar, Type

from pydantic import BaseModel, Field

from app.core.config import settings
from services.tools.definitions import BaseTool

logger = logging.getLogger(__name__)

class AddKeywordInput(BaseModel):
    term: str = Field(..., description="The keyword term to add (e.g., 'digital nomad visa')")
    category: str = Field(..., description="Category (business, immigration, tax, property, tech, lifestyle)")
    level: str = Field(default="medium", description="Importance level: direct (100), high (90), medium (70)")

class IntelManagementTool(BaseTool):
    """
    Tool for managing Intelligence Center configuration.
    Allows adding new keywords to the scoring system.
    """
    name: ClassVar[str] = "manage_intelligence"
    description: ClassVar[str] = "Add or update intelligence keywords and scoring rules. Use this when the user wants to track new topics."
    args_schema: ClassVar[Type[BaseModel]] = AddKeywordInput

    async def execute(self, term: str, category: str, level: str = "medium", **kwargs) -> str:
        try:
            import asyncpg
            
            if not settings.database_url:
                return "Error: Database URL not configured."

            conn = await asyncpg.connect(settings.database_url)
            try:
                # Check if exists
                exists = await conn.fetchval(
                    "SELECT id FROM intel_keywords WHERE term = $1 AND category = $2",
                    term, category
                )
                
                if exists:
                    await conn.execute(
                        "UPDATE intel_keywords SET level = $3, is_active = true, updated_at = NOW() WHERE term = $1 AND category = $2",
                        term, category, level
                    )
                    return f"Updated existing keyword '{term}' in category '{category}' to level '{level}'."
                else:
                    await conn.execute(
                        "INSERT INTO intel_keywords (term, category, level, is_active, created_at, updated_at) VALUES ($1, $2, $3, true, NOW(), NOW())",
                        term, category, level
                    )
                    return f"Successfully added new keyword '{term}' to category '{category}' with level '{level}'."
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"IntelManagementTool error: {e}")
            return f"Error managing intelligence: {str(e)}"
