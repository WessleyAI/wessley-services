"""Core utilities for the knowledge scraper service."""

from .config import settings
from .schemas import (
    ScraperJob,
    ScraperJobStatus,
    KnowledgeEntry,
    KnowledgeType,
    VehicleSignature,
    ComponentLocation,
    ComponentConnection,
    ScrapedPost,
)

__all__ = [
    "settings",
    "ScraperJob",
    "ScraperJobStatus",
    "KnowledgeEntry",
    "KnowledgeType",
    "VehicleSignature",
    "ComponentLocation",
    "ComponentConnection",
    "ScrapedPost",
]
