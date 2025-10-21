"""
API package initialization.
"""

from .search_routes import router as search_router

__all__ = ["search_router"]