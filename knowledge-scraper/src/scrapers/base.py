"""Base scraper interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.core.schemas import ScrapedPost, SourceType


class BaseScraper(ABC):
    """Base class for all scrapers."""

    source_type: SourceType

    @abstractmethod
    async def scrape(
        self,
        query: str | None = None,
        limit: int = 1000,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape posts from the source.

        Args:
            query: Optional search query
            limit: Maximum number of posts to scrape

        Yields:
            ScrapedPost objects
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass
