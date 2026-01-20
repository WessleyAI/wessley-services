"""Generic forum scraper using BeautifulSoup."""

import asyncio
import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import ScrapedPost, SourceType, VehicleSignature

from .base import BaseScraper

logger = get_logger(__name__)


class ForumConfig:
    """Configuration for a specific forum."""

    def __init__(
        self,
        name: str,
        base_url: str,
        search_path: str = "/search",
        thread_selector: str = "div.thread",
        title_selector: str = "h3.title a",
        content_selector: str = "div.post-content",
        author_selector: str = "span.author",
        date_selector: str = "span.date",
        next_page_selector: str = "a.next",
    ):
        self.name = name
        self.base_url = base_url
        self.search_path = search_path
        self.thread_selector = thread_selector
        self.title_selector = title_selector
        self.content_selector = content_selector
        self.author_selector = author_selector
        self.date_selector = date_selector
        self.next_page_selector = next_page_selector


FORUM_CONFIGS = {
    "bimmerpost": ForumConfig(
        name="BimmerPost",
        base_url="https://www.bimmerpost.com",
        search_path="/forums/search.php",
        thread_selector="li.searchResult",
        title_selector="h3.title a",
        content_selector="div.snippet",
    ),
    "toyotanation": ForumConfig(
        name="ToyotaNation",
        base_url="https://www.toyotanation.com",
        search_path="/threads/search",
        thread_selector="div.structItem",
        title_selector="div.structItem-title a",
        content_selector="div.structItem-snippet",
    ),
    "hondatech": ForumConfig(
        name="Honda-Tech",
        base_url="https://honda-tech.com",
        search_path="/forums/search/",
        thread_selector="li.searchResult",
        title_selector="h3.title a",
        content_selector="div.snippet",
    ),
    "jeepforum": ForumConfig(
        name="JeepForum",
        base_url="https://www.jeepforum.com",
        search_path="/forum/search.php",
        thread_selector="li.searchResult",
        title_selector="h3.title a",
        content_selector="div.snippet",
    ),
    "mbworld": ForumConfig(
        name="MBWorld",
        base_url="https://mbworld.org",
        search_path="/forums/search.php",
        thread_selector="li.searchResult",
        title_selector="h3.title a",
        content_selector="div.snippet",
    ),
}


class ForumScraper(BaseScraper):
    """Generic scraper for automotive forums."""

    source_type = SourceType.FORUM

    VEHICLE_PATTERN = re.compile(
        r"(\d{4})(?:\s*-\s*(\d{4}))?\s+([A-Za-z]+)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        forums: list[str] | None = None,
        requests_per_minute: int | None = None,
    ):
        """Initialize forum scraper.

        Args:
            forums: List of forum keys to scrape (e.g., ["bimmerpost", "toyotanation"])
            requests_per_minute: Rate limit
        """
        self.forums = forums or list(FORUM_CONFIGS.keys())
        self.requests_per_minute = requests_per_minute or settings.requests_per_minute
        self._delay = 60.0 / self.requests_per_minute
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; WessleyBot/1.0; +https://wessley.ai)",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                follow_redirects=True,
            )
        return self._client

    def _extract_vehicle(self, text: str) -> VehicleSignature | None:
        """Extract vehicle information from text."""
        match = self.VEHICLE_PATTERN.search(text)
        if not match:
            return None

        year_start = int(match.group(1))
        year_end = int(match.group(2)) if match.group(2) else year_start
        make = match.group(3).strip()
        model = match.group(4).strip()

        return VehicleSignature(
            make=make,
            model=model,
            year_start=year_start,
            year_end=year_end,
        )

    async def _fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse a page."""
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except Exception as e:
            logger.error("fetch_error", url=url, error=str(e))
            return None

    async def _scrape_forum(
        self,
        config: ForumConfig,
        query: str | None,
        limit: int,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape a single forum."""
        posts_scraped = 0

        if query:
            search_url = f"{config.base_url}{config.search_path}?q={query}"
        else:
            search_url = config.base_url

        logger.info("scraping_forum", forum=config.name, url=search_url)

        page_url = search_url
        while page_url and posts_scraped < limit:
            soup = await self._fetch_page(page_url)
            if not soup:
                break

            threads = soup.select(config.thread_selector)
            if not threads:
                logger.warning("no_threads_found", forum=config.name)
                break

            for thread in threads:
                if posts_scraped >= limit:
                    break

                try:
                    title_elem = thread.select_one(config.title_selector)
                    content_elem = thread.select_one(config.content_selector)
                    author_elem = thread.select_one(config.author_selector)
                    date_elem = thread.select_one(config.date_selector)

                    if not title_elem or not content_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    content = content_elem.get_text(strip=True)
                    full_text = f"{title}\n\n{content}"

                    url = title_elem.get("href")
                    if url and not url.startswith("http"):
                        url = urljoin(config.base_url, url)

                    post = ScrapedPost(
                        source=SourceType.FORUM,
                        source_id=url,
                        url=url,
                        title=title,
                        content=content,
                        author=author_elem.get_text(strip=True) if author_elem else None,
                        vehicle=self._extract_vehicle(full_text),
                        forum_name=config.name,
                    )

                    yield post
                    posts_scraped += 1

                except Exception as e:
                    logger.error("thread_parse_error", error=str(e))
                    continue

            await asyncio.sleep(self._delay)

            next_link = soup.select_one(config.next_page_selector)
            if next_link and next_link.get("href"):
                next_url = next_link.get("href")
                if not next_url.startswith("http"):
                    next_url = urljoin(config.base_url, next_url)
                page_url = next_url
            else:
                page_url = None

        logger.info("forum_complete", forum=config.name, posts=posts_scraped)

    async def scrape(
        self,
        query: str | None = None,
        limit: int = 1000,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape posts from forums.

        Args:
            query: Optional search query
            limit: Maximum number of posts per forum

        Yields:
            ScrapedPost objects
        """
        limit_per_forum = limit // len(self.forums)

        for forum_key in self.forums:
            config = FORUM_CONFIGS.get(forum_key)
            if not config:
                logger.warning("unknown_forum", forum=forum_key)
                continue

            async for post in self._scrape_forum(config, query, limit_per_forum):
                yield post

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
