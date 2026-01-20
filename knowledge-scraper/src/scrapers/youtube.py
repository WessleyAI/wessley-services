"""YouTube transcript scraper."""

import asyncio
import re
from typing import AsyncIterator

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import ScrapedPost, SourceType, VehicleSignature

from .base import BaseScraper

logger = get_logger(__name__)


class YouTubeScraper(BaseScraper):
    """Scraper for YouTube repair video transcripts."""

    source_type = SourceType.YOUTUBE

    AUTOMOTIVE_CHANNELS = [
        "ChrisFix",
        "ScottyKilmer",
        "SouthMainAutoRepairAvoca",
        "EricTheCarGuy",
        "RainmanRaysRepairs",
        "1AAuto",
        "BleepinJeep",
    ]

    AUTOMOTIVE_KEYWORDS = [
        "car repair",
        "auto repair",
        "mechanic",
        "how to fix",
        "engine",
        "transmission",
        "brakes",
        "electrical",
        "wiring",
        "fuse",
        "relay",
        "alternator",
        "starter",
        "battery",
    ]

    VEHICLE_PATTERN = re.compile(
        r"(\d{4})(?:\s*-\s*(\d{4}))?\s+([A-Za-z]+)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        channels: list[str] | None = None,
        requests_per_minute: int | None = None,
    ):
        """Initialize YouTube scraper.

        Args:
            channels: List of channel names to prioritize
            requests_per_minute: Rate limit
        """
        self.channels = channels or self.AUTOMOTIVE_CHANNELS
        self.requests_per_minute = requests_per_minute or settings.requests_per_minute
        self._delay = 60.0 / self.requests_per_minute

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

    def _get_transcript(self, video_id: str) -> str | None:
        """Get transcript for a video."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript_list])
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
            logger.debug("transcript_unavailable", video_id=video_id, error=str(e))
            return None
        except Exception as e:
            logger.error("transcript_error", video_id=video_id, error=str(e))
            return None

    async def scrape_video(self, video_id: str, title: str = "") -> ScrapedPost | None:
        """Scrape a single video transcript.

        Args:
            video_id: YouTube video ID
            title: Video title (optional)

        Returns:
            ScrapedPost or None if transcript unavailable
        """
        transcript = await asyncio.to_thread(self._get_transcript, video_id)
        if not transcript:
            return None

        full_text = f"{title}\n\n{transcript}"
        vehicle = self._extract_vehicle(full_text)

        return ScrapedPost(
            source=SourceType.YOUTUBE,
            source_id=video_id,
            url=f"https://youtube.com/watch?v={video_id}",
            title=title,
            content=transcript,
            vehicle=vehicle,
        )

    async def scrape(
        self,
        query: str | None = None,
        limit: int = 1000,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape YouTube video transcripts.

        Note: This is a placeholder that processes video IDs.
        Full implementation would require YouTube Data API for search.

        Args:
            query: Search query (requires YouTube API key)
            limit: Maximum number of videos

        Yields:
            ScrapedPost objects
        """
        logger.warning(
            "youtube_scraper_placeholder",
            message="Full YouTube search requires YouTube Data API key. "
            "Use scrape_video() with known video IDs.",
        )

    async def scrape_video_ids(
        self,
        video_ids: list[str],
        titles: list[str] | None = None,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape transcripts for a list of video IDs.

        Args:
            video_ids: List of YouTube video IDs
            titles: Optional list of titles (same length as video_ids)

        Yields:
            ScrapedPost objects
        """
        titles = titles or [""] * len(video_ids)

        for video_id, title in zip(video_ids, titles):
            post = await self.scrape_video(video_id, title)
            if post:
                yield post

            await asyncio.sleep(self._delay)

    async def close(self) -> None:
        """Clean up resources."""
        pass
