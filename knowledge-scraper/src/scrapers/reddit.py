"""Reddit scraper using PRAW."""

import asyncio
import re
from datetime import datetime
from typing import AsyncIterator

import praw
from praw.models import Submission

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import ScrapedPost, SourceType, VehicleSignature

from .base import BaseScraper

logger = get_logger(__name__)


class RedditScraper(BaseScraper):
    """Scraper for Reddit automotive subreddits."""

    source_type = SourceType.REDDIT

    DEFAULT_SUBREDDITS = [
        "MechanicAdvice",
        "cartalk",
        "AskMechanics",
        "Justrolledintotheshop",
        "autorepair",
        "CarHelp",
        "MechanicAdviceEurope",
    ]

    VEHICLE_PATTERN = re.compile(
        r"(\d{4})(?:\s*-\s*(\d{4}))?\s+([A-Za-z]+)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        subreddits: list[str] | None = None,
        requests_per_minute: int | None = None,
    ):
        """Initialize Reddit scraper.

        Args:
            subreddits: List of subreddits to scrape
            requests_per_minute: Rate limit
        """
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self.requests_per_minute = requests_per_minute or settings.requests_per_minute
        self._reddit: praw.Reddit | None = None
        self._delay = 60.0 / self.requests_per_minute

    def _get_reddit(self) -> praw.Reddit:
        """Get or create Reddit client."""
        if self._reddit is None:
            if not settings.reddit_client_id or not settings.reddit_client_secret:
                raise ValueError("Reddit API credentials not configured")

            self._reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )
        return self._reddit

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

    def _submission_to_post(self, submission: Submission, subreddit: str) -> ScrapedPost:
        """Convert a Reddit submission to a ScrapedPost."""
        full_text = f"{submission.title}\n\n{submission.selftext}"
        vehicle = self._extract_vehicle(full_text)

        return ScrapedPost(
            source=SourceType.REDDIT,
            source_id=submission.id,
            url=f"https://reddit.com{submission.permalink}",
            title=submission.title,
            content=submission.selftext or submission.title,
            author=str(submission.author) if submission.author else None,
            created_at=datetime.fromtimestamp(submission.created_utc),
            vehicle=vehicle,
            subreddit=subreddit,
            upvotes=submission.score,
            comments_count=submission.num_comments,
        )

    async def scrape(
        self,
        query: str | None = None,
        limit: int = 1000,
    ) -> AsyncIterator[ScrapedPost]:
        """Scrape posts from Reddit.

        Args:
            query: Optional search query
            limit: Maximum number of posts per subreddit

        Yields:
            ScrapedPost objects
        """
        reddit = self._get_reddit()
        posts_scraped = 0
        limit_per_subreddit = limit // len(self.subreddits)

        for subreddit_name in self.subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                logger.info(
                    "scraping_subreddit",
                    subreddit=subreddit_name,
                    limit=limit_per_subreddit,
                )

                if query:
                    submissions = subreddit.search(query, limit=limit_per_subreddit)
                else:
                    submissions = subreddit.hot(limit=limit_per_subreddit)

                for submission in submissions:
                    if not submission.selftext:
                        continue

                    post = self._submission_to_post(submission, subreddit_name)
                    yield post
                    posts_scraped += 1

                    await asyncio.sleep(self._delay)

                    if posts_scraped >= limit:
                        logger.info("limit_reached", total=posts_scraped)
                        return

            except Exception as e:
                logger.error(
                    "subreddit_scrape_error",
                    subreddit=subreddit_name,
                    error=str(e),
                )
                continue

        logger.info("scraping_complete", total=posts_scraped)

    async def scrape_comments(
        self,
        submission_id: str,
        limit: int = 100,
    ) -> list[str]:
        """Scrape comments from a submission.

        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments

        Returns:
            List of comment bodies
        """
        reddit = self._get_reddit()
        comments = []

        try:
            submission = reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list()[:limit]:
                if comment.body and comment.body != "[deleted]":
                    comments.append(comment.body)
                    await asyncio.sleep(self._delay)

        except Exception as e:
            logger.error(
                "comment_scrape_error",
                submission_id=submission_id,
                error=str(e),
            )

        return comments

    async def close(self) -> None:
        """Clean up Reddit client."""
        self._reddit = None
