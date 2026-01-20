"""Tests for Reddit scraper."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.scrapers.reddit import RedditScraper
from src.core.schemas import SourceType


class TestRedditScraper:
    """Tests for RedditScraper."""

    def test_scraper_source_type(self):
        """Test scraper has correct source type."""
        scraper = RedditScraper()
        assert scraper.source_type == SourceType.REDDIT

    def test_default_subreddits(self):
        """Test default subreddits are set."""
        scraper = RedditScraper()
        assert "MechanicAdvice" in scraper.subreddits
        assert "cartalk" in scraper.subreddits

    def test_custom_subreddits(self):
        """Test custom subreddits."""
        scraper = RedditScraper(subreddits=["customsub"])
        assert scraper.subreddits == ["customsub"]

    def test_rate_limit_delay(self):
        """Test rate limit delay calculation."""
        scraper = RedditScraper(requests_per_minute=30)
        assert scraper._delay == 2.0

        scraper = RedditScraper(requests_per_minute=60)
        assert scraper._delay == 1.0

    def test_extract_vehicle_full(self):
        """Test vehicle extraction with full info."""
        scraper = RedditScraper()
        text = "My 2008 Honda Accord has window issues"

        vehicle = scraper._extract_vehicle(text)

        assert vehicle is not None
        assert vehicle.make == "Honda"
        assert vehicle.model == "Accord"
        assert vehicle.year_start == 2008
        assert vehicle.year_end == 2008

    def test_extract_vehicle_year_range(self):
        """Test vehicle extraction with year range."""
        scraper = RedditScraper()
        text = "2005-2010 Toyota Camry common problem"

        vehicle = scraper._extract_vehicle(text)

        assert vehicle is not None
        assert vehicle.make == "Toyota"
        assert vehicle.model == "Camry"
        assert vehicle.year_start == 2005
        assert vehicle.year_end == 2010

    def test_extract_vehicle_multi_word_model(self):
        """Test vehicle extraction with multi-word model."""
        scraper = RedditScraper()
        text = "2015 Ford F150 XLT brake issues"

        vehicle = scraper._extract_vehicle(text)

        assert vehicle is not None
        assert vehicle.make == "Ford"
        assert vehicle.model == "F150 XLT"

    def test_extract_vehicle_no_match(self):
        """Test vehicle extraction with no vehicle info."""
        scraper = RedditScraper()
        text = "General car maintenance tips"

        vehicle = scraper._extract_vehicle(text)

        assert vehicle is None

    @pytest.mark.asyncio
    async def test_scraper_requires_credentials(self):
        """Test scraper raises error without credentials."""
        scraper = RedditScraper()

        with patch.object(scraper, '_get_reddit') as mock_get:
            mock_get.side_effect = ValueError("Reddit API credentials not configured")

            with pytest.raises(ValueError, match="credentials"):
                async for _ in scraper.scrape(limit=1):
                    pass

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """Test close method cleans up resources."""
        scraper = RedditScraper()
        scraper._reddit = Mock()

        await scraper.close()

        assert scraper._reddit is None
