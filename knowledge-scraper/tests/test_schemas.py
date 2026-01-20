"""Tests for Pydantic schemas."""

import pytest
from uuid import UUID

from src.core.schemas import (
    ComponentConnection,
    ComponentLocation,
    KnowledgeEntry,
    KnowledgeType,
    ScrapedPost,
    ScraperJob,
    ScraperJobStatus,
    SourceType,
    VehicleSignature,
)


class TestVehicleSignature:
    """Tests for VehicleSignature."""

    def test_create_vehicle_signature(self):
        """Test creating a vehicle signature."""
        vehicle = VehicleSignature(
            make="Honda",
            model="Accord",
            year_start=2008,
            year_end=2012,
        )

        assert vehicle.make == "Honda"
        assert vehicle.model == "Accord"
        assert vehicle.year_start == 2008
        assert vehicle.year_end == 2012

    def test_vehicle_matches_exact(self):
        """Test exact vehicle matching."""
        v1 = VehicleSignature(make="Honda", model="Accord", year_start=2008, year_end=2012)
        v2 = VehicleSignature(make="Honda", model="Accord", year_start=2010, year_end=2010)

        assert v1.matches(v2)
        assert v2.matches(v1)

    def test_vehicle_matches_partial(self):
        """Test partial vehicle matching."""
        v1 = VehicleSignature(make="Honda")
        v2 = VehicleSignature(make="Honda", model="Accord")

        assert v1.matches(v2)
        assert v2.matches(v1)

    def test_vehicle_no_match_different_make(self):
        """Test vehicles with different makes don't match."""
        v1 = VehicleSignature(make="Honda")
        v2 = VehicleSignature(make="Toyota")

        assert not v1.matches(v2)

    def test_vehicle_no_match_different_years(self):
        """Test vehicles with non-overlapping years don't match."""
        v1 = VehicleSignature(year_start=2000, year_end=2005)
        v2 = VehicleSignature(year_start=2010, year_end=2015)

        assert not v1.matches(v2)


class TestScrapedPost:
    """Tests for ScrapedPost."""

    def test_create_scraped_post(self):
        """Test creating a scraped post."""
        post = ScrapedPost(
            source=SourceType.REDDIT,
            source_id="abc123",
            title="2008 Accord window issue",
            content="My power window stopped working...",
            subreddit="MechanicAdvice",
        )

        assert post.source == SourceType.REDDIT
        assert post.source_id == "abc123"
        assert post.title == "2008 Accord window issue"
        assert post.subreddit == "MechanicAdvice"
        assert isinstance(post.id, UUID)

    def test_scraped_post_with_vehicle(self):
        """Test scraped post with vehicle info."""
        vehicle = VehicleSignature(make="Honda", model="Accord")
        post = ScrapedPost(
            source=SourceType.FORUM,
            content="Test content",
            vehicle=vehicle,
            forum_name="HondaTech",
        )

        assert post.vehicle is not None
        assert post.vehicle.make == "Honda"
        assert post.forum_name == "HondaTech"


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry."""

    def test_create_knowledge_entry(self):
        """Test creating a knowledge entry."""
        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.LOCATION,
            component_type="relay",
            confidence=0.85,
        )

        assert entry.knowledge_type == KnowledgeType.LOCATION
        assert entry.component_type == "relay"
        assert entry.confidence == 0.85
        assert isinstance(entry.id, UUID)

    def test_knowledge_entry_with_location(self):
        """Test knowledge entry with location."""
        location = ComponentLocation(
            component_type="relay",
            zone="engine_bay",
            sub_zone="fuse_box",
            access="easy",
        )

        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.LOCATION,
            location=location,
        )

        assert entry.location is not None
        assert entry.location.zone == "engine_bay"
        assert entry.location.sub_zone == "fuse_box"

    def test_knowledge_entry_with_connections(self):
        """Test knowledge entry with connections."""
        conn = ComponentConnection(
            source="BCM",
            target="relay",
            connection_type="control",
            wire_color="green",
        )

        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.CONNECTION,
            connections=[conn],
        )

        assert len(entry.connections) == 1
        assert entry.connections[0].source == "BCM"
        assert entry.connections[0].wire_color == "green"


class TestScraperJob:
    """Tests for ScraperJob."""

    def test_create_scraper_job(self):
        """Test creating a scraper job."""
        job = ScraperJob(
            source=SourceType.REDDIT,
            subreddits=["MechanicAdvice", "cartalk"],
        )

        assert job.source == SourceType.REDDIT
        assert job.status == ScraperJobStatus.PENDING
        assert len(job.subreddits) == 2
        assert job.posts_scraped == 0

    def test_scraper_job_with_vehicle_filter(self):
        """Test scraper job with vehicle filter."""
        vehicle = VehicleSignature(make="Toyota", model="Camry")
        job = ScraperJob(
            source=SourceType.FORUM,
            vehicle=vehicle,
            query="window motor",
        )

        assert job.vehicle is not None
        assert job.vehicle.make == "Toyota"
        assert job.query == "window motor"
