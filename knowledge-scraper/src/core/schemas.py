"""Pydantic schemas for the knowledge scraper service."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ScraperJobStatus(str, Enum):
    """Status of a scraper job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KnowledgeType(str, Enum):
    """Type of extracted knowledge."""

    LOCATION = "location"
    CONNECTION = "connection"
    FAILURE = "failure"
    SPECIFICATION = "specification"
    HARNESS_PATH = "harness_path"
    ACCESSIBILITY = "accessibility"


class SourceType(str, Enum):
    """Source of scraped data."""

    REDDIT = "reddit"
    FORUM = "forum"
    YOUTUBE = "youtube"
    IFIXIT = "ifixit"
    PARTS_CATALOG = "parts_catalog"
    NHTSA = "nhtsa"


class VehicleSignature(BaseModel):
    """Vehicle identification."""

    make: Optional[str] = None
    model: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    engine: Optional[str] = None
    trim: Optional[str] = None

    def matches(self, other: "VehicleSignature") -> bool:
        """Check if this signature matches another."""
        if self.make and other.make and self.make.lower() != other.make.lower():
            return False
        if self.model and other.model and self.model.lower() != other.model.lower():
            return False
        if self.year_start and other.year_end and self.year_start > other.year_end:
            return False
        if self.year_end and other.year_start and self.year_end < other.year_start:
            return False
        return True


class ComponentLocation(BaseModel):
    """Location of a component in a vehicle."""

    component_type: str
    zone: str
    sub_zone: Optional[str] = None
    side: Optional[str] = None
    access: Optional[str] = None
    description: Optional[str] = None


class ComponentConnection(BaseModel):
    """Connection between components."""

    source: str
    target: str
    connection_type: str
    wire_color: Optional[str] = None
    wire_gauge: Optional[str] = None
    description: Optional[str] = None


class ScrapedPost(BaseModel):
    """Raw scraped post from a source."""

    id: UUID = Field(default_factory=uuid4)
    source: SourceType
    source_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    content: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    vehicle: Optional[VehicleSignature] = None
    subreddit: Optional[str] = None
    forum_name: Optional[str] = None
    upvotes: Optional[int] = None
    comments_count: Optional[int] = None


class KnowledgeEntry(BaseModel):
    """Extracted knowledge from a post."""

    id: UUID = Field(default_factory=uuid4)
    post_id: Optional[UUID] = None
    knowledge_type: KnowledgeType
    vehicle: Optional[VehicleSignature] = None
    component_type: Optional[str] = None
    location: Optional[ComponentLocation] = None
    connections: list[ComponentConnection] = Field(default_factory=list)
    symptom: Optional[str] = None
    root_cause: Optional[str] = None
    solution: Optional[str] = None
    cost_estimate: Optional[float] = None
    confidence: float = 0.5
    raw_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ScraperJob(BaseModel):
    """A scraper job."""

    id: UUID = Field(default_factory=uuid4)
    source: SourceType
    status: ScraperJobStatus = ScraperJobStatus.PENDING
    query: Optional[str] = None
    vehicle: Optional[VehicleSignature] = None
    subreddits: list[str] = Field(default_factory=list)
    forum_urls: list[str] = Field(default_factory=list)
    posts_scraped: int = 0
    knowledge_extracted: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ScrapeRequest(BaseModel):
    """Request to start a scraping job."""

    source: SourceType
    query: Optional[str] = None
    vehicle: Optional[VehicleSignature] = None
    subreddits: list[str] = Field(
        default_factory=lambda: ["MechanicAdvice", "cartalk", "AskMechanics"]
    )
    forum_urls: list[str] = Field(default_factory=list)
    limit: int = 1000


class ScrapeResponse(BaseModel):
    """Response from starting a scraping job."""

    job_id: UUID
    status: ScraperJobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response with job status."""

    job: ScraperJob
    progress_percent: Optional[float] = None


class KnowledgeSearchRequest(BaseModel):
    """Request to search knowledge."""

    query: str
    vehicle: Optional[VehicleSignature] = None
    knowledge_type: Optional[KnowledgeType] = None
    limit: int = 10


class KnowledgeSearchResponse(BaseModel):
    """Response with search results."""

    results: list[KnowledgeEntry]
    total: int
    query: str
