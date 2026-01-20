"""API routes for scraper operations."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.core.logging import get_logger
from src.core.schemas import (
    JobStatusResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    ScrapeRequest,
    ScrapeResponse,
    ScraperJob,
    ScraperJobStatus,
    SourceType,
)
from src.jobs.queue import JobQueue
from src.services.persistence import PersistenceService

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["scraper"])


async def get_job_queue() -> JobQueue:
    """Dependency to get job queue."""
    from src.main import app

    return app.state.job_queue


async def get_persistence() -> PersistenceService:
    """Dependency to get persistence service."""
    from src.main import app

    return app.state.persistence


@router.post("/scrape", response_model=ScrapeResponse)
async def start_scrape(
    request: ScrapeRequest,
    job_queue: JobQueue = Depends(get_job_queue),
) -> ScrapeResponse:
    """Start a new scraping job.

    Args:
        request: Scrape request parameters
        job_queue: Job queue service

    Returns:
        Response with job ID and status
    """
    job = ScraperJob(
        source=request.source,
        query=request.query,
        vehicle=request.vehicle,
        subreddits=request.subreddits,
        forum_urls=request.forum_urls,
    )

    await job_queue.enqueue(job)

    logger.info(
        "scrape_job_started",
        job_id=str(job.id),
        source=request.source.value,
    )

    return ScrapeResponse(
        job_id=job.id,
        status=job.status,
        message=f"Scraping job started for {request.source.value}",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: UUID,
    job_queue: JobQueue = Depends(get_job_queue),
) -> JobStatusResponse:
    """Get the status of a scraping job.

    Args:
        job_id: The job ID
        job_queue: Job queue service

    Returns:
        Job status response
    """
    job = await job_queue.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = None
    if job.status == ScraperJobStatus.RUNNING and job.posts_scraped > 0:
        progress = min(100.0, job.posts_scraped / 100.0 * 100)

    return JobStatusResponse(job=job, progress_percent=progress)


@router.get("/jobs", response_model=list[ScraperJob])
async def list_jobs(
    limit: int = 100,
    job_queue: JobQueue = Depends(get_job_queue),
) -> list[ScraperJob]:
    """List recent scraping jobs.

    Args:
        limit: Maximum number of jobs to return
        job_queue: Job queue service

    Returns:
        List of jobs
    """
    return await job_queue.list_jobs(limit)


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    persistence: PersistenceService = Depends(get_persistence),
) -> KnowledgeSearchResponse:
    """Search the knowledge base.

    Args:
        request: Search request
        persistence: Persistence service

    Returns:
        Search results
    """
    results = await persistence.search(
        query=request.query,
        vehicle_make=request.vehicle.make if request.vehicle else None,
        vehicle_model=request.vehicle.model if request.vehicle else None,
        limit=request.limit,
    )

    return KnowledgeSearchResponse(
        results=[],
        total=len(results),
        query=request.query,
    )
