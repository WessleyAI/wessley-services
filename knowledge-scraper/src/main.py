"""Knowledge Scraper Service - FastAPI Application."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api import scraper_router
from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.jobs.queue import JobQueue
from src.scrapers.reddit import RedditScraper
from src.scrapers.forum import ForumScraper
from src.services.extraction import KnowledgeExtractor
from src.services.persistence import PersistenceService
from src.core.schemas import ScraperJobStatus, SourceType

logger = get_logger(__name__)


async def run_scraper_worker(app: FastAPI) -> None:
    """Background worker that processes scraping jobs."""
    job_queue: JobQueue = app.state.job_queue
    persistence: PersistenceService = app.state.persistence
    extractor = KnowledgeExtractor()

    while True:
        try:
            job = await job_queue.dequeue()
            if not job:
                await asyncio.sleep(1)
                continue

            logger.info("processing_job", job_id=str(job.id), source=job.source.value)

            try:
                if job.source == SourceType.REDDIT:
                    scraper = RedditScraper(subreddits=job.subreddits)
                elif job.source == SourceType.FORUM:
                    scraper = ForumScraper()
                else:
                    await job_queue.fail(job, f"Unsupported source: {job.source.value}")
                    continue

                async for post in scraper.scrape(query=job.query, limit=1000):
                    await persistence.store_post(post)
                    job.posts_scraped += 1

                    if settings.enable_llm_extraction:
                        entries = await extractor.extract(post)
                        for entry in entries:
                            await persistence.store_knowledge(entry)
                            job.knowledge_extracted += 1

                    if job.posts_scraped % 10 == 0:
                        await job_queue.update(job)

                await scraper.close()
                await job_queue.complete(job)

            except Exception as e:
                logger.error("job_processing_error", job_id=str(job.id), error=str(e))
                await job_queue.fail(job, str(e))

        except asyncio.CancelledError:
            logger.info("worker_cancelled")
            break
        except Exception as e:
            logger.error("worker_error", error=str(e))
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    configure_logging()
    logger.info("starting_knowledge_scraper", env=settings.app_env)

    job_queue = JobQueue()
    await job_queue.initialize()
    app.state.job_queue = job_queue

    persistence = PersistenceService()
    await persistence.initialize()
    app.state.persistence = persistence

    worker_task = asyncio.create_task(run_scraper_worker(app))
    app.state.worker_task = worker_task

    logger.info("knowledge_scraper_started")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    await persistence.close()
    await job_queue.close()

    logger.info("knowledge_scraper_stopped")


app = FastAPI(
    title="Knowledge Scraper Service",
    description="Automotive knowledge acquisition from forums, YouTube, and parts catalogs",
    version="0.1.0",
    lifespan=lifespan,
)

if settings.is_production:
    allowed_origins = [
        "https://wessley.ai",
        "https://www.wessley.ai",
        "https://app.wessley.ai",
    ]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(scraper_router)


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Readiness check with dependency status."""
    status = {"status": "healthy"}

    try:
        if hasattr(app.state, "job_queue") and app.state.job_queue._redis:
            await app.state.job_queue._redis.ping()
            status["redis"] = "connected"
    except Exception:
        status["redis"] = "disconnected"
        status["status"] = "degraded"

    try:
        if hasattr(app.state, "persistence") and app.state.persistence._qdrant_client:
            await app.state.persistence._qdrant_client.get_collections()
            status["qdrant"] = "connected"
    except Exception:
        status["qdrant"] = "disconnected"
        status["status"] = "degraded"

    return status
