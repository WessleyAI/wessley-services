"""Redis-based job queue for scraping jobs."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import redis.asyncio as redis

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import ScraperJob, ScraperJobStatus

logger = get_logger(__name__)


class JobQueue:
    """Redis-based job queue for managing scraper jobs."""

    JOBS_KEY = "knowledge_scraper:jobs"
    QUEUE_KEY = "knowledge_scraper:queue"

    def __init__(self):
        """Initialize the job queue."""
        self._redis: redis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self._redis = redis.from_url(settings.redis_url)
        logger.info("redis_connected", url=settings.redis_url)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _serialize_job(self, job: ScraperJob) -> str:
        """Serialize a job to JSON."""
        data = job.model_dump()
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
        return json.dumps(data)

    def _deserialize_job(self, data: str) -> ScraperJob:
        """Deserialize a job from JSON."""
        return ScraperJob.model_validate_json(data)

    async def enqueue(self, job: ScraperJob) -> None:
        """Add a job to the queue.

        Args:
            job: The job to enqueue
        """
        if not self._redis:
            raise RuntimeError("Redis not initialized")

        await self._redis.hset(
            self.JOBS_KEY,
            str(job.id),
            self._serialize_job(job),
        )

        await self._redis.rpush(self.QUEUE_KEY, str(job.id))

        logger.info("job_enqueued", job_id=str(job.id))

    async def dequeue(self) -> ScraperJob | None:
        """Get the next job from the queue.

        Returns:
            The next job or None if queue is empty
        """
        if not self._redis:
            raise RuntimeError("Redis not initialized")

        result = await self._redis.blpop(self.QUEUE_KEY, timeout=1)
        if not result:
            return None

        _, job_id = result
        job_data = await self._redis.hget(self.JOBS_KEY, job_id)

        if not job_data:
            return None

        job = self._deserialize_job(job_data)

        job.status = ScraperJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await self.update(job)

        return job

    async def get(self, job_id: UUID) -> ScraperJob | None:
        """Get a job by ID.

        Args:
            job_id: The job ID

        Returns:
            The job or None if not found
        """
        if not self._redis:
            raise RuntimeError("Redis not initialized")

        job_data = await self._redis.hget(self.JOBS_KEY, str(job_id))
        if not job_data:
            return None

        return self._deserialize_job(job_data)

    async def update(self, job: ScraperJob) -> None:
        """Update a job.

        Args:
            job: The job to update
        """
        if not self._redis:
            raise RuntimeError("Redis not initialized")

        await self._redis.hset(
            self.JOBS_KEY,
            str(job.id),
            self._serialize_job(job),
        )

        logger.debug("job_updated", job_id=str(job.id), status=job.status.value)

    async def complete(self, job: ScraperJob) -> None:
        """Mark a job as completed.

        Args:
            job: The job to complete
        """
        job.status = ScraperJobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        await self.update(job)

        logger.info(
            "job_completed",
            job_id=str(job.id),
            posts_scraped=job.posts_scraped,
            knowledge_extracted=job.knowledge_extracted,
        )

    async def fail(self, job: ScraperJob, error: str) -> None:
        """Mark a job as failed.

        Args:
            job: The job that failed
            error: Error message
        """
        job.status = ScraperJobStatus.FAILED
        job.errors.append(error)
        job.completed_at = datetime.utcnow()
        await self.update(job)

        logger.error("job_failed", job_id=str(job.id), error=error)

    async def list_jobs(self, limit: int = 100) -> list[ScraperJob]:
        """List recent jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        if not self._redis:
            raise RuntimeError("Redis not initialized")

        all_jobs = await self._redis.hgetall(self.JOBS_KEY)
        jobs = [self._deserialize_job(data) for data in all_jobs.values()]

        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]
