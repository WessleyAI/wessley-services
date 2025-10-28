"""
Redis queue implementation for job processing.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from ..core.schemas import CreateIngestionRequest, IngestionStatus, RealtimeUpdate


class QueueManager:
    """
    Manages Redis queue operations for ingestion jobs.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize queue manager with Redis connection."""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client: Optional[Redis] = None
        self.queue_name = "ingestion_jobs"
        
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            # Test connection
            await self.redis_client.ping()
            print(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            print("Disconnected from Redis")
    
    async def enqueue_job(
        self, 
        job_id: uuid.UUID, 
        request: CreateIngestionRequest,
        user_id: str,
    ) -> None:
        """
        Enqueue an ingestion job for processing.
        
        Args:
            job_id: Unique job identifier
            request: Ingestion request details
            user_id: User who created the job
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        job_data = {
            "job_id": str(job_id),
            "user_id": user_id,
            "request": request.model_dump(),
            "status": IngestionStatus.QUEUED.value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Store job data
        await self.redis_client.hset(
            f"job:{job_id}",
            mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in job_data.items()}
        )
        
        # Add to processing queue
        await self.redis_client.lpush(self.queue_name, str(job_id))
        
        print(f"Enqueued job {job_id} for processing")
    
    async def dequeue_job(self, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Dequeue the next job for processing.
        
        Args:
            timeout: Blocking timeout in seconds
            
        Returns:
            Job data dictionary or None if timeout
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            # Blocking pop from queue
            result = await self.redis_client.brpop(self.queue_name, timeout=timeout)
            
            if not result:
                return None
            
            queue_name, job_id = result
            
            # Get job data
            job_data = await self.redis_client.hgetall(f"job:{job_id}")
            
            if not job_data:
                print(f"Warning: Job {job_id} not found in storage")
                return None
            
            # Parse JSON fields
            for key in ["request"]:
                if key in job_data:
                    job_data[key] = json.loads(job_data[key])
            
            print(f"Dequeued job {job_id} for processing")
            return job_data
            
        except Exception as e:
            print(f"Error dequeuing job: {e}")
            return None
    
    async def update_job_status(
        self,
        job_id: uuid.UUID,
        status: IngestionStatus,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update job status and metadata.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Progress percentage (0-100)
            stage: Current processing stage
            error: Error message if failed
            artifacts: Generated artifacts URLs
            metrics: Processing metrics
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        if progress is not None:
            update_data["progress"] = str(progress)
        if stage:
            update_data["stage"] = stage
        if error:
            update_data["error"] = error
        if artifacts:
            update_data["artifacts"] = json.dumps(artifacts)
        if metrics:
            update_data["metrics"] = json.dumps(metrics)
        
        # Update job data
        await self.redis_client.hset(f"job:{job_id}", mapping=update_data)
        
        print(f"Updated job {job_id} status to {status.value}")
    
    async def get_job_status(self, job_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Get current job status and data.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data dictionary or None if not found
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        job_data = await self.redis_client.hgetall(f"job:{job_id}")
        
        if not job_data:
            return None
        
        # Parse JSON fields
        for key in ["request", "artifacts", "metrics"]:
            if key in job_data and job_data[key]:
                try:
                    job_data[key] = json.loads(job_data[key])
                except json.JSONDecodeError:
                    pass
        
        return job_data
    
    async def publish_realtime_update(
        self,
        job_id: uuid.UUID,
        status: IngestionStatus,
        progress: int,
        stage: str,
        metrics: Optional[Dict[str, float]] = None,
        channel: Optional[str] = None,
    ) -> None:
        """
        Publish realtime update to notification channel.
        
        Args:
            job_id: Job identifier
            status: Current status
            progress: Progress percentage
            stage: Current stage
            metrics: Processing metrics
            channel: Notification channel (optional)
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        update = RealtimeUpdate(
            job_id=job_id,
            status=status,
            progress=progress,
            stage=stage,
            metrics=metrics,
            timestamp=datetime.utcnow(),
        )
        
        # Default channel
        if not channel:
            channel = f"realtime:ingestions:{job_id}"
        
        await self.redis_client.publish(
            channel,
            update.model_dump_json(),
        )
        
        print(f"Published realtime update for job {job_id} to {channel}")
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.redis_client:
                return False
            await self.redis_client.ping()
            return True
        except Exception:
            return False


# Global queue manager instance
queue_manager = QueueManager()