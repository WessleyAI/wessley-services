"""
Background worker for processing ingestion jobs.
"""
import asyncio
import uuid
from typing import Optional

from ..core.pipeline import IngestionPipeline
from ..core.schemas import CreateIngestionRequest
from .queue import queue_manager


class JobProcessor:
    """
    Background worker that processes ingestion jobs from the queue.
    """
    
    def __init__(self):
        self.running = False
        self.worker_id = str(uuid.uuid4())[:8]
    
    async def start(self) -> None:
        """Start the worker process."""
        print(f"Starting job processor worker {self.worker_id}")
        self.running = True
        
        while self.running:
            try:
                # Wait for a job with timeout
                job_data = await queue_manager.dequeue_job(timeout=30)
                
                if job_data:
                    await self._process_job(job_data)
                else:
                    # No job available, continue polling
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop(self) -> None:
        """Stop the worker process."""
        print(f"Stopping job processor worker {self.worker_id}")
        self.running = False
    
    async def _process_job(self, job_data: dict) -> None:
        """
        Process a single ingestion job.
        
        Args:
            job_data: Job data from the queue
        """
        job_id = uuid.UUID(job_data["job_id"])
        request_data = job_data["request"]
        
        print(f"Worker {self.worker_id} processing job {job_id}")
        
        try:
            # Parse request
            request = CreateIngestionRequest.model_validate(request_data)
            
            # Create and execute pipeline
            pipeline = IngestionPipeline(job_id, request)
            await pipeline.execute()
            
            print(f"Worker {self.worker_id} completed job {job_id}")
            
        except Exception as e:
            print(f"Worker {self.worker_id} failed job {job_id}: {e}")
            # Error handling is done in the pipeline


async def run_worker() -> None:
    """
    Run a job processor worker.
    This function can be called from a separate process or container.
    """
    # Connect to Redis
    await queue_manager.connect()
    
    try:
        # Create and start processor
        processor = JobProcessor()
        await processor.start()
        
    except KeyboardInterrupt:
        print("Worker interrupted by user")
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        # Cleanup
        await queue_manager.disconnect()


if __name__ == "__main__":
    # Run worker directly
    asyncio.run(run_worker())