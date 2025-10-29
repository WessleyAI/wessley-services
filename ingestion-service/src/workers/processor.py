"""
Background worker for processing ingestion jobs.
"""
import asyncio
import uuid
import os
import time
import psutil
from typing import Optional

from ..core.pipeline import IngestionPipeline
from ..core.schemas import CreateIngestionRequest
from ..core.logging import setup_logging, StructuredLogger, log_context
from .queue import queue_manager

# Setup ultra-verbose logging for worker
setup_logging(level='DEBUG', enable_structured=True)
logger = StructuredLogger(__name__)


class JobProcessor:
    """
    Background worker that processes ingestion jobs from the queue.
    """
    
    def __init__(self):
        self.running = False
        self.worker_id = str(uuid.uuid4())[:8]
        self.jobs_processed = 0
        self.start_time = time.time()
        
        # Log worker initialization with system info
        process = psutil.Process()
        logger.info("🤖 Initializing job processor worker",
                   worker_id=self.worker_id,
                   worker_pid=os.getpid(),
                   worker_ppid=os.getppid(),
                   worker_memory_mb=round(process.memory_info().rss / 1024 / 1024, 2),
                   worker_cpu_count=psutil.cpu_count(),
                   worker_hostname=os.uname().nodename,
                   worker_python_version=f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                   worker_cwd=os.getcwd(),
                   worker_environment=dict(os.environ))
    
    async def start(self) -> None:
        """Start the worker process."""
        logger.info("🚀 Starting job processor worker",
                   worker_id=self.worker_id,
                   worker_startup_time=time.time())
        self.running = True
        
        idle_count = 0
        
        while self.running:
            try:
                # Log polling attempt
                logger.debug("🔍 Polling for jobs",
                           worker_id=self.worker_id,
                           queue_poll_timeout=30,
                           idle_cycles=idle_count)
                
                # Wait for a job with timeout
                job_data = await queue_manager.dequeue_job(timeout=30)
                
                if job_data:
                    idle_count = 0
                    logger.info("📋 Job dequeued successfully",
                               worker_id=self.worker_id,
                               job_id=job_data.get("job_id"),
                               job_data_keys=list(job_data.keys()),
                               job_data_size_bytes=len(str(job_data)))
                    
                    await self._process_job(job_data)
                    self.jobs_processed += 1
                    
                    # Log worker stats
                    uptime = time.time() - self.start_time
                    logger.info("📊 Worker processing stats",
                               worker_id=self.worker_id,
                               jobs_processed_total=self.jobs_processed,
                               worker_uptime_seconds=round(uptime, 2),
                               jobs_per_minute=round((self.jobs_processed / uptime) * 60, 2) if uptime > 0 else 0)
                    
                else:
                    # No job available, continue polling
                    idle_count += 1
                    logger.debug("⏳ No jobs available, continuing to poll",
                               worker_id=self.worker_id,
                               idle_cycles=idle_count,
                               next_poll_delay_seconds=1)
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.exception("💥 Worker error occurred",
                               worker_id=self.worker_id,
                               error_type=type(e).__name__,
                               error_message=str(e),
                               retry_delay_seconds=5)
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop(self) -> None:
        """Stop the worker process."""
        uptime = time.time() - self.start_time
        logger.info("🛑 Stopping job processor worker",
                   worker_id=self.worker_id,
                   final_jobs_processed=self.jobs_processed,
                   final_uptime_seconds=round(uptime, 2),
                   final_jobs_per_minute=round((self.jobs_processed / uptime) * 60, 2) if uptime > 0 else 0)
        self.running = False
    
    async def _process_job(self, job_data: dict) -> None:
        """
        Process a single ingestion job.
        
        Args:
            job_data: Job data from the queue
        """
        job_id = uuid.UUID(job_data["job_id"])
        request_data = job_data["request"]
        user_id = job_data.get("user_id")
        created_at = job_data.get("created_at")
        
        # Set up logging context for this job
        with log_context(correlation_id=str(job_id), job_id=str(job_id), user_id=user_id):
            job_start_time = time.time()
            
            # Log detailed job start information
            logger.info("🔄 Starting job processing",
                       worker_id=self.worker_id,
                       job_id=str(job_id),
                       job_user_id=user_id,
                       job_created_at=created_at,
                       job_queue_wait_time=round(job_start_time - time.mktime(time.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%f")), 2) if created_at else None,
                       job_data_keys=list(job_data.keys()),
                       job_request_keys=list(request_data.keys()) if request_data else None,
                       worker_memory_before_mb=round(psutil.Process().memory_info().rss / 1024 / 1024, 2))
            
            try:
                # Parse and validate request
                logger.info("📋 Parsing job request data",
                           request_data_size_bytes=len(str(request_data)),
                           request_source_type=request_data.get("source", {}).get("type"),
                           request_file_id=request_data.get("source", {}).get("file_id"),
                           request_modes=request_data.get("modes"),
                           request_vehicle=request_data.get("doc_meta", {}).get("vehicle"))
                
                request = CreateIngestionRequest.model_validate(request_data)
                
                logger.info("✅ Job request validated successfully",
                           validated_source_type=request.source.type,
                           validated_modes_ocr=request.modes.ocr,
                           validated_modes_schematic_parse=request.modes.schematic_parse)
                
                # Create and execute pipeline
                logger.info("🏗️ Creating ingestion pipeline",
                           pipeline_job_id=str(job_id),
                           pipeline_source_type=request.source.type.value,
                           pipeline_file_id=request.source.file_id)
                
                pipeline = IngestionPipeline(job_id, request)
                
                logger.info("▶️ Executing ingestion pipeline")
                await pipeline.execute()
                
                job_duration = time.time() - job_start_time
                memory_after = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
                
                logger.info("🎉 Job processing completed successfully",
                           worker_id=self.worker_id,
                           job_id=str(job_id),
                           job_duration_seconds=round(job_duration, 2),
                           job_duration_minutes=round(job_duration / 60, 2),
                           worker_memory_after_mb=memory_after,
                           worker_memory_delta_mb=round(memory_after - psutil.Process().memory_info().rss / 1024 / 1024, 2))
                
            except Exception as e:
                job_duration = time.time() - job_start_time
                memory_after = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
                
                logger.exception("❌ Job processing failed",
                               worker_id=self.worker_id,
                               job_id=str(job_id),
                               job_duration_seconds=round(job_duration, 2),
                               error_type=type(e).__name__,
                               error_message=str(e),
                               worker_memory_after_mb=memory_after,
                               error_occurred_at=time.time())
                
                # Error handling is done in the pipeline, but we still want to track it
                raise


async def run_worker() -> None:
    """
    Run a job processor worker.
    This function can be called from a separate process or container.
    """
    logger.info("🏁 Starting worker process",
               worker_process_pid=os.getpid(),
               worker_process_cwd=os.getcwd(),
               worker_start_timestamp=time.time())
    
    try:
        # Connect to Redis with detailed logging
        logger.info("🔌 Connecting to Redis queue",
                   redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
        await queue_manager.connect()
        logger.info("✅ Redis connection established successfully")
        
        # Create and start processor
        logger.info("🤖 Creating job processor instance")
        processor = JobProcessor()
        
        logger.info("🚀 Starting job processor main loop")
        await processor.start()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Worker interrupted by user signal (Ctrl+C)")
    except Exception as e:
        logger.exception("💥 Worker process failed with exception",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        worker_shutdown_reason="exception")
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up worker resources")
        try:
            await queue_manager.disconnect()
            logger.info("✅ Redis disconnection completed")
        except Exception as cleanup_error:
            logger.exception("❌ Error during cleanup",
                           cleanup_error_type=type(cleanup_error).__name__,
                           cleanup_error_message=str(cleanup_error))
        
        logger.info("🏁 Worker process terminated",
                   worker_termination_timestamp=time.time())


if __name__ == "__main__":
    # Run worker directly
    asyncio.run(run_worker())