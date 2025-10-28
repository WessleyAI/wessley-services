"""
FastAPI routes for the ingestion service.
"""
import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from ..core.schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    CreateIngestionRequest,
    CreateIngestionResponse,
    GetIngestionResponse,
    HealthStatus,
    IngestionStatus,
)
from ..workers.queue import queue_manager

# Security
security = HTTPBearer()

# Routers
api_router = APIRouter(prefix="/v1")
health_router = APIRouter()


def get_current_user(token: str = Depends(security)) -> dict:
    """
    Validate JWT token and return user info.
    TODO: Implement proper JWT validation with Supabase.
    """
    # Placeholder implementation
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "placeholder", "sub": "placeholder"}


@api_router.post("/ingestions", response_model=CreateIngestionResponse)
async def create_ingestion(
    request: CreateIngestionRequest,
    current_user: dict = Depends(get_current_user),
) -> CreateIngestionResponse:
    """
    Create a new ingestion job.
    
    Validates the request, creates a job entry, and queues it for processing.
    """
    try:
        # Generate unique job ID
        job_id = uuid.uuid4()
        
        # TODO: Validate file access permissions
        # TODO: Store job metadata in Supabase
        
        # Enqueue job for processing
        await queue_manager.enqueue_job(
            job_id=job_id,
            request=request,
            user_id=current_user.get("user_id", "unknown"),
        )
        
        return CreateIngestionResponse(
            job_id=job_id,
            status=IngestionStatus.QUEUED,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ingestion job: {str(e)}",
        )


@api_router.get("/ingestions/{job_id}", response_model=GetIngestionResponse)
async def get_ingestion(
    job_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
) -> GetIngestionResponse:
    """
    Get ingestion job status and results.
    
    Returns current status, progress, artifacts, and metrics for the specified job.
    """
    try:
        # Get job status from queue
        job_data = await queue_manager.get_job_status(job_id)
        
        if not job_data:
            raise ValueError("Job not found")
        
        # TODO: Check user permissions for job access
        
        # Parse timestamps
        created_at = None
        updated_at = None
        if job_data.get("created_at"):
            created_at = datetime.fromisoformat(job_data["created_at"].replace("Z", "+00:00"))
        if job_data.get("updated_at"):
            updated_at = datetime.fromisoformat(job_data["updated_at"].replace("Z", "+00:00"))
        
        return GetIngestionResponse(
            job_id=job_id,
            status=IngestionStatus(job_data.get("status", "queued")),
            progress=int(job_data.get("progress", 0)) if job_data.get("progress") else None,
            current_stage=job_data.get("stage"),
            error=job_data.get("error"),
            created_at=created_at,
            updated_at=updated_at,
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ingestion job: {str(e)}",
        )


@api_router.post("/benchmarks/run")
async def run_benchmarks(
    request: BenchmarkRequest,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Run benchmark suite on test fixtures.
    
    Executes performance tests across different engines and datasets.
    """
    try:
        # TODO: Implement benchmark execution
        # TODO: Store results in Supabase benchmarks table
        
        return {
            "message": f"Benchmark started for engine: {request.engine}",
            "format": request.report_format,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run benchmarks: {str(e)}",
        )


@health_router.get("/healthz", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.
    
    Returns service status and dependency health.
    """
    # Check actual dependency health
    dependencies = {
        "redis": await queue_manager.health_check(),
        "supabase": True,  # TODO: Check Supabase connection
        "neo4j": True,  # TODO: Check Neo4j connection
        "qdrant": True,  # TODO: Check Qdrant connection
    }
    
    return HealthStatus(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow(),
        dependencies=dependencies,
    )


@health_router.get("/readyz", response_model=HealthStatus)
async def readiness_check() -> HealthStatus:
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to accept requests.
    """
    # Check actual dependency readiness
    dependencies = {
        "redis": await queue_manager.health_check(),
        "supabase": True,  # TODO: Check Supabase connection
        "neo4j": True,  # TODO: Check Neo4j connection
        "qdrant": True,  # TODO: Check Qdrant connection
    }
    
    # Check if all dependencies are ready
    all_ready = all(dependencies.values())
    
    if not all_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )
    
    return HealthStatus(
        status="ready",
        version="0.1.0",
        timestamp=datetime.utcnow(),
        dependencies=dependencies,
    )