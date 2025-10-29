"""
FastAPI routes for the ingestion service.
"""
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer
from fastapi.responses import PlainTextResponse

from ..core.schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    CreateIngestionRequest,
    CreateIngestionResponse,
    GetIngestionResponse,
    HealthStatus,
    IngestionStatus,
)
from ..core.health import health_checker, SystemHealth
from ..core.metrics import metrics
from ..core.logging import StructuredLogger, log_context, generate_correlation_id
from ..core.security import security_manager, AuthenticationMiddleware, UserContext, require_auth
from ..core.sentry import sentry_manager
from ..workers.queue import queue_manager
from ..semantic.search import get_search_engine, SearchFilter
from ..semantic.ontology import VehicleSignature
from ..evaluation.runner import ComprehensiveEvaluationRunner, EvaluationConfig
from ..evaluation.benchmarks import BenchmarkManager, AutomotiveBenchmarks

# Initialize logger
logger = StructuredLogger(__name__)

# Security
security = HTTPBearer()
auth_middleware = AuthenticationMiddleware(security_manager)

# Routers
api_router = APIRouter(prefix="/v1")
health_router = APIRouter()
metrics_router = APIRouter()


async def get_current_user(request: Request) -> UserContext:
    """
    Validate JWT token and return user context.
    Uses the security middleware for authentication and rate limiting.
    """
    import os
    # Skip auth if REQUIRE_AUTH is false
    if os.getenv("REQUIRE_AUTH", "true").lower() == "false":
        return UserContext(user_id="dev-user", email="dev@example.com")
    
    return await auth_middleware(request)


@api_router.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables."""
    import os
    return {
        "REQUIRE_AUTH": os.getenv("REQUIRE_AUTH", "not_set"),
        "APP_ENV": os.getenv("APP_ENV", "not_set"),
        "PORT": os.getenv("PORT", "not_set")
    }


@api_router.post("/ingestions", response_model=CreateIngestionResponse)
async def create_ingestion(
    request_data: CreateIngestionRequest,
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
) -> CreateIngestionResponse:
    """
    Create a new ingestion job.
    
    Validates the request, creates a job entry, and queues it for processing.
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with log_context(correlation_id=correlation_id, user_id=current_user.user_id):
        try:
            # Generate unique job ID
            job_id = uuid.uuid4()
            
            logger.log_user_action("create_ingestion", 
                                 job_id=str(job_id),
                                 source_type=request_data.source.type,
                                 modes=request_data.modes)
            
            # Sanitize input data
            sanitized_request = security_manager.sanitize_input(request_data.dict())
            
            # Record metrics
            metrics.record_job_started(current_user.user_id)
            
            # Enqueue job for processing
            await queue_manager.enqueue_job(
                job_id=job_id,
                request=request_data,
                user_id=current_user.user_id,
            )
            
            # Record success metrics
            duration = time.time() - start_time
            metrics.record_http_request("POST", "/v1/ingestions", 200, duration)
            
            logger.log_operation_success("create_ingestion", duration, job_id=str(job_id))
            
            return CreateIngestionResponse(
                job_id=str(job_id),
                status=IngestionStatus.QUEUED,
            )
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics.record_http_request("POST", "/v1/ingestions", 500, duration)
            metrics.record_error(type(e).__name__, "api", "error")
            
            # Log and capture error
            logger.log_operation_error("create_ingestion", e, duration, job_id=str(job_id))
            sentry_manager.capture_exception(
                e,
                tags={"endpoint": "create_ingestion", "user_id": current_user.user_id},
                extra={"request_data": request_data.json() if hasattr(request_data, 'json') else str(request_data)}
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create ingestion job: {str(e)}",
            )


@api_router.get("/ingestions/{job_id}", response_model=GetIngestionResponse)
async def get_ingestion(
    job_id: uuid.UUID,
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
) -> GetIngestionResponse:
    """
    Get ingestion job status and results.
    
    Returns current status, progress, artifacts, and metrics for the specified job.
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with log_context(correlation_id=correlation_id, user_id=current_user.user_id, job_id=str(job_id)):
        try:
            logger.log_user_action("get_ingestion", job_id=str(job_id))
            
            # Get job status from queue
            job_data = await queue_manager.get_job_status(job_id)
            
            if not job_data:
                metrics.record_http_request("GET", f"/v1/ingestions/{job_id}", 404, time.time() - start_time)
                raise ValueError("Job not found")
            
            # Parse timestamps
            created_at = None
            updated_at = None
            if job_data.get("created_at"):
                created_at = datetime.fromisoformat(job_data["created_at"].replace("Z", "+00:00"))
            if job_data.get("updated_at"):
                updated_at = datetime.fromisoformat(job_data["updated_at"].replace("Z", "+00:00"))
            
            # Record success metrics
            duration = time.time() - start_time
            metrics.record_http_request("GET", f"/v1/ingestions/{job_id}", 200, duration)
            
            logger.log_operation_success("get_ingestion", duration, job_id=str(job_id))
            
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
            duration = time.time() - start_time
            metrics.record_http_request("GET", f"/v1/ingestions/{job_id}", 404, duration)
            logger.warning("Job not found", job_id=str(job_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ingestion job not found",
            )
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics.record_http_request("GET", f"/v1/ingestions/{job_id}", 500, duration)
            metrics.record_error(type(e).__name__, "api", "error")
            
            # Log and capture error
            logger.log_operation_error("get_ingestion", e, duration, job_id=str(job_id))
            sentry_manager.capture_exception(
                e,
                tags={"endpoint": "get_ingestion", "user_id": current_user.user_id},
                extra={"job_id": str(job_id)}
            )
            
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


@health_router.get("/healthz")
async def health_check() -> Dict:
    """
    Basic health check endpoint.
    
    Returns service status and dependency health.
    """
    try:
        system_health = await health_checker.check_all(include_external=True)
        
        response_data = system_health.to_dict()
        
        # Determine HTTP status based on health
        if system_health.status.value == "unhealthy":
            status_code = 503
        elif system_health.status.value == "degraded":
            status_code = 200  # Still serving requests
        else:
            status_code = 200
        
        return response_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sentry_manager.capture_exception(e, tags={"endpoint": "healthz"})
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@health_router.get("/readyz")
async def readiness_check() -> Dict:
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to accept requests.
    """
    try:
        system_health = await health_checker.check_all(include_external=True)
        
        # Service is ready only if all components are healthy
        if system_health.status.value == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - unhealthy components detected"
            )
        
        response_data = system_health.to_dict()
        response_data["ready"] = True
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        sentry_manager.capture_exception(e, tags={"endpoint": "readyz"})
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Readiness check failed"
        )


@metrics_router.get("/metrics")
async def get_metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format.
    """
    try:
        # Update system metrics before returning
        metrics.update_system_metrics()
        
        # Get metrics in Prometheus format
        metrics_data = metrics.get_metrics()
        
        return PlainTextResponse(
            content=metrics_data,
            media_type=metrics.get_content_type()
        )
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        sentry_manager.capture_exception(e, tags={"endpoint": "metrics"})
        
        # Return empty metrics on error to avoid breaking monitoring
        return PlainTextResponse(
            content="# Error generating metrics\n",
            media_type=metrics.get_content_type()
        )


@api_router.get("/search")
async def semantic_search(
    q: str,
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
    project_id: Optional[str] = None,
    vehicle_make: Optional[str] = None,
    vehicle_model: Optional[str] = None,
    vehicle_year: Optional[int] = None,
    system: Optional[str] = None,
    limit: int = 10,
    strategy: str = "hybrid"
) -> Dict[str, Any]:
    """
    Semantic search across automotive documents.
    
    Performs intelligent search using hybrid dense/sparse retrieval
    with automotive domain knowledge and re-ranking.
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with log_context(correlation_id=correlation_id, user_id=current_user.user_id):
        try:
            logger.log_user_action("semantic_search", 
                                 query=q,
                                 project_id=project_id,
                                 strategy=strategy,
                                 limit=limit)
            
            # Build search filters
            filters = SearchFilter(
                project_id=project_id,
                system=system
            )
            
            # Add vehicle filter if provided
            if vehicle_make and vehicle_model:
                filters.vehicle = VehicleSignature(
                    make=vehicle_make,
                    model=vehicle_model,
                    year=vehicle_year or 2000
                )
            
            # Get search engine and perform search
            search_engine = get_search_engine()
            result = await search_engine.search(
                query=q,
                filters=filters,
                limit=limit,
                strategy=strategy
            )
            
            # Record success metrics
            duration = time.time() - start_time
            metrics.record_http_request("GET", "/v1/search", 200, duration)
            metrics.record_external_service_call("semantic_search", "search", "success", duration)
            
            logger.log_operation_success("semantic_search", duration, 
                                       query=q,
                                       hits=len(result.hits),
                                       strategy=strategy)
            
            return result.to_dict()
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics.record_http_request("GET", "/v1/search", 500, duration)
            metrics.record_error(type(e).__name__, "semantic_search", "error")
            
            # Log and capture error
            logger.log_operation_error("semantic_search", e, duration, query=q)
            sentry_manager.capture_exception(
                e,
                tags={"endpoint": "semantic_search", "user_id": current_user.user_id},
                extra={"query": q, "filters": str(filters)}
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            )


@api_router.get("/search/suggestions")
async def search_suggestions(
    q: str,
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
    project_id: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get search suggestions based on automotive domain knowledge.
    
    Returns suggested queries and related automotive concepts.
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with log_context(correlation_id=correlation_id, user_id=current_user.user_id):
        try:
            # Generate suggestions based on automotive ontology
            suggestions = []
            query_lower = q.lower()
            
            # Component-based suggestions
            if any(comp in query_lower for comp in ['relay', 'k']):
                suggestions.extend([
                    "starter relay location",
                    "fuel pump relay circuit", 
                    "relay wiring diagram",
                    "relay pin assignment"
                ])
            
            if any(comp in query_lower for comp in ['fuse', 'f']):
                suggestions.extend([
                    "fuse box diagram",
                    "fuse rating chart",
                    "blown fuse symptoms",
                    "fuse panel location"
                ])
            
            if any(comp in query_lower for comp in ['ecu', 'ecm', 'pcm']):
                suggestions.extend([
                    "ECU pin diagram",
                    "ECM connector wiring",
                    "PCM ground connections",
                    "ECU power supply"
                ])
            
            # System-based suggestions
            if 'start' in query_lower:
                suggestions.extend([
                    "starting system diagram",
                    "starter motor circuit",
                    "ignition switch wiring",
                    "no start diagnosis"
                ])
            
            if 'fuel' in query_lower:
                suggestions.extend([
                    "fuel pump circuit",
                    "fuel injection system",
                    "fuel pressure regulator",
                    "fuel pump relay"
                ])
            
            # Ground-related suggestions
            if any(term in query_lower for term in ['ground', 'gnd', 'earth']):
                suggestions.extend([
                    "ground point locations",
                    "chassis ground diagram",
                    "poor ground symptoms",
                    "ground resistance check"
                ])
            
            # Limit and deduplicate suggestions
            unique_suggestions = list(dict.fromkeys(suggestions))[:limit]
            
            duration = time.time() - start_time
            metrics.record_http_request("GET", "/v1/search/suggestions", 200, duration)
            
            logger.log_operation_success("search_suggestions", duration, 
                                       query=q,
                                       suggestions_count=len(unique_suggestions))
            
            return {
                "query": q,
                "suggestions": unique_suggestions,
                "processing_time": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_http_request("GET", "/v1/search/suggestions", 500, duration)
            metrics.record_error(type(e).__name__, "search_suggestions", "error")
            
            logger.log_operation_error("search_suggestions", e, duration, query=q)
            sentry_manager.capture_exception(e, tags={"endpoint": "search_suggestions"})
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate suggestions: {str(e)}",
            )


@api_router.post("/evaluation/run")
async def run_comprehensive_evaluation(
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
    benchmark_suites: Optional[List[str]] = None,
    search_strategies: Optional[List[str]] = None,
    include_learning: bool = True,
    include_performance: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of the automotive search and learning system.
    
    Executes benchmarks across multiple dimensions including search quality,
    learning system performance, and overall system metrics.
    """
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with log_context(correlation_id=correlation_id, user_id=current_user.user_id):
        try:
            logger.log_user_action("run_comprehensive_evaluation",
                                 benchmark_suites=benchmark_suites or ["all"],
                                 strategies=search_strategies or ["hybrid"])
            
            # Create evaluation configuration
            config = EvaluationConfig(
                benchmark_suites=benchmark_suites or ["component_identification", "wiring_analysis", "troubleshooting"],
                search_strategies=search_strategies or ["dense", "sparse", "hybrid"],
                include_learning_eval=include_learning,
                include_performance_monitoring=include_performance
            )
            
            # Get system components (in real implementation, would be injected)
            search_engine = get_search_engine()
            
            # Mock model registry and signal collector for evaluation
            from ..core.models import ModelRegistry
            from ..learning.signals import SignalCollector
            
            model_registry = ModelRegistry()
            signal_collector = SignalCollector()
            
            # Create and run evaluation
            evaluator = ComprehensiveEvaluationRunner(
                search_engine=search_engine,
                model_registry=model_registry,
                signal_collector=signal_collector,
                config=config
            )
            
            # Run the evaluation
            summary = await evaluator.run_comprehensive_evaluation()
            
            # Record success metrics
            duration = time.time() - start_time
            metrics.record_http_request("POST", "/v1/evaluation/run", 200, duration)
            metrics.record_external_service_call("evaluation", "comprehensive", "success", duration)
            
            logger.log_operation_success("run_comprehensive_evaluation", duration,
                                       benchmarks_run=summary.benchmarks_run,
                                       total_queries=summary.total_queries,
                                       overall_f1=summary.overall_scores.get("overall_f1", 0))
            
            return {
                "status": "completed",
                "summary": {
                    "duration": summary.total_duration,
                    "benchmarks_run": summary.benchmarks_run,
                    "total_queries": summary.total_queries,
                    "overall_scores": summary.overall_scores,
                    "timestamp": summary.timestamp.isoformat()
                },
                "strategy_comparison": summary.strategy_comparison,
                "learning_metrics": summary.learning_metrics if include_learning else {},
                "performance_metrics": summary.performance_metrics if include_performance else {}
            }
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics.record_http_request("POST", "/v1/evaluation/run", 500, duration)
            metrics.record_error(type(e).__name__, "evaluation", "error")
            
            # Log and capture error
            logger.log_operation_error("run_comprehensive_evaluation", e, duration)
            sentry_manager.capture_exception(
                e,
                tags={"endpoint": "run_comprehensive_evaluation", "user_id": current_user.user_id},
                extra={"config": str(config)}
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {str(e)}",
            )


@api_router.get("/evaluation/benchmarks")
async def list_available_benchmarks(
    http_request: Request,
    current_user: UserContext = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all available benchmark suites and their descriptions.
    """
    try:
        benchmarks = []
        for suite in AutomotiveBenchmarks.get_all_suites():
            benchmarks.append({
                "name": suite.name,
                "description": suite.description,
                "version": suite.version,
                "query_count": len(suite.queries),
                "automotive_focus": suite.automotive_focus
            })
        
        logger.log_user_action("list_benchmarks", benchmark_count=len(benchmarks))
        
        return {
            "benchmarks": benchmarks,
            "total_count": len(benchmarks)
        }
        
    except Exception as e:
        logger.log_operation_error("list_benchmarks", e, 0)
        sentry_manager.capture_exception(e, tags={"endpoint": "list_benchmarks"})
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list benchmarks: {str(e)}",
        )


@api_router.get("/evaluation/results/{benchmark_name}")
async def get_benchmark_results(
    benchmark_name: str,
    http_request: Request,
    current_user: UserContext = Depends(get_current_user),
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get historical results for a specific benchmark.
    """
    try:
        benchmark_manager = BenchmarkManager()
        historical_results = benchmark_manager.load_historical_results(benchmark_name)
        
        # Limit results and sort by timestamp
        recent_results = sorted(
            historical_results,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limit]
        
        logger.log_user_action("get_benchmark_results", 
                             benchmark_name=benchmark_name,
                             results_found=len(recent_results))
        
        return {
            "benchmark_name": benchmark_name,
            "results": recent_results,
            "count": len(recent_results)
        }
        
    except Exception as e:
        logger.log_operation_error("get_benchmark_results", e, 0, benchmark_name=benchmark_name)
        sentry_manager.capture_exception(e, tags={"endpoint": "get_benchmark_results"})
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get benchmark results: {str(e)}",
        )