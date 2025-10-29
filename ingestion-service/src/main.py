"""
Main FastAPI application for the ingestion service.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .api.routes import api_router, health_router
from .workers.queue import queue_manager
from .core.logging import setup_logging, StructuredLogger

# Setup ultra-verbose logging
setup_logging(level='DEBUG', enable_structured=True)
logger = StructuredLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    import os
    from .core.sentry import sentry_manager
    
    # Startup
    logger.info("🚀 Starting Wessley ingestion service", 
               service_name="ingestion-service",
               version="0.1.0",
               environment=os.getenv("APP_ENV", "unknown"))
    
    try:
        logger.info("📡 Initializing Redis queue connection",
                   redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
        await queue_manager.connect()
        logger.info("✅ Redis queue connection established successfully")
        
        # Initialize Sentry if configured
        sentry_dsn = os.getenv("SENTRY_DSN")
        if sentry_dsn:
            logger.info("🔍 Initializing Sentry error tracking", sentry_dsn=sentry_dsn[:20] + "...")
            sentry_manager.initialize(
                dsn=sentry_dsn,
                environment=os.getenv("APP_ENV", "development"),
                sample_rate=1.0,
                traces_sample_rate=1.0,
                enable_profiling=True,
                profiles_sample_rate=1.0
            )
            logger.info("✅ Sentry initialization completed")
        else:
            logger.warning("⚠️ No Sentry DSN configured - error tracking disabled")
        
        logger.info("🔄 Service startup completed successfully",
                   startup_phase="complete")
        
    except Exception as e:
        logger.exception("❌ Service startup failed", 
                        startup_phase="failed",
                        error_type=type(e).__name__,
                        error_message=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ingestion service",
               shutdown_phase="starting")
    
    try:
        logger.info("📡 Closing Redis connection")
        await queue_manager.disconnect()
        logger.info("✅ Redis connection closed successfully")
        
        logger.info("🔄 Service shutdown completed successfully",
                   shutdown_phase="complete")
    except Exception as e:
        logger.exception("❌ Service shutdown failed",
                        shutdown_phase="failed", 
                        error_type=type(e).__name__,
                        error_message=str(e))


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    """
    logger.info("🏗️ Creating FastAPI application", 
               app_title="Wessley Ingestion Service",
               app_version="0.1.0")
    
    app = FastAPI(
        title="Wessley Ingestion Service",
        description="Data ingestion service for processing electrical schematics and technical documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if os.getenv("APP_ENV") != "prod" else None,
        redoc_url="/redoc" if os.getenv("APP_ENV") != "prod" else None,
    )
    
    # Add ultra-verbose request/response logging middleware
    @app.middleware("http")
    async def ultra_verbose_logging_middleware(request, call_next):
        import time
        import json
        from .core.logging import generate_correlation_id, log_context
        
        correlation_id = generate_correlation_id()
        start_time = time.time()
        
        # Log incoming request with full details
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if body:
                try:
                    request_body = json.loads(body.decode())
                except:
                    request_body = body.decode()[:1000] + "..." if len(body) > 1000 else body.decode()
        
        with log_context(correlation_id=correlation_id):
            logger.info("📥 Incoming HTTP request", 
                       http_method=request.method,
                       http_url=str(request.url),
                       http_path=request.url.path,
                       http_query_params=dict(request.query_params),
                       http_headers=dict(request.headers),
                       http_client_host=request.client.host if request.client else None,
                       http_user_agent=request.headers.get("user-agent"),
                       http_content_type=request.headers.get("content-type"),
                       http_content_length=request.headers.get("content-length"),
                       http_request_body=request_body,
                       request_start_time=start_time)
            
            # Process request
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                # Log response details
                logger.info("📤 HTTP response sent",
                           http_status_code=response.status_code,
                           http_response_headers=dict(response.headers),
                           http_duration_ms=round(duration * 1000, 2),
                           http_duration_seconds=round(duration, 4),
                           response_status="success" if response.status_code < 400 else "error")
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                logger.exception("💥 HTTP request failed with exception",
                               http_duration_ms=round(duration * 1000, 2),
                               http_duration_seconds=round(duration, 4),
                               exception_type=type(e).__name__,
                               exception_message=str(e))
                raise
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # TODO: Configure allowed hosts for production
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure CORS origins for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router, tags=["health"])
    app.include_router(api_router, tags=["ingestion"])
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("APP_ENV") == "dev",
        log_level="info",
    )