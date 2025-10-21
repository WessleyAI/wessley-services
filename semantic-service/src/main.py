"""
FastAPI application entry point for the semantic search service.
Configures middleware, routes, and service initialization.
"""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config.settings import get_settings
from .core.logging import get_logger
from .api.search_routes import router as search_router
from .services.search_service import SearchService
from .services.vector_store import VectorStoreService
from .services.embedding import EmbeddingService

settings = get_settings()
logger = get_logger(__name__)

# Global service instances
search_service: SearchService = None
vector_store: VectorStoreService = None
embedding_service: EmbeddingService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown.
    """
    global search_service, vector_store, embedding_service
    
    logger.logger.info("Starting semantic search service")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        
        vector_store = VectorStoreService()
        await vector_store.initialize()
        
        search_service = SearchService()
        await search_service.initialize()
        
        # Store in app state for access in routes
        app.state.search_service = search_service
        app.state.vector_store = vector_store
        app.state.embedding_service = embedding_service
        
        logger.logger.info("Service initialization completed")
        
        yield
        
    except Exception as e:
        logger.logger.error("Service initialization failed", error=str(e))
        raise
    finally:
        # Cleanup on shutdown
        logger.logger.info("Shutting down semantic search service")
        
        if search_service:
            await search_service.close()
        if vector_store:
            await vector_store.close()
        if embedding_service:
            await embedding_service.close()
        
        logger.logger.info("Service shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Semantic Search Service",
    description="High-performance semantic search for electrical components and documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(search_router)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return JSONResponse(content={
        "service": "semantic-search-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/search/health"
    })


@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return JSONResponse(content={"status": "ok", "message": "pong"})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )