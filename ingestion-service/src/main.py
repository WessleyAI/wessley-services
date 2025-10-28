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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    # Startup
    print("Starting ingestion service...")
    
    # Initialize Redis queue connection
    await queue_manager.connect()
    
    # TODO: Initialize connections to Supabase, Neo4j, Qdrant
    # TODO: Set up Prometheus metrics
    # TODO: Initialize Sentry
    
    yield
    
    # Shutdown
    print("Shutting down ingestion service...")
    
    # Close Redis connection
    await queue_manager.disconnect()
    
    # TODO: Close database connections
    # TODO: Clean up resources


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    """
    app = FastAPI(
        title="Wessley Ingestion Service",
        description="Data ingestion service for processing electrical schematics and technical documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if os.getenv("APP_ENV") != "prod" else None,
        redoc_url="/redoc" if os.getenv("APP_ENV") != "prod" else None,
    )
    
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