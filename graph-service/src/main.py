"""FastAPI application for graph service"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from typing import Dict, Any

# Import all API routers
from .api.health import router as health_router
from .api.vehicles import router as vehicles_router
from .api.components import router as components_router
from .api.circuits import router as circuits_router
from .api.analysis import router as analysis_router
from .api.import_export import router as import_export_router

# Import utilities and dependencies
from .utils.neo4j_utils import Neo4jConnectionManager
from .config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application settings
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Graph Service...")
    
    # Initialize Neo4j connection
    connection_manager = Neo4jConnectionManager()
    connection_manager.initialize(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database
    )
    
    # Test connection
    try:
        client = await connection_manager.get_client()
        health = await client.health_check()
        logger.info(f"Neo4j connection: {health['status']}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise
    
    # Create constraints and indexes
    await create_database_constraints(client)
    
    logger.info("Graph Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Graph Service...")
    await connection_manager.close()
    logger.info("Graph Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Vehicle Electrical System Graph Service",
    description="A graph database service for managing vehicle electrical system data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(health_router)
app.include_router(vehicles_router)
app.include_router(components_router)
app.include_router(circuits_router)
app.include_router(analysis_router)
app.include_router(import_export_router)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Vehicle Electrical System Graph Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/info")
async def service_info():
    """Get detailed service information"""
    from .utils.performance_utils import performance_monitor
    
    return {
        "service": "graph-service",
        "version": "1.0.0",
        "environment": settings.environment,
        "database": {
            "type": "neo4j",
            "uri": settings.neo4j_uri,
            "database": settings.neo4j_database
        },
        "performance": performance_monitor.get_all_stats(),
        "features": {
            "vehicle_management": True,
            "component_analysis": True,
            "circuit_analysis": True,
            "spatial_analysis": True,
            "data_import": True,
            "performance_monitoring": True
        }
    }

async def create_database_constraints(client):
    """Create necessary database constraints and indexes"""
    try:
        # Unique constraints
        await client.create_constraint("Component", "id", "UNIQUE")
        await client.create_constraint("Component", "vehicle_signature", "EXISTS")
        await client.create_constraint("Circuit", "id", "UNIQUE")
        await client.create_constraint("Circuit", "vehicle_signature", "EXISTS")
        await client.create_constraint("Vehicle", "signature", "UNIQUE")
        await client.create_constraint("Zone", "id", "UNIQUE")
        await client.create_constraint("Connector", "id", "UNIQUE")
        
        # Performance indexes
        await client.create_index("Component", "type")
        await client.create_index("Component", "name")
        await client.create_index("Circuit", "circuit_type")
        await client.create_index("Vehicle", "make")
        await client.create_index("Vehicle", "model")
        await client.create_index("Vehicle", "year")
        
        logger.info("Database constraints and indexes created successfully")
    except Exception as e:
        logger.warning(f"Some constraints/indexes may already exist: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": "2024-01-01T00:00:00Z"  # Should use actual timestamp
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": "2024-01-01T00:00:00Z"  # Should use actual timestamp
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.environment == "development" else False,
        log_level="info"
    )