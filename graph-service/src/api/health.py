"""Health check API endpoints"""

from fastapi import APIRouter, Depends
from ..services.graph_service import GraphService
from ..utils.neo4j_utils import Neo4jClient

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check(graph_service: GraphService = Depends()):
    return await graph_service.health_check()

@router.get("/database")
async def database_health(neo4j_client: Neo4jClient = Depends()):
    try:
        await neo4j_client.run("RETURN 1 as test")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@router.get("/stats")
async def database_stats(graph_service: GraphService = Depends()):
    return await graph_service.get_database_stats()

@router.get("/ready")
async def readiness_check():
    return {"status": "ready", "service": "graph-service"}

@router.get("/live")
async def liveness_check():
    return {"status": "alive", "service": "graph-service"}