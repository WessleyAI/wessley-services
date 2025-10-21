"""
FastAPI routes for semantic search operations.
Provides endpoints for component search, chat enhancement, and analytics.
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..services.search_service import SearchService
from ..models.search_models import (
    UniversalSearchQuery, SearchResponse, ChatEnhancementRequest, 
    ChatEnhancementResponse, AnalyticsQuery, SearchAnalytics,
    CollectionName, SearchType, QueryIntent
)
from ..models.component_models import (
    ComponentSearchQuery, ComponentSearchResponse,
    ComponentRecommendationResponse, ComponentIndexRequest,
    ComponentIndexStatus
)
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])

# Dependency to get search service instance
async def get_search_service() -> SearchService:
    """Get initialized search service instance."""
    search_service = SearchService()
    await search_service.initialize()
    return search_service


@router.post("/universal", response_model=SearchResponse)
async def universal_search(
    query: UniversalSearchQuery,
    search_service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    """
    Perform universal semantic search across multiple collections.
    Supports vector, keyword, hybrid, and semantic search types.
    """
    start_time = time.time()
    
    try:
        logger.logger.info(
            "Universal search request",
            query=query.query,
            search_type=query.search_type.value,
            collections=[c.value for c in query.collections],
            filters_count=len(query.filters)
        )
        
        # Execute search
        response = await search_service.universal_search(query)
        
        # Log performance metrics
        search_time = (time.time() - start_time) * 1000
        logger.log_search_operation(
            query=query.query,
            collection="universal",
            results_count=len(response.results),
            response_time_ms=search_time
        )
        
        return response
        
    except ValueError as e:
        logger.logger.error("Invalid search query", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.logger.error("Search operation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Search operation failed")


@router.post("/components", response_model=ComponentSearchResponse)
async def search_components(
    query: ComponentSearchQuery,
    search_service: SearchService = Depends(get_search_service)
) -> ComponentSearchResponse:
    """
    Search specifically for electrical components with component-specific filters.
    """
    start_time = time.time()
    
    try:
        logger.logger.info(
            "Component search request",
            query=query.query,
            vehicle_signature=query.vehicle_signature,
            component_types=[ct.value for ct in query.component_types] if query.component_types else None
        )
        
        # Execute component search
        response = await search_service.search_components(query)
        
        # Log metrics
        search_time = (time.time() - start_time) * 1000
        logger.log_search_operation(
            query=query.query,
            collection="components",
            results_count=len(response.results),
            response_time_ms=search_time
        )
        
        return response
        
    except ValueError as e:
        logger.logger.error("Invalid component search query", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.logger.error("Component search failed", error=str(e))
        raise HTTPException(status_code=500, detail="Component search failed")


@router.post("/chat/enhance", response_model=ChatEnhancementResponse)
async def enhance_chat_context(
    request: ChatEnhancementRequest,
    search_service: SearchService = Depends(get_search_service)
) -> ChatEnhancementResponse:
    """
    Enhance chat context with relevant components and documentation.
    Used to provide context for AI chat responses.
    """
    start_time = time.time()
    
    try:
        logger.logger.info(
            "Chat enhancement request",
            user_query=request.user_query,
            vehicle_signature=request.vehicle_signature,
            history_length=len(request.chat_history)
        )
        
        # Generate enhanced context
        response = await search_service.enhance_chat_context(request)
        
        # Log metrics
        enhancement_time = (time.time() - start_time) * 1000
        logger.logger.info(
            "Chat enhancement completed",
            components_found=len(response.relevant_components),
            docs_found=len(response.relevant_documentation),
            enhancement_time_ms=enhancement_time
        )
        
        return response
        
    except ValueError as e:
        logger.logger.error("Invalid chat enhancement request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.logger.error("Chat enhancement failed", error=str(e))
        raise HTTPException(status_code=500, detail="Chat enhancement failed")


@router.get("/recommendations/{component_id}", response_model=ComponentRecommendationResponse)
async def get_component_recommendations(
    component_id: str,
    limit: int = Query(default=5, ge=1, le=20),
    similarity_threshold: float = Query(default=0.7, ge=0, le=1),
    search_service: SearchService = Depends(get_search_service)
) -> ComponentRecommendationResponse:
    """
    Get component recommendations based on similarity to a given component.
    """
    start_time = time.time()
    
    try:
        logger.logger.info(
            "Component recommendations request",
            component_id=component_id,
            limit=limit,
            threshold=similarity_threshold
        )
        
        # Generate recommendations
        response = await search_service.get_component_recommendations(
            component_id=component_id,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Log metrics
        recommendation_time = (time.time() - start_time) * 1000
        logger.logger.info(
            "Recommendations generated",
            component_id=component_id,
            recommendations_count=len(response.recommendations),
            time_ms=recommendation_time
        )
        
        return response
        
    except ValueError as e:
        logger.logger.error("Invalid recommendation request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.logger.error("Recommendation generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Recommendation generation failed")


@router.get("/suggest")
async def get_search_suggestions(
    query: str = Query(..., min_length=2),
    limit: int = Query(default=5, ge=1, le=20),
    collections: List[CollectionName] = Query(default=[CollectionName.COMPONENTS]),
    search_service: SearchService = Depends(get_search_service)
) -> JSONResponse:
    """
    Get search suggestions based on partial query.
    """
    try:
        logger.logger.info(
            "Search suggestions request",
            partial_query=query,
            collections=[c.value for c in collections]
        )
        
        # Generate suggestions
        suggestions = await search_service.get_search_suggestions(
            partial_query=query,
            collections=collections,
            limit=limit
        )
        
        return JSONResponse(content={"suggestions": suggestions})
        
    except Exception as e:
        logger.logger.error("Suggestion generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Suggestion generation failed")


@router.post("/index/components")
async def index_components(
    request: ComponentIndexRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service)
) -> JSONResponse:
    """
    Start component indexing operation for a vehicle.
    Runs as a background task.
    """
    try:
        logger.logger.info(
            "Component indexing request",
            vehicle_signature=request.vehicle_signature,
            force_reindex=request.force_reindex,
            batch_size=request.batch_size
        )
        
        # Start indexing as background task
        background_tasks.add_task(
            search_service.index_vehicle_components,
            request
        )
        
        return JSONResponse(content={
            "message": "Component indexing started",
            "vehicle_signature": request.vehicle_signature,
            "status": "initiated"
        })
        
    except Exception as e:
        logger.logger.error("Failed to start component indexing", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start indexing")


@router.get("/index/status/{vehicle_signature}", response_model=ComponentIndexStatus)
async def get_indexing_status(
    vehicle_signature: str,
    search_service: SearchService = Depends(get_search_service)
) -> ComponentIndexStatus:
    """
    Get status of component indexing operation.
    """
    try:
        status = await search_service.get_indexing_status(vehicle_signature)
        
        if not status:
            raise HTTPException(
                status_code=404, 
                detail=f"No indexing status found for vehicle: {vehicle_signature}"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.logger.error("Failed to get indexing status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get indexing status")


@router.get("/analytics", response_model=SearchAnalytics)
async def get_search_analytics(
    query: AnalyticsQuery = Depends(),
    search_service: SearchService = Depends(get_search_service)
) -> SearchAnalytics:
    """
    Get search analytics and performance metrics.
    """
    try:
        logger.logger.info(
            "Analytics request",
            start_date=query.start_date,
            end_date=query.end_date,
            user_id=query.user_id
        )
        
        analytics = await search_service.get_search_analytics(query)
        
        return analytics
        
    except Exception as e:
        logger.logger.error("Failed to get analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get analytics")


@router.get("/health")
async def health_check(
    search_service: SearchService = Depends(get_search_service)
) -> JSONResponse:
    """
    Health check endpoint for service monitoring.
    """
    try:
        # Check service health
        health_status = await search_service.get_service_health()
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.delete("/cache/clear")
async def clear_cache(
    collection: Optional[CollectionName] = Query(None),
    search_service: SearchService = Depends(get_search_service)
) -> JSONResponse:
    """
    Clear embedding cache for performance troubleshooting.
    """
    try:
        logger.logger.info("Cache clear request", collection=collection.value if collection else "all")
        
        cleared_count = await search_service.clear_cache(collection)
        
        return JSONResponse(content={
            "message": "Cache cleared successfully",
            "cleared_items": cleared_count,
            "collection": collection.value if collection else "all"
        })
        
    except Exception as e:
        logger.logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: CollectionName,
    search_service: SearchService = Depends(get_search_service)
) -> JSONResponse:
    """
    Get statistics for a specific collection.
    """
    try:
        stats = await search_service.get_collection_stats(collection_name)
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.logger.error("Failed to get collection stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get collection stats")