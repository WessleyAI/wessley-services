"""
Vector store service using Qdrant for high-performance semantic search.
Manages collections, indexing, and vector similarity search operations.
"""

import asyncio
import uuid
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, CreateCollection, PointStruct, UpdateResult,
    SearchRequest, SearchParams, Filter, FieldCondition, 
    MatchValue, MatchAny, Range, GeoBoundingBox
)

from ..config.settings import get_settings
from ..core.logging import get_logger
from ..models.search_models import CollectionName, SearchFilter
from ..models.component_models import ElectricalComponent

settings = get_settings()
logger = get_logger(__name__)


class VectorStoreService:
    """High-performance vector storage and search using Qdrant."""
    
    def __init__(self):
        self.client = None
        self.collections_initialized = False
        
    async def initialize(self) -> None:
        """Initialize Qdrant client and create collections."""
        logger.logger.info("Initializing vector store service")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.vector_search_timeout
        )
        
        # Test connection
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            logger.logger.info("Qdrant connection established")
        except Exception as e:
            logger.logger.error("Failed to connect to Qdrant", error=str(e))
            raise
        
        # Initialize collections
        await self._initialize_collections()
        self.collections_initialized = True
        
    async def close(self) -> None:
        """Close Qdrant client connection."""
        if self.client:
            self.client.close()
    
    async def _initialize_collections(self) -> None:
        """Create Qdrant collections if they don't exist."""
        collections_config = {
            CollectionName.COMPONENTS: {
                "vector_size": 768,  # sentence-transformers output size
                "distance": models.Distance.COSINE
            },
            CollectionName.DOCUMENTATION: {
                "vector_size": 768,
                "distance": models.Distance.COSINE
            },
            CollectionName.KG_ENTITIES: {
                "vector_size": 768,
                "distance": models.Distance.COSINE
            }
        }
        
        existing_collections = await asyncio.get_event_loop().run_in_executor(
            None, self.client.get_collections
        )
        existing_names = {col.name for col in existing_collections.collections}
        
        for collection_name, config in collections_config.items():
            if collection_name.value not in existing_names:
                await self._create_collection(collection_name, config)
                logger.logger.info("Created collection", collection=collection_name.value)
            else:
                logger.logger.info("Collection already exists", collection=collection_name.value)
    
    async def _create_collection(self, collection_name: CollectionName, config: Dict[str, Any]) -> None:
        """Create a new Qdrant collection."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.client.create_collection,
            collection_name.value,
            VectorParams(
                size=config["vector_size"],
                distance=config["distance"]
            )
        )
    
    async def upsert_points(self, collection_name: CollectionName, 
                           points: List[Dict[str, Any]]) -> bool:
        """Insert or update points in a collection."""
        if not points:
            return True
        
        try:
            # Convert to Qdrant point structures
            qdrant_points = []
            for point in points:
                qdrant_point = PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {})
                )
                qdrant_points.append(qdrant_point)
            
            # Upsert points
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.upsert,
                collection_name.value,
                qdrant_points
            )
            
            success = result.status == models.UpdateStatus.COMPLETED
            
            logger.logger.info(
                "Upserted points",
                collection=collection_name.value,
                points_count=len(points),
                success=success
            )
            
            return success
            
        except Exception as e:
            logger.logger.error(
                "Failed to upsert points",
                collection=collection_name.value,
                error=str(e)
            )
            return False
    
    async def search_vectors(self, collection_name: CollectionName,
                           query_vector: List[float],
                           limit: int = 20,
                           score_threshold: Optional[float] = None,
                           filters: Optional[List[SearchFilter]] = None,
                           with_payload: bool = True) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            # Build Qdrant filter
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None
            
            # Perform search
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.search,
                collection_name.value,
                query_vector,
                limit,
                qdrant_filter,
                with_payload,
                score_threshold
            )
            
            # Convert results to our format
            results = []
            for result in search_results:
                result_dict = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload if with_payload else {}
                }
                results.append(result_dict)
            
            logger.log_search_operation(
                query="vector_search",
                collection=collection_name.value,
                results_count=len(results),
                response_time_ms=0  # TODO: Add timing
            )
            
            return results
            
        except Exception as e:
            logger.logger.error(
                "Vector search failed",
                collection=collection_name.value,
                error=str(e)
            )
            return []
    
    async def search_multiple_collections(self, 
                                        collections: List[CollectionName],
                                        query_vector: List[float],
                                        limit_per_collection: int = 20,
                                        score_threshold: Optional[float] = None,
                                        filters: Optional[List[SearchFilter]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections simultaneously."""
        tasks = []
        
        for collection in collections:
            task = self.search_vectors(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit_per_collection,
                score_threshold=score_threshold,
                filters=filters
            )
            tasks.append(task)
        
        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {}
        for collection, result in zip(collections, results):
            if isinstance(result, Exception):
                logger.logger.error(
                    "Collection search failed", 
                    collection=collection.value,
                    error=str(result)
                )
                combined_results[collection.value] = []
            else:
                combined_results[collection.value] = result
        
        return combined_results
    
    def _build_qdrant_filter(self, filters: List[SearchFilter]) -> Optional[Filter]:
        """Convert SearchFilter objects to Qdrant filter format."""
        if not filters:
            return None
        
        must_conditions = []
        
        for search_filter in filters:
            condition = self._build_filter_condition(search_filter)
            if condition:
                must_conditions.append(condition)
        
        if must_conditions:
            return Filter(must=must_conditions)
        
        return None
    
    def _build_filter_condition(self, search_filter: SearchFilter) -> Optional[FieldCondition]:
        """Build a single filter condition."""
        field = search_filter.field
        operator = search_filter.operator
        value = search_filter.value
        
        try:
            if operator == "eq":
                return FieldCondition(key=field, match=MatchValue(value=value))
            elif operator == "in":
                if isinstance(value, list):
                    return FieldCondition(key=field, match=MatchAny(any=value))
            elif operator == "gt":
                return FieldCondition(key=field, range=Range(gt=value))
            elif operator == "lt":
                return FieldCondition(key=field, range=Range(lt=value))
            elif operator == "gte":
                return FieldCondition(key=field, range=Range(gte=value))
            elif operator == "lte":
                return FieldCondition(key=field, range=Range(lte=value))
            else:
                logger.logger.warning("Unsupported filter operator", operator=operator)
        except Exception as e:
            logger.logger.warning("Failed to build filter condition", error=str(e))
        
        return None
    
    async def get_point_by_id(self, collection_name: CollectionName, 
                            point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific point by ID."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.retrieve,
                collection_name.value,
                [point_id],
                True  # with_payload
            )
            
            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
        except Exception as e:
            logger.logger.error(
                "Failed to retrieve point",
                collection=collection_name.value,
                point_id=point_id,
                error=str(e)
            )
        
        return None
    
    async def delete_points(self, collection_name: CollectionName, 
                          point_ids: List[str]) -> bool:
        """Delete points from a collection."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.delete,
                collection_name.value,
                point_ids
            )
            
            success = result.status == models.UpdateStatus.COMPLETED
            
            logger.logger.info(
                "Deleted points",
                collection=collection_name.value,
                points_count=len(point_ids),
                success=success
            )
            
            return success
            
        except Exception as e:
            logger.logger.error(
                "Failed to delete points",
                collection=collection_name.value,
                error=str(e)
            )
            return False
    
    async def count_points(self, collection_name: CollectionName,
                         filters: Optional[List[SearchFilter]] = None) -> int:
        """Count points in a collection with optional filters."""
        try:
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.count,
                collection_name.value,
                qdrant_filter
            )
            
            return result.count
            
        except Exception as e:
            logger.logger.error(
                "Failed to count points",
                collection=collection_name.value,
                error=str(e)
            )
            return 0
    
    async def scroll_points(self, collection_name: CollectionName,
                          limit: int = 100,
                          offset: Optional[str] = None,
                          filters: Optional[List[SearchFilter]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Scroll through points in a collection."""
        try:
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.scroll,
                collection_name.value,
                qdrant_filter,
                limit,
                offset
            )
            
            points = []
            for point in result[0]:
                points.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
            next_offset = result[1]
            
            return points, next_offset
            
        except Exception as e:
            logger.logger.error(
                "Failed to scroll points",
                collection=collection_name.value,
                error=str(e)
            )
            return [], None
    
    async def get_collection_info(self, collection_name: CollectionName) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_collection,
                collection_name.value
            )
            
            return {
                "name": collection_name.value,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
            
        except Exception as e:
            logger.logger.error(
                "Failed to get collection info",
                collection=collection_name.value,
                error=str(e)
            )
            return {}
    
    async def recreate_collection(self, collection_name: CollectionName) -> bool:
        """Delete and recreate a collection (use with caution)."""
        try:
            # Delete existing collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.delete_collection,
                collection_name.value
            )
            
            # Recreate collection
            config = {
                "vector_size": 768,
                "distance": models.Distance.COSINE
            }
            await self._create_collection(collection_name, config)
            
            logger.logger.info("Recreated collection", collection=collection_name.value)
            return True
            
        except Exception as e:
            logger.logger.error(
                "Failed to recreate collection",
                collection=collection_name.value,
                error=str(e)
            )
            return False
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get vector store service statistics."""
        stats = {
            "collections": {},
            "total_points": 0,
            "total_vectors": 0
        }
        
        for collection in CollectionName:
            try:
                info = await self.get_collection_info(collection)
                stats["collections"][collection.value] = info
                stats["total_points"] += info.get("points_count", 0)
                stats["total_vectors"] += info.get("vectors_count", 0)
            except Exception as e:
                logger.logger.warning(
                    "Failed to get collection stats",
                    collection=collection.value,
                    error=str(e)
                )
        
        return stats
    
    def build_component_point(self, component: ElectricalComponent, 
                            embedding: List[float]) -> Dict[str, Any]:
        """Build a Qdrant point from an electrical component."""
        return {
            "id": component.component_id,
            "vector": embedding,
            "payload": {
                "component_id": component.component_id,
                "vehicle_signature": component.vehicle_signature,
                "canonical_id": component.canonical_id,
                "code_id": component.code_id,
                "component_type": component.component_type.value,
                "node_type": component.node_type,
                "anchor_zone": component.anchor_zone.value,
                "description": component.description,
                "function": component.function,
                "purpose": component.purpose,
                "manufacturer": component.manufacturer,
                "part_number": component.part_number,
                "model_number": component.model_number,
                "categories": component.categories,
                "tags": component.tags,
                "keywords": component.keywords,
                "voltage_rating": component.specifications.voltage_rating,
                "current_rating": component.specifications.current_rating,
                "power_rating": component.specifications.power_rating,
                "position_3d": component.position_3d,
                "indexed_at": component.metadata.indexed_at.isoformat()
            }
        }
    
    def build_documentation_point(self, doc_id: str, title: str, content: str,
                                embedding: List[float], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build a Qdrant point from documentation."""
        return {
            "id": doc_id,
            "vector": embedding,
            "payload": {
                "document_id": doc_id,
                "title": title,
                "content_chunk": content,
                "document_type": metadata.get("document_type", "unknown"),
                "page_number": metadata.get("page_number"),
                "vehicle_models": metadata.get("vehicle_models", []),
                "component_types": metadata.get("component_types", []),
                "keywords": metadata.get("keywords", []),
                "language": metadata.get("language", "en"),
                "indexed_at": datetime.utcnow().isoformat()
            }
        }