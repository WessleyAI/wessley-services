"""
Hybrid semantic search for automotive electronics.

This module implements multi-modal search combining dense embeddings,
sparse lexical search, and cross-encoder re-ranking for automotive documents.
"""
import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

# Optional dependencies
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from ..core.logging import StructuredLogger
from ..core.metrics import metrics
from ..persist.qdrant import QdrantPersistence, SemanticSearchResult
from .embed import AutomotiveEmbedder, get_embedder
from .ontology import VehicleSignature, automotive_ontology

logger = StructuredLogger(__name__)


@dataclass
class SearchFilter:
    """Search filters for automotive queries."""
    project_id: Optional[str] = None
    vehicle: Optional[VehicleSignature] = None
    system: Optional[str] = None
    component_types: List[str] = field(default_factory=list)
    page_range: Optional[Tuple[int, int]] = None
    confidence_threshold: float = 0.0
    include_symbolic: bool = True


@dataclass
class SearchHit:
    """Individual search result hit."""
    id: str
    text: str
    score: float
    rank: int
    source: str  # "dense", "sparse", or "reranked"
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    component_ids: List[str] = field(default_factory=list)
    net_names: List[str] = field(default_factory=list)
    system: Optional[str] = None
    vehicle: Optional[VehicleSignature] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox,
            "component_ids": self.component_ids,
            "net_names": self.net_names,
            "system": self.system,
            "metadata": self.metadata
        }
        
        if self.vehicle:
            result["vehicle"] = {
                "make": self.vehicle.make,
                "model": self.vehicle.model,
                "year": self.vehicle.year,
                "market": self.vehicle.market
            }
        
        if self.explanation:
            result["explanation"] = self.explanation
            
        return result


@dataclass
class SearchResult:
    """Complete search result with hits and metadata."""
    hits: List[SearchHit]
    total_hits: int
    query: str
    filters: SearchFilter
    processing_time: float
    search_strategy: str
    model_versions: Dict[str, str] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "total_hits": self.total_hits,
            "query": self.query,
            "processing_time": self.processing_time,
            "search_strategy": self.search_strategy,
            "model_versions": self.model_versions,
            "debug_info": self.debug_info if logger.logger.level <= logging.DEBUG else {}
        }


class SparseSearchProvider(ABC):
    """Abstract sparse search provider."""
    
    @abstractmethod
    async def search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform sparse search."""
        pass


class PostgresFullTextSearch(SparseSearchProvider):
    """PostgreSQL full-text search for automotive terms."""
    
    def __init__(self, connection_string: str):
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available")
        
        self.connection_string = connection_string
        self.connection = None
    
    async def connect(self):
        """Establish database connection."""
        if not self.connection:
            self.connection = psycopg2.connect(self.connection_string)
    
    async def search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform PostgreSQL full-text search."""
        if not self.connection:
            await self.connect()
        
        # Build tsvector query
        query_parts = []
        terms = query.split()
        
        for term in terms:
            # Handle automotive-specific terms
            if term.upper().startswith(('F', 'K', 'R', 'C', 'U', 'Q')):  # Component IDs
                query_parts.append(f"{term}:*")
            elif any(unit in term.lower() for unit in ['ohm', 'amp', 'volt', 'farad']):
                query_parts.append(f"{term}")
            else:
                query_parts.append(f"{term}:*")
        
        ts_query = " & ".join(query_parts)
        
        # Build SQL query
        sql = """
        SELECT id, text, ts_rank(ts_lex, to_tsquery(%s)) as score,
               page, bbox, component_ids, net_names, system, vehicle
        FROM semantic_chunks 
        WHERE ts_lex @@ to_tsquery(%s)
        """
        
        conditions = []
        params = [ts_query, ts_query]
        
        if filters.project_id:
            conditions.append("project_id = %s")
            params.append(filters.project_id)
        
        if filters.vehicle:
            conditions.append("vehicle->>'make' = %s AND vehicle->>'model' = %s")
            params.extend([filters.vehicle.make, filters.vehicle.model])
        
        if conditions:
            sql += " AND " + " AND ".join(conditions)
        
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                hits = []
                for i, row in enumerate(rows):
                    hit = SearchHit(
                        id=row['id'],
                        text=row['text'],
                        score=float(row['score']),
                        rank=i + 1,
                        source="sparse",
                        page=row.get('page'),
                        bbox=row.get('bbox'),
                        component_ids=row.get('component_ids', []),
                        net_names=row.get('net_names', []),
                        system=row.get('system'),
                        metadata={"ts_rank": row['score']}
                    )
                    
                    if row.get('vehicle'):
                        vehicle_data = row['vehicle']
                        hit.vehicle = VehicleSignature(
                            make=vehicle_data.get('make'),
                            model=vehicle_data.get('model'),
                            year=vehicle_data.get('year'),
                            market=vehicle_data.get('market')
                        )
                    
                    hits.append(hit)
                
                return hits
                
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []


class MockSparseSearch(SparseSearchProvider):
    """Mock sparse search for testing."""
    
    def __init__(self):
        self.mock_docs = [
            {
                "id": "doc1",
                "text": "Starter relay K1 controls engine cranking circuit via IG1 signal",
                "component_ids": ["K1"],
                "net_names": ["IG1"],
                "system": "starting"
            },
            {
                "id": "doc2", 
                "text": "Fuel pump relay K2 energized by ECM signal on pin 15",
                "component_ids": ["K2", "ECM"],
                "net_names": [],
                "system": "fuel"
            },
            {
                "id": "doc3",
                "text": "Fuse F10 30A protects starter motor circuit and related components",
                "component_ids": ["F10"],
                "net_names": [],
                "system": "starting"
            }
        ]
    
    async def search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform mock sparse search."""
        query_lower = query.lower()
        results = []
        
        for i, doc in enumerate(self.mock_docs[:limit]):
            # Simple keyword matching
            score = 0.0
            for word in query_lower.split():
                if word in doc["text"].lower():
                    score += 1.0
            
            if score > 0:
                hit = SearchHit(
                    id=doc["id"],
                    text=doc["text"],
                    score=score / len(query_lower.split()),
                    rank=len(results) + 1,
                    source="sparse",
                    component_ids=doc["component_ids"],
                    net_names=doc["net_names"],
                    system=doc["system"],
                    metadata={"keyword_matches": score}
                )
                results.append(hit)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results


class CrossEncoderReranker:
    """Cross-encoder for search result re-ranking."""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.model = None
        
        if model_name != "mock":
            logger.warning(f"Cross-encoder {model_name} not implemented, using mock")
    
    async def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        """Re-rank search hits using cross-encoder."""
        if not hits:
            return hits
        
        # Mock re-ranking: boost automotive-specific terms
        automotive_terms = ['relay', 'fuse', 'ecu', 'starter', 'fuel', 'ignition', 'ground']
        query_lower = query.lower()
        
        reranked_hits = []
        for hit in hits:
            new_score = hit.score
            
            # Boost if query contains automotive terms
            for term in automotive_terms:
                if term in query_lower and term in hit.text.lower():
                    new_score *= 1.2
            
            # Boost if component IDs match
            for comp_id in hit.component_ids:
                if comp_id.lower() in query_lower:
                    new_score *= 1.5
            
            # Boost if net names match
            for net_name in hit.net_names:
                if net_name.lower() in query_lower:
                    new_score *= 1.3
            
            reranked_hit = SearchHit(
                id=hit.id,
                text=hit.text,
                score=new_score,
                rank=hit.rank,
                source="reranked",
                page=hit.page,
                bbox=hit.bbox,
                component_ids=hit.component_ids,
                net_names=hit.net_names,
                system=hit.system,
                vehicle=hit.vehicle,
                metadata={**hit.metadata, "original_score": hit.score, "rerank_boost": new_score / hit.score},
                explanation=self._generate_explanation(query, hit, new_score / hit.score)
            )
            reranked_hits.append(reranked_hit)
        
        # Sort by new scores and update ranks
        reranked_hits.sort(key=lambda x: x.score, reverse=True)
        for i, hit in enumerate(reranked_hits[:top_k]):
            hit.rank = i + 1
        
        return reranked_hits[:top_k]
    
    def _generate_explanation(self, query: str, hit: SearchHit, boost_factor: float) -> str:
        """Generate explanation for ranking decision."""
        if boost_factor <= 1.1:
            return "Standard relevance match"
        
        explanations = []
        
        if any(comp_id.lower() in query.lower() for comp_id in hit.component_ids):
            explanations.append("exact component match")
        
        if any(net.lower() in query.lower() for net in hit.net_names):
            explanations.append("exact net match")
        
        automotive_terms = ['relay', 'fuse', 'ecu', 'starter', 'fuel']
        matched_terms = [term for term in automotive_terms 
                        if term in query.lower() and term in hit.text.lower()]
        if matched_terms:
            explanations.append(f"automotive terms: {', '.join(matched_terms)}")
        
        if explanations:
            return f"Boosted for: {'; '.join(explanations)}"
        else:
            return f"General relevance boost ({boost_factor:.2f}x)"


class HybridAutomotiveSearch:
    """Hybrid search combining dense, sparse, and re-ranking."""
    
    def __init__(self,
                 qdrant_client: Optional[QdrantPersistence] = None,
                 sparse_provider: Optional[SparseSearchProvider] = None,
                 reranker: Optional[CrossEncoderReranker] = None,
                 embedder: Optional[AutomotiveEmbedder] = None):
        """
        Initialize hybrid search system.
        
        Args:
            qdrant_client: Qdrant persistence for dense search
            sparse_provider: Sparse search provider (PostgreSQL FTS)
            reranker: Cross-encoder for re-ranking
            embedder: Automotive embedder
        """
        self.qdrant = qdrant_client
        self.sparse = sparse_provider or MockSparseSearch()
        self.reranker = reranker or CrossEncoderReranker()
        self.embedder = embedder or get_embedder()
        
        # Search configuration
        self.topk_dense = 50
        self.topk_sparse = 50
        self.topk_final = 20
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        
        logger.info("HybridAutomotiveSearch initialized")
    
    async def search(self, 
                    query: str,
                    filters: SearchFilter = None,
                    limit: int = 10,
                    strategy: str = "hybrid") -> SearchResult:
        """
        Perform hybrid automotive search.
        
        Args:
            query: Search query
            filters: Search filters
            limit: Maximum results to return
            strategy: Search strategy ("dense", "sparse", "hybrid")
            
        Returns:
            SearchResult with ranked hits
        """
        start_time = time.time()
        
        if filters is None:
            filters = SearchFilter()
        
        try:
            if strategy == "dense":
                hits = await self._dense_search(query, filters, limit)
            elif strategy == "sparse":
                hits = await self._sparse_search(query, filters, limit)
            else:  # hybrid
                hits = await self._hybrid_search(query, filters, limit)
            
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_external_service_call(
                "semantic_search",
                strategy,
                "success",
                processing_time
            )
            
            logger.info(f"Search completed",
                       query=query,
                       strategy=strategy,
                       hits=len(hits),
                       processing_time=processing_time)
            
            return SearchResult(
                hits=hits,
                total_hits=len(hits),
                query=query,
                filters=filters,
                processing_time=processing_time,
                search_strategy=strategy,
                model_versions={
                    "embedder": self.embedder.semantic_provider.get_model_name(),
                    "reranker": self.reranker.model_name
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(f"Search failed: {e}",
                        query=query,
                        strategy=strategy,
                        processing_time=processing_time)
            
            metrics.record_external_service_call(
                "semantic_search",
                strategy,
                "error",
                processing_time
            )
            
            # Return empty result on error
            return SearchResult(
                hits=[],
                total_hits=0,
                query=query,
                filters=filters,
                processing_time=processing_time,
                search_strategy=strategy,
                debug_info={"error": str(e)}
            )
    
    async def _dense_search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform dense semantic search."""
        if not self.qdrant:
            logger.warning("Qdrant not available for dense search")
            return []
        
        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query, "semantic")
        
        # Perform vector search
        try:
            results = await self.qdrant.semantic_search(
                query=query,
                project_id=uuid.UUID(filters.project_id) if filters.project_id else None,
                limit=limit,
                score_threshold=filters.confidence_threshold
            )
            
            hits = []
            for i, result in enumerate(results):
                hit = SearchHit(
                    id=result.chunk.id,
                    text=result.chunk.text,
                    score=result.score,
                    rank=i + 1,
                    source="dense",
                    page=result.chunk.page,
                    metadata={
                        "distance": result.distance,
                        "chunk_type": result.chunk.chunk_type
                    }
                )
                
                # Extract automotive context from metadata
                if result.chunk.metadata:
                    hit.component_ids = result.chunk.metadata.get("component_ids", [])
                    hit.net_names = result.chunk.metadata.get("net_names", [])
                    hit.system = result.chunk.metadata.get("system")
                
                hits.append(hit)
            
            return hits
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    async def _sparse_search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform sparse lexical search."""
        return await self.sparse.search(query, filters, limit)
    
    async def _hybrid_search(self, query: str, filters: SearchFilter, limit: int) -> List[SearchHit]:
        """Perform hybrid search with fusion and re-ranking."""
        # Run dense and sparse searches concurrently
        dense_task = self._dense_search(query, filters, self.topk_dense)
        sparse_task = self._sparse_search(query, filters, self.topk_sparse)
        
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        
        # Combine and deduplicate hits
        combined_hits = self._fuse_results(dense_hits, sparse_hits)
        
        # Re-rank top results
        if len(combined_hits) > self.topk_final:
            combined_hits = await self.reranker.rerank(query, combined_hits, self.topk_final)
        
        return combined_hits[:limit]
    
    def _fuse_results(self, dense_hits: List[SearchHit], sparse_hits: List[SearchHit]) -> List[SearchHit]:
        """Fuse dense and sparse results using reciprocal rank fusion."""
        # Create ID to hit mapping
        hit_map = {}
        
        # Add dense hits with weighted scores
        for hit in dense_hits:
            hit.score = hit.score * self.dense_weight
            hit_map[hit.id] = hit
        
        # Add sparse hits, combining if already present
        for hit in sparse_hits:
            weighted_score = hit.score * self.sparse_weight
            
            if hit.id in hit_map:
                # Combine scores for duplicate hits
                existing_hit = hit_map[hit.id]
                existing_hit.score = existing_hit.score + weighted_score
                existing_hit.source = "hybrid"
                # Merge metadata
                existing_hit.metadata.update(hit.metadata)
            else:
                hit.score = weighted_score
                hit_map[hit.id] = hit
        
        # Sort by combined score
        fused_hits = list(hit_map.values())
        fused_hits.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, hit in enumerate(fused_hits):
            hit.rank = i + 1
        
        return fused_hits
    
    def update_configuration(self, config: Dict[str, Any]):
        """Update search configuration."""
        self.topk_dense = config.get("topk_dense", self.topk_dense)
        self.topk_sparse = config.get("topk_sparse", self.topk_sparse)
        self.topk_final = config.get("topk_final", self.topk_final)
        self.dense_weight = config.get("dense_weight", self.dense_weight)
        self.sparse_weight = config.get("sparse_weight", self.sparse_weight)
        
        logger.info("Search configuration updated", **config)


# Global search instance
_global_search: Optional[HybridAutomotiveSearch] = None

def get_search_engine() -> HybridAutomotiveSearch:
    """Get global automotive search engine."""
    global _global_search
    if _global_search is None:
        _global_search = HybridAutomotiveSearch()
    return _global_search

def set_search_engine(search_engine: HybridAutomotiveSearch):
    """Set global automotive search engine."""
    global _global_search
    _global_search = search_engine