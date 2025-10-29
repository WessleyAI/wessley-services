"""
Qdrant vector database integration for semantic search and text embeddings.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import uuid
import hashlib

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from ..core.schemas import TextSpan, Component, Net


@dataclass
class TextChunk:
    """Represents a text chunk for embedding."""
    id: str
    text: str
    chunk_type: str  # "text_span", "component_description", "net_description"
    project_id: str
    page: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SemanticSearchResult:
    """Result from semantic search query."""
    chunk: TextChunk
    score: float
    distance: float


@dataclass
class EmbeddingStats:
    """Statistics about embedding operations."""
    chunks_processed: int
    embeddings_created: int
    embedding_dimension: int
    processing_time_ms: float
    errors: List[str]


class EmbeddingProvider:
    """Abstract base for embedding providers."""
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-ada-002."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.dimension = 1536  # text-embedding-ada-002 dimension
        
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.api_key
        self.logger = logging.getLogger(__name__)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []
        
        try:
            # OpenAI recommends batching up to 2048 texts at once
            batch_size = 100  # Conservative batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await openai.Embedding.acreate(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if len(texts) > batch_size:
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self.dimension


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate random embeddings for testing."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            embeddings.append(embedding)
        
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension


class QdrantPersistence:
    """
    Handles persistence and semantic search using Qdrant vector database.
    
    Collections:
    - wessley_docs: Main collection for document chunks
    - component_descriptions: Component-specific embeddings
    - net_descriptions: Network description embeddings
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = "wessley_docs",
        embedding_provider: EmbeddingProvider = None
    ):
        """
        Initialize Qdrant client and embedding provider.
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Collection name for storing embeddings
            embedding_provider: Provider for generating embeddings
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        # Initialize embedding provider
        if embedding_provider:
            self.embedding_provider = embedding_provider
        elif os.getenv("OPENAI_API_KEY"):
            self.embedding_provider = OpenAIEmbeddingProvider()
        else:
            self.embedding_provider = MockEmbeddingProvider()
        
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """
        Connect to Qdrant and initialize collections.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize client
            if self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(url=self.url)
            
            # Test connection
            collections = self.client.get_collections()
            self.logger.info(f"Connected to Qdrant at {self.url}")
            
            # Initialize collection
            await self._initialize_collection()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    async def _initialize_collection(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name not in existing_collections:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_provider.get_dimension(),
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise
    
    async def store_text_chunks(
        self,
        project_id: uuid.UUID,
        text_spans: List[TextSpan],
        components: List[Component] = None,
        nets: List[Net] = None
    ) -> EmbeddingStats:
        """
        Store text chunks with embeddings in Qdrant.
        
        Args:
            project_id: Project identifier
            text_spans: List of text spans from OCR
            components: List of components for description generation
            nets: List of nets for description generation
            
        Returns:
            Statistics about the embedding operation
        """
        start_time = datetime.now()
        stats = EmbeddingStats(
            chunks_processed=0,
            embeddings_created=0,
            embedding_dimension=self.embedding_provider.get_dimension(),
            processing_time_ms=0.0,
            errors=[]
        )
        
        try:
            # 1. Create chunks from text spans
            text_chunks = await self._create_text_chunks(project_id, text_spans)
            
            # 2. Create component description chunks
            if components:
                component_chunks = await self._create_component_chunks(project_id, components)
                text_chunks.extend(component_chunks)
            
            # 3. Create net description chunks
            if nets:
                net_chunks = await self._create_net_chunks(project_id, nets)
                text_chunks.extend(net_chunks)
            
            stats.chunks_processed = len(text_chunks)
            
            if not text_chunks:
                return stats
            
            # 4. Generate embeddings
            texts = [chunk.text for chunk in text_chunks]
            embeddings = await self.embedding_provider.embed_texts(texts)
            
            # 5. Store in Qdrant
            points = []
            for chunk, embedding in zip(text_chunks, embeddings):
                chunk.embedding = embedding
                
                point = models.PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "chunk_type": chunk.chunk_type,
                        "project_id": chunk.project_id,
                        "page": chunk.page,
                        "metadata": chunk.metadata,
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upsert points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
                stats.embeddings_created += len(batch)
            
            # Calculate execution time
            end_time = datetime.now()
            stats.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Stored {stats.embeddings_created} embeddings in {stats.processing_time_ms:.2f}ms"
            )
            
        except Exception as e:
            error_msg = f"Failed to store embeddings: {e}"
            stats.errors.append(error_msg)
            self.logger.error(error_msg)
            raise
        
        return stats
    
    async def _create_text_chunks(self, project_id: uuid.UUID, text_spans: List[TextSpan]) -> List[TextChunk]:
        """Create chunks from text spans."""
        chunks = []
        
        for i, text_span in enumerate(text_spans):
            chunk_id = f"{project_id}_text_{text_span.page}_{i}"
            
            chunk = TextChunk(
                id=chunk_id,
                text=text_span.text,
                chunk_type="text_span",
                project_id=str(project_id),
                page=text_span.page,
                metadata={
                    "bbox": text_span.bbox,
                    "confidence": text_span.confidence,
                    "engine": text_span.engine.value if hasattr(text_span.engine, 'value') else str(text_span.engine),
                    "rotation": getattr(text_span, 'rotation', 0)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_component_chunks(self, project_id: uuid.UUID, components: List[Component]) -> List[TextChunk]:
        """Create searchable chunks from component descriptions."""
        chunks = []
        
        for component in components:
            # Create descriptive text for the component
            description_parts = [
                f"Component {component.id}",
                f"Type: {component.type.value if hasattr(component.type, 'value') else str(component.type)}"
            ]
            
            if component.value:
                description_parts.append(f"Value: {component.value}")
            
            if component.pins:
                pin_names = [pin.name for pin in component.pins]
                description_parts.append(f"Pins: {', '.join(pin_names)}")
            
            description_parts.append(f"Page: {component.page}")
            description_parts.append(f"Confidence: {component.confidence:.2f}")
            
            description_text = ". ".join(description_parts)
            
            chunk_id = f"{project_id}_component_{component.id}"
            
            chunk = TextChunk(
                id=chunk_id,
                text=description_text,
                chunk_type="component_description",
                project_id=str(project_id),
                page=component.page,
                metadata={
                    "component_id": component.id,
                    "component_type": component.type.value if hasattr(component.type, 'value') else str(component.type),
                    "component_value": component.value,
                    "bbox": component.bbox,
                    "confidence": component.confidence,
                    "pin_count": len(component.pins) if component.pins else 0
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_net_chunks(self, project_id: uuid.UUID, nets: List[Net]) -> List[TextChunk]:
        """Create searchable chunks from net descriptions."""
        chunks = []
        
        for net in nets:
            # Create descriptive text for the net
            description_parts = [
                f"Electrical net {net.name}",
                f"Connections: {len(net.connections)} components"
            ]
            
            if net.connections:
                connection_descriptions = []
                for conn in net.connections[:5]:  # Limit to first 5 connections
                    connection_descriptions.append(f"{conn.component_id} pin {conn.pin}")
                
                description_parts.append(f"Connected to: {', '.join(connection_descriptions)}")
                
                if len(net.connections) > 5:
                    description_parts.append(f"and {len(net.connections) - 5} more")
            
            description_parts.append(f"Pages: {', '.join(map(str, net.page_spans))}")
            description_parts.append(f"Confidence: {net.confidence:.2f}")
            
            description_text = ". ".join(description_parts)
            
            chunk_id = f"{project_id}_net_{net.name.replace(' ', '_')}"
            
            chunk = TextChunk(
                id=chunk_id,
                text=description_text,
                chunk_type="net_description",
                project_id=str(project_id),
                page=net.page_spans[0] if net.page_spans else 1,
                metadata={
                    "net_name": net.name,
                    "connection_count": len(net.connections),
                    "page_spans": net.page_spans,
                    "confidence": net.confidence,
                    "component_ids": [conn.component_id for conn in net.connections]
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def semantic_search(
        self,
        query: str,
        project_id: uuid.UUID = None,
        chunk_types: List[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[SemanticSearchResult]:
        """
        Perform semantic search on stored embeddings.
        
        Args:
            query: Search query text
            project_id: Filter by project ID
            chunk_types: Filter by chunk types
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with scores
        """
        try:
            # Generate embedding for query
            query_embeddings = await self.embedding_provider.embed_texts([query])
            query_vector = query_embeddings[0]
            
            # Build filter conditions
            filter_conditions = []
            
            if project_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=str(project_id))
                    )
                )
            
            if chunk_types:
                filter_conditions.append(
                    models.FieldCondition(
                        key="chunk_type",
                        match=models.MatchAny(any=chunk_types)
                    )
                )
            
            # Create filter
            search_filter = None
            if filter_conditions:
                if len(filter_conditions) == 1:
                    search_filter = models.Filter(must=[filter_conditions[0]])
                else:
                    search_filter = models.Filter(must=filter_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert to result objects
            results = []
            for result in search_results:
                chunk = TextChunk(
                    id=result.id,
                    text=result.payload["text"],
                    chunk_type=result.payload["chunk_type"],
                    project_id=result.payload["project_id"],
                    page=result.payload["page"],
                    metadata=result.payload.get("metadata", {})
                )
                
                search_result = SemanticSearchResult(
                    chunk=chunk,
                    score=result.score,
                    distance=1.0 - result.score  # Convert similarity to distance
                )
                results.append(search_result)
            
            self.logger.info(f"Semantic search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise
    
    async def delete_project_embeddings(self, project_id: uuid.UUID) -> bool:
        """Delete all embeddings for a project."""
        try:
            # Delete by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="project_id",
                                match=models.MatchValue(value=str(project_id))
                            )
                        ]
                    )
                )
            )
            
            self.logger.info(f"Deleted embeddings for project {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete project embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.value
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def similarity_search_by_component(
        self,
        component_id: str,
        project_id: uuid.UUID,
        limit: int = 5
    ) -> List[SemanticSearchResult]:
        """Find similar components by embedding similarity."""
        try:
            # Get the component's embedding
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=str(project_id))
                        ),
                        models.FieldCondition(
                            key="metadata.component_id",
                            match=models.MatchValue(value=component_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if not search_results[0]:
                return []
            
            component_point = search_results[0][0]
            
            # Search for similar embeddings
            similar_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=component_point.vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_type",
                            match=models.MatchValue(value="component_description")
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="metadata.component_id",
                            match=models.MatchValue(value=component_id)
                        )
                    ]
                ),
                limit=limit
            )
            
            # Convert to result objects
            results = []
            for result in similar_results:
                chunk = TextChunk(
                    id=result.id,
                    text=result.payload["text"],
                    chunk_type=result.payload["chunk_type"],
                    project_id=result.payload["project_id"],
                    page=result.payload["page"],
                    metadata=result.payload.get("metadata", {})
                )
                
                search_result = SemanticSearchResult(
                    chunk=chunk,
                    score=result.score,
                    distance=1.0 - result.score
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Component similarity search failed: {e}")
            return []


# Convenience function for creating Qdrant persistence instance
def create_qdrant_persistence(use_openai: bool = True, **kwargs) -> QdrantPersistence:
    """Create Qdrant persistence instance with appropriate embedding provider."""
    if use_openai and os.getenv("OPENAI_API_KEY"):
        embedding_provider = OpenAIEmbeddingProvider()
    else:
        embedding_provider = MockEmbeddingProvider()
    
    return QdrantPersistence(embedding_provider=embedding_provider, **kwargs)