"""
Documentation semantic search and indexing service.
Handles technical manuals, repair guides, and service documentation.
"""

import asyncio
import uuid
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

from ..config.settings import get_settings
from ..core.logging import get_logger
from ..models.search_models import CollectionName, SearchFilter, SearchResult
from .vector_store import VectorStoreService
from .embedding import EmbeddingService

settings = get_settings()
logger = get_logger(__name__)


class DocumentationType:
    """Standard documentation types."""
    SERVICE_MANUAL = "service_manual"
    WIRING_DIAGRAM = "wiring_diagram"
    REPAIR_GUIDE = "repair_guide"
    PARTS_CATALOG = "parts_catalog"
    TECHNICAL_BULLETIN = "technical_bulletin"
    TROUBLESHOOTING = "troubleshooting"
    INSTALLATION_GUIDE = "installation_guide"


class DocumentationService:
    """Service for processing and searching technical documentation."""
    
    def __init__(self):
        self.vector_store = None
        self.embedding_service = None
        self.chunk_size = 512
        self.chunk_overlap = 64
        
    async def initialize(self) -> None:
        """Initialize documentation service dependencies."""
        logger.logger.info("Initializing documentation service")
        
        self.vector_store = VectorStoreService()
        await self.vector_store.initialize()
        
        self.embedding_service = EmbeddingService()
        await self.embedding_service.initialize()
        
        logger.logger.info("Documentation service initialized")
    
    async def close(self) -> None:
        """Close service connections."""
        if self.vector_store:
            await self.vector_store.close()
        if self.embedding_service:
            await self.embedding_service.close()
    
    async def index_document(self, 
                           document_content: str,
                           title: str,
                           document_type: str = DocumentationType.SERVICE_MANUAL,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Index a technical document for semantic search.
        Splits document into chunks and creates vector embeddings.
        """
        if not metadata:
            metadata = {}
        
        start_time = datetime.utcnow()
        document_id = str(uuid.uuid4())
        
        try:
            logger.logger.info(
                "Starting document indexing",
                document_id=document_id,
                title=title,
                document_type=document_type,
                content_length=len(document_content)
            )
            
            # Pre-process document content
            processed_content = self._preprocess_document(document_content)
            
            # Split into searchable chunks
            chunks = self._chunk_document(processed_content, title, metadata)
            
            # Extract technical entities and keywords
            entities = self._extract_technical_entities(processed_content)
            metadata.update({"extracted_entities": entities})
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings_batch(
                chunk_texts, batch_size=32
            )
            
            # Create vector points for indexing
            vector_points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                point = self.embedding_service.build_documentation_point(
                    doc_id=chunk_id,
                    title=title,
                    content=chunk["content"],
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "document_id": document_id,
                        "document_type": document_type,
                        "chunk_index": i,
                        "chunk_start": chunk["start"],
                        "chunk_end": chunk["end"],
                        "section_title": chunk.get("section_title"),
                        "page_number": chunk.get("page_number")
                    }
                )
                vector_points.append(point)
            
            # Index in vector store
            success = await self.vector_store.upsert_points(
                CollectionName.DOCUMENTATION, vector_points
            )
            
            if not success:
                raise RuntimeError("Failed to index document chunks")
            
            indexing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                "document_id": document_id,
                "title": title,
                "document_type": document_type,
                "chunks_created": len(chunks),
                "entities_extracted": len(entities),
                "indexing_time_ms": indexing_time,
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            logger.logger.info(
                "Document indexing completed",
                **result
            )
            
            return result
            
        except Exception as e:
            logger.logger.error(
                "Document indexing failed",
                document_id=document_id,
                error=str(e)
            )
            raise
    
    async def search_documentation(self,
                                 query: str,
                                 document_types: List[str] = None,
                                 vehicle_models: List[str] = None,
                                 component_types: List[str] = None,
                                 limit: int = 20,
                                 similarity_threshold: float = 0.7) -> List[SearchResult]:
        """
        Search technical documentation using semantic similarity.
        """
        try:
            logger.logger.info(
                "Documentation search request",
                query=query,
                document_types=document_types,
                vehicle_models=vehicle_models,
                component_types=component_types
            )
            
            # Build search filters
            filters = []
            if document_types:
                filters.append(SearchFilter(
                    field="document_type",
                    operator="in",
                    value=document_types
                ))
            
            if vehicle_models:
                filters.append(SearchFilter(
                    field="vehicle_models",
                    operator="in",
                    value=vehicle_models
                ))
            
            if component_types:
                filters.append(SearchFilter(
                    field="component_types",
                    operator="in",
                    value=component_types
                ))
            
            # Enhance query with technical context
            enhanced_query = await self._enhance_documentation_query(query)
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(enhanced_query)
            
            # Search vector store
            search_results = await self.vector_store.search_vectors(
                collection_name=CollectionName.DOCUMENTATION,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold,
                filters=filters
            )
            
            # Process and rank results
            processed_results = []
            for result in search_results:
                search_result = self._process_documentation_result(result, query)
                if search_result:
                    processed_results.append(search_result)
            
            # Re-rank results based on relevance
            ranked_results = self._rerank_documentation_results(processed_results, query)
            
            logger.logger.info(
                "Documentation search completed",
                query=query,
                results_found=len(ranked_results)
            )
            
            return ranked_results
            
        except Exception as e:
            logger.logger.error(
                "Documentation search failed",
                query=query,
                error=str(e)
            )
            return []
    
    async def get_document_context(self,
                                 document_id: str,
                                 context_window: int = 3) -> Dict[str, Any]:
        """
        Get surrounding context for a document chunk.
        Useful for showing more complete information.
        """
        try:
            # Find all chunks for this document
            filters = [SearchFilter(
                field="document_id",
                operator="eq",
                value=document_id
            )]
            
            chunks, _ = await self.vector_store.scroll_points(
                collection_name=CollectionName.DOCUMENTATION,
                filters=filters,
                limit=1000
            )
            
            # Sort chunks by index
            sorted_chunks = sorted(chunks, key=lambda c: c["payload"].get("chunk_index", 0))
            
            # Build context
            context = {
                "document_id": document_id,
                "total_chunks": len(sorted_chunks),
                "chunks": []
            }
            
            for chunk in sorted_chunks:
                chunk_data = {
                    "chunk_id": chunk["id"],
                    "content": chunk["payload"].get("content_chunk", ""),
                    "section_title": chunk["payload"].get("section_title"),
                    "page_number": chunk["payload"].get("page_number"),
                    "chunk_index": chunk["payload"].get("chunk_index")
                }
                context["chunks"].append(chunk_data)
            
            return context
            
        except Exception as e:
            logger.logger.error(
                "Failed to get document context",
                document_id=document_id,
                error=str(e)
            )
            return {}
    
    def _preprocess_document(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Normalize line breaks
        content = re.sub(r'\r\n|\r', '\n', content)
        
        # Remove special characters that might interfere with processing
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\\]', ' ', content)
        
        # Normalize technical terminology
        content = self._normalize_technical_terms(content)
        
        return content.strip()
    
    def _normalize_technical_terms(self, content: str) -> str:
        """Normalize common technical terms for better matching."""
        # Voltage normalization
        content = re.sub(r'(\d+)\s*V(?:olt)?s?', r'\1V', content, flags=re.IGNORECASE)
        
        # Current normalization
        content = re.sub(r'(\d+)\s*A(?:mp)?s?', r'\1A', content, flags=re.IGNORECASE)
        
        # Power normalization
        content = re.sub(r'(\d+)\s*W(?:att)?s?', r'\1W', content, flags=re.IGNORECASE)
        
        # Wire gauge normalization
        content = re.sub(r'(\d+)\s*(?:mm²|AWG)', r'\1mm²', content, flags=re.IGNORECASE)
        
        return content
    
    def _chunk_document(self, content: str, title: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks for indexing."""
        chunks = []
        words = content.split()
        
        # Estimate words per chunk (rough approximation)
        words_per_chunk = self.chunk_size // 4  # Assume ~4 chars per word
        overlap_words = self.chunk_overlap // 4
        
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(words):
            # Calculate chunk boundaries
            end_pos = min(current_pos + words_per_chunk, len(words))
            
            # Extract chunk content
            chunk_words = words[current_pos:end_pos]
            chunk_content = ' '.join(chunk_words)
            
            # Try to find section headers in the chunk
            section_title = self._extract_section_title(chunk_content)
            
            chunk = {
                "content": f"{title}. {chunk_content}",  # Prepend title for context
                "start": current_pos,
                "end": end_pos,
                "section_title": section_title,
                "page_number": self._estimate_page_number(current_pos, len(words))
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            current_pos = max(current_pos + words_per_chunk - overlap_words, current_pos + 1)
            chunk_index += 1
            
            # Prevent infinite loops
            if chunk_index > 1000:
                logger.logger.warning("Document chunking stopped at 1000 chunks")
                break
        
        return chunks
    
    def _extract_section_title(self, chunk_content: str) -> Optional[str]:
        """Extract section title from chunk content."""
        lines = chunk_content.split('\n')
        
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            
            # Look for patterns that indicate section headers
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^\d+\.', line) or
                 re.match(r'^[A-Z][^.]{10,}$', line))):
                return line
        
        return None
    
    def _estimate_page_number(self, word_position: int, total_words: int) -> int:
        """Estimate page number based on word position."""
        # Assume ~300 words per page (rough estimate)
        words_per_page = 300
        return (word_position // words_per_page) + 1
    
    def _extract_technical_entities(self, content: str) -> List[str]:
        """Extract technical entities and keywords from content."""
        entities = []
        
        # Component names (basic pattern matching)
        component_patterns = [
            r'\b(?:relay|fuse|sensor|actuator|ecu|battery|alternator|starter)\b',
            r'\b(?:connector|wire|harness|ground|circuit|switch)\b',
            r'\b(?:module|controller|pump|motor|injector|coil)\b'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities.extend([match.lower() for match in matches])
        
        # Part numbers (basic pattern)
        part_numbers = re.findall(r'\b[A-Z0-9]{5,15}\b', content)
        entities.extend(part_numbers)
        
        # Remove duplicates and return
        return list(set(entities))
    
    async def _enhance_documentation_query(self, query: str) -> str:
        """Enhance query with technical context and synonyms."""
        # Add common technical synonyms
        enhancements = []
        
        # Component synonyms
        synonyms = {
            "relay": "switch control",
            "fuse": "protection circuit breaker",
            "sensor": "detector measurement",
            "ecu": "computer controller module",
            "harness": "wiring cable assembly"
        }
        
        query_lower = query.lower()
        for term, synonym in synonyms.items():
            if term in query_lower:
                enhancements.append(synonym)
        
        enhanced_query = query
        if enhancements:
            enhanced_query += " " + " ".join(enhancements)
        
        return enhanced_query
    
    def _process_documentation_result(self, result: Dict[str, Any], query: str) -> Optional[SearchResult]:
        """Process vector search result into SearchResult format."""
        try:
            payload = result.get("payload", {})
            
            # Extract highlights (simple word matching)
            content = payload.get("content_chunk", "")
            highlights = self._extract_highlights(content, query)
            
            search_result = SearchResult(
                id=result["id"],
                collection=CollectionName.DOCUMENTATION,
                content=payload,
                score=result["score"],
                rank=0,  # Will be set during ranking
                match_type="semantic",
                matched_fields=["content_chunk", "title"],
                highlights={"content": highlights},
                explanation=f"Found relevant documentation about: {query}",
                confidence=result["score"],
                source_metadata={
                    "document_type": payload.get("document_type"),
                    "page_number": payload.get("page_number"),
                    "section_title": payload.get("section_title")
                }
            )
            
            return search_result
            
        except Exception as e:
            logger.logger.warning("Failed to process documentation result", error=str(e))
            return None
    
    def _extract_highlights(self, content: str, query: str) -> List[str]:
        """Extract highlighted snippets from content based on query."""
        highlights = []
        query_words = query.lower().split()
        
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            
            # Check if sentence contains query words
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > 0:
                # Highlight matching words
                highlighted = sentence
                for word in query_words:
                    if word in sentence_lower:
                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                        highlighted = pattern.sub(f"**{word}**", highlighted)
                
                highlights.append(highlighted)
        
        return highlights[:3]  # Return top 3 highlights
    
    def _rerank_documentation_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Re-rank documentation results based on additional relevance factors."""
        
        for i, result in enumerate(results):
            # Apply ranking position
            result.rank = i + 1
            
            # Boost certain document types for specific queries
            doc_type = result.source_metadata.get("document_type", "")
            
            boost_factor = 1.0
            query_lower = query.lower()
            
            if "troubleshoot" in query_lower and doc_type == DocumentationType.TROUBLESHOOTING:
                boost_factor = 1.2
            elif "wiring" in query_lower and doc_type == DocumentationType.WIRING_DIAGRAM:
                boost_factor = 1.2
            elif "repair" in query_lower and doc_type == DocumentationType.REPAIR_GUIDE:
                boost_factor = 1.1
            elif "install" in query_lower and doc_type == DocumentationType.INSTALLATION_GUIDE:
                boost_factor = 1.1
            
            # Apply boost
            result.score = min(1.0, result.score * boost_factor)
        
        # Sort by boosted score
        return sorted(results, key=lambda r: r.score, reverse=True)