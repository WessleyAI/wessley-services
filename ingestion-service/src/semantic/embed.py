"""
Multi-head embedding generation for automotive semantic search.

This module provides dense semantic embeddings and symbolic context embeddings
for automotive electronics documents, optimized for domain-specific retrieval.
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

# Optional dependencies with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.logging import StructuredLogger
from ..core.metrics import metrics
from .ontology import automotive_ontology, AutomotiveComponent, ElectricalNet

logger = StructuredLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    semantic_embedding: np.ndarray
    symbolic_embedding: Optional[np.ndarray] = None
    processing_time: float = 0.0
    model_version: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name/version."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence transformer based embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully, dimension: {self.get_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode([text], convert_to_numpy=True)[0]
        )
        return embedding
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        return [emb for emb in embeddings]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if not self.model:
            return 384  # Default for many sentence transformers
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get model name/version."""
        return self.model_name


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = f"mock-{dimension}d"
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate mock embedding for a single text."""
        # Use hash for consistency
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        return np.random.normal(0, 1, self.dimension).astype(np.float32)
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get model name/version."""
        return self.model_name


class AutomotiveEmbedder:
    """Multi-head embedder for automotive electronics documents."""
    
    def __init__(self, 
                 semantic_provider: Optional[EmbeddingProvider] = None,
                 symbolic_provider: Optional[EmbeddingProvider] = None,
                 enable_caching: bool = True):
        """
        Initialize automotive embedder.
        
        Args:
            semantic_provider: Provider for semantic text embeddings
            symbolic_provider: Provider for symbolic context embeddings (optional)
            enable_caching: Whether to cache embeddings
        """
        self.semantic_provider = semantic_provider or self._create_default_provider()
        self.symbolic_provider = symbolic_provider
        self.enable_caching = enable_caching
        self.cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized AutomotiveEmbedder with semantic: {self.semantic_provider.get_model_name()}")
        if self.symbolic_provider:
            logger.info(f"Symbolic provider: {self.symbolic_provider.get_model_name()}")
    
    def _create_default_provider(self) -> EmbeddingProvider:
        """Create default embedding provider."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try automotive-specific model first, fallback to general
                return SentenceTransformerProvider("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
        
        logger.info("Using mock embeddings for development")
        return MockEmbeddingProvider(384)
    
    async def embed_document_chunk(self, 
                                 text: str,
                                 components: List[AutomotiveComponent] = None,
                                 nets: List[ElectricalNet] = None,
                                 page: Optional[int] = None,
                                 vehicle_context: Optional[str] = None) -> EmbeddingResult:
        """
        Generate embeddings for a document chunk with automotive context.
        
        Args:
            text: Main text content
            components: Associated automotive components
            nets: Associated electrical nets
            page: Page number
            vehicle_context: Vehicle identification context
            
        Returns:
            EmbeddingResult with semantic and optional symbolic embeddings
        """
        start_time = time.time()
        
        try:
            # Generate semantic embedding from text
            semantic_emb = await self._embed_with_cache(
                self.semantic_provider, 
                text, 
                cache_key=f"sem_{hash(text)}"
            )
            
            # Generate symbolic context embedding if provider available
            symbolic_emb = None
            if self.symbolic_provider and (components or nets):
                symbolic_context = self._build_symbolic_context(components, nets, vehicle_context)
                if symbolic_context:
                    symbolic_emb = await self._embed_with_cache(
                        self.symbolic_provider,
                        symbolic_context,
                        cache_key=f"sym_{hash(symbolic_context)}"
                    )
            
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_external_service_call(
                "embeddings", 
                "embed_chunk", 
                "success", 
                processing_time
            )
            
            return EmbeddingResult(
                semantic_embedding=semantic_emb,
                symbolic_embedding=symbolic_emb,
                processing_time=processing_time,
                model_version=self.semantic_provider.get_model_name(),
                metadata={
                    "text_length": len(text),
                    "component_count": len(components) if components else 0,
                    "net_count": len(nets) if nets else 0,
                    "page": page,
                    "has_symbolic": symbolic_emb is not None
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Embedding generation failed: {e}", 
                        text_length=len(text),
                        processing_time=processing_time)
            
            metrics.record_external_service_call(
                "embeddings", 
                "embed_chunk", 
                "error", 
                processing_time
            )
            raise
    
    async def embed_query(self, query: str, query_type: str = "semantic") -> np.ndarray:
        """
        Generate embedding for search query.
        
        Args:
            query: Search query text
            query_type: Type of query ("semantic" or "symbolic")
            
        Returns:
            Query embedding
        """
        provider = self.semantic_provider
        if query_type == "symbolic" and self.symbolic_provider:
            provider = self.symbolic_provider
        
        return await self._embed_with_cache(
            provider,
            query,
            cache_key=f"query_{query_type}_{hash(query)}"
        )
    
    async def _embed_with_cache(self, 
                               provider: EmbeddingProvider, 
                               text: str, 
                               cache_key: str) -> np.ndarray:
        """Embed text with optional caching."""
        if self.enable_caching and cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = await provider.embed_text(text)
        
        if self.enable_caching:
            self.cache[cache_key] = embedding
            
            # Limit cache size
            if len(self.cache) > 10000:
                # Remove oldest 20%
                keys_to_remove = list(self.cache.keys())[:2000]
                for key in keys_to_remove:
                    del self.cache[key]
        
        return embedding
    
    def _build_symbolic_context(self, 
                               components: List[AutomotiveComponent],
                               nets: List[ElectricalNet],
                               vehicle_context: Optional[str]) -> str:
        """Build symbolic context string for embedding."""
        context_parts = []
        
        # Vehicle context
        if vehicle_context:
            context_parts.append(f"vehicle {vehicle_context}")
        
        # Component contexts
        if components:
            for comp in components[:5]:  # Limit to avoid too long context
                comp_context = automotive_ontology.get_component_context(comp)
                context_parts.append(comp_context)
        
        # Net contexts  
        if nets:
            for net in nets[:5]:  # Limit to avoid too long context
                net_context = automotive_ontology.get_net_context(net)
                context_parts.append(net_context)
        
        return " ".join(context_parts)
    
    def get_semantic_dimension(self) -> int:
        """Get semantic embedding dimension."""
        return self.semantic_provider.get_dimension()
    
    def get_symbolic_dimension(self) -> Optional[int]:
        """Get symbolic embedding dimension."""
        if self.symbolic_provider:
            return self.symbolic_provider.get_dimension()
        return None
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.enable_caching,
            "semantic_model": self.semantic_provider.get_model_name(),
            "symbolic_model": self.symbolic_provider.get_model_name() if self.symbolic_provider else None
        }


class AutomotiveTextProcessor:
    """Process and prepare automotive text for embedding."""
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Clean OCR text for better embedding quality."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Fix common OCR errors in automotive context
        replacements = {
            "0hm": "ohm",
            "1K": "1k",
            "uF": "µF",
            "pF": "pF",
            "mH": "mH",
            "V DC": "VDC",
            "V AC": "VAC",
            "A DC": "ADC",
            "A AC": "AAC",
        }
        
        for wrong, correct in replacements.items():
            cleaned = cleaned.replace(wrong, correct)
        
        return cleaned
    
    @staticmethod
    def extract_technical_terms(text: str) -> List[str]:
        """Extract automotive technical terms from text."""
        import re
        
        # Patterns for automotive electrical terms
        patterns = [
            r"\b[A-Z]+\d+[A-Z]*\b",  # Component IDs: R1, C25, IC1A
            r"\b\d+[kKmM]?[Ω\u03A9]?\b",  # Resistance: 10k, 1M, 100Ω
            r"\b\d+[\.,]?\d*[uµnpmk]?[FH]\b",  # Capacitance/Inductance: 100nF, 1µH
            r"\b\d+[\.,]?\d*[Vv]\b",  # Voltage: 12V, 5.0v
            r"\b\d+[\.,]?\d*[Aa]\b",  # Current: 10A, 2.5a  
            r"\bG-?\d+\b",  # Ground points: G1, G-102
            r"\bF-?\d+\b",  # Fuses: F1, F-10
            r"\bK-?\d+\b",  # Relays: K1, K-5
            r"\b(?:CAN|LIN|K-?LINE)\b",  # Bus types
            r"\b(?:ECM|PCM|ECU|TCM|BCM)\b",  # Control modules
        ]
        
        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))  # Remove duplicates
    
    @staticmethod
    def create_layout_chunks(text_spans: List[Dict], max_chunk_size: int = 512) -> List[str]:
        """Create layout-aware text chunks from OCR spans."""
        if not text_spans:
            return []
        
        # Sort by page, then by y-coordinate, then by x-coordinate
        sorted_spans = sorted(text_spans, key=lambda s: (
            s.get('page', 0),
            s.get('bbox', [0, 0, 0, 0])[1],  # y1
            s.get('bbox', [0, 0, 0, 0])[0]   # x1
        ))
        
        chunks = []
        current_chunk = []
        current_length = 0
        current_page = None
        
        for span in sorted_spans:
            text = span.get('text', '').strip()
            if not text:
                continue
            
            page = span.get('page', 0)
            
            # Start new chunk if page changes or size limit reached
            if (current_page is not None and page != current_page) or \
               (current_length + len(text) > max_chunk_size):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(text)
            current_length += len(text) + 1  # +1 for space
            current_page = page
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


# Global embedder instance (lazy initialization)
_global_embedder: Optional[AutomotiveEmbedder] = None

def get_embedder() -> AutomotiveEmbedder:
    """Get global automotive embedder instance."""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = AutomotiveEmbedder()
    return _global_embedder


def set_embedder(embedder: AutomotiveEmbedder):
    """Set global automotive embedder instance."""
    global _global_embedder
    _global_embedder = embedder