"""
Embedding generation service for converting text into vector representations.
Supports multiple embedding models and caching for performance optimization.
"""

import asyncio
import hashlib
import pickle
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import redis.asyncio as redis
from functools import lru_cache
import time

from ..config.settings import get_settings
from ..core.logging import get_logger
from ..models.component_models import ElectricalComponent

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating and caching text embeddings."""
    
    def __init__(self):
        self.sentence_transformer = None
        self.redis_client = None
        self.cache_prefix = "embedding:"
        self.model_cache = {}
        
    async def initialize(self) -> None:
        """Initialize embedding models and cache connections."""
        logger.logger.info("Initializing embedding service")
        
        # Initialize Redis cache
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.logger.info("Redis cache connection established")
        except Exception as e:
            logger.logger.warning("Redis cache unavailable", error=str(e))
            self.redis_client = None
        
        # Load sentence transformer model
        try:
            await self._load_sentence_transformer()
            logger.logger.info("Sentence transformer model loaded", 
                             model=settings.sentence_transformers_model)
        except Exception as e:
            logger.logger.error("Failed to load sentence transformer", error=str(e))
            raise
        
        # Initialize OpenAI if API key is provided
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            logger.logger.info("OpenAI API initialized")
    
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def _load_sentence_transformer(self) -> None:
        """Load sentence transformer model in executor to avoid blocking."""
        loop = asyncio.get_event_loop()
        self.sentence_transformer = await loop.run_in_executor(
            None, 
            SentenceTransformer, 
            settings.sentence_transformers_model
        )
    
    async def generate_embedding(self, text: str, model: str = "sentence_transformer",
                               use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        if use_cache:
            cached_embedding = await self._get_cached_embedding(text, model)
            if cached_embedding is not None:
                return cached_embedding
        
        start_time = time.time()
        
        # Generate embedding based on model type
        if model == "sentence_transformer":
            embedding = await self._generate_sentence_transformer_embedding(text)
        elif model == "openai":
            embedding = await self._generate_openai_embedding(text)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Cache the result
        if use_cache:
            await self._cache_embedding(text, model, embedding)
        
        # Log metrics
        logger.log_embedding_generation(
            text_length=len(text),
            model_name=model,
            generation_time_ms=generation_time_ms,
            embedding_dim=len(embedding)
        )
        
        return embedding
    
    async def generate_embeddings_batch(self, texts: List[str], 
                                      model: str = "sentence_transformer",
                                      batch_size: int = 32,
                                      use_cache: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        if not valid_texts:
            return [[0.0] * 768] * len(texts)  # Return zero vectors for empty texts
        
        # Check cache for existing embeddings
        cached_results = {}
        uncached_texts = []
        
        if use_cache:
            for i, text in valid_texts:
                cached = await self._get_cached_embedding(text, model)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    uncached_texts.append((i, text))
        else:
            uncached_texts = valid_texts
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            
            if model == "sentence_transformer":
                new_embeddings = await self._generate_sentence_transformer_embeddings_batch(
                    [text for _, text in uncached_texts], batch_size
                )
            elif model == "openai":
                new_embeddings = await self._generate_openai_embeddings_batch(
                    [text for _, text in uncached_texts]
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Cache new embeddings and add to results
            for (i, text), embedding in zip(uncached_texts, new_embeddings):
                cached_results[i] = embedding
                if use_cache:
                    await self._cache_embedding(text, model, embedding)
            
            # Log batch metrics
            logger.log_embedding_generation(
                text_length=sum(len(text) for _, text in uncached_texts),
                model_name=f"{model}_batch_{len(uncached_texts)}",
                generation_time_ms=generation_time_ms,
                embedding_dim=len(new_embeddings[0]) if new_embeddings else 0
            )
        
        # Reconstruct results in original order
        results = []
        for i in range(len(texts)):
            if i in cached_results:
                results.append(cached_results[i])
            else:
                # Empty text, return zero vector
                results.append([0.0] * 768)
        
        return results
    
    async def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformer."""
        if not self.sentence_transformer:
            raise RuntimeError("Sentence transformer not initialized")
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            self.sentence_transformer.encode, 
            text
        )
        return embedding.tolist()
    
    async def _generate_sentence_transformer_embeddings_batch(self, 
                                                            texts: List[str], 
                                                            batch_size: int) -> List[List[float]]:
        """Generate embeddings in batches using sentence transformer."""
        if not self.sentence_transformer:
            raise RuntimeError("Sentence transformer not initialized")
        
        all_embeddings = []
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await loop.run_in_executor(
                None, 
                self.sentence_transformer.encode, 
                batch
            )
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")
        
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API in batch."""
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")
        
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item['embedding'] for item in response['data']]
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.cache_prefix}{model}:{text_hash}"
    
    async def _get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve embedding from cache."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(text, model)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_embedding(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(text, model)
            cached_data = pickle.dumps(embedding)
            
            await self.redis_client.setex(
                cache_key, 
                settings.cache_ttl_seconds, 
                cached_data
            )
        except Exception as e:
            logger.logger.warning("Cache storage failed", error=str(e))
    
    def build_component_text(self, component: ElectricalComponent) -> str:
        """Build searchable text representation of a component."""
        text_parts = [
            component.canonical_id,
            component.description,
            component.component_type.value,
            component.anchor_zone.value
        ]
        
        # Add technical specifications
        specs = component.specifications
        if specs.voltage_rating:
            text_parts.append(f"{specs.voltage_rating}V")
        if specs.current_rating:
            text_parts.append(f"{specs.current_rating}A")
        if specs.power_rating:
            text_parts.append(f"{specs.power_rating}W")
        
        # Add manufacturer and part info
        if component.manufacturer:
            text_parts.append(component.manufacturer)
        if component.part_number:
            text_parts.append(component.part_number)
        
        # Add function and purpose
        if component.function:
            text_parts.append(component.function)
        if component.purpose:
            text_parts.append(component.purpose)
        
        # Add categories and tags
        text_parts.extend(component.categories)
        text_parts.extend(component.tags)
        text_parts.extend(component.keywords)
        
        # Filter out None values and join
        return " ".join(str(part) for part in text_parts if part)
    
    def build_documentation_text(self, title: str, content: str, 
                                metadata: Dict[str, Any]) -> str:
        """Build searchable text representation of documentation."""
        text_parts = [title, content]
        
        # Add metadata context
        if metadata.get('component_types'):
            text_parts.extend(metadata['component_types'])
        if metadata.get('vehicle_models'):
            text_parts.extend(metadata['vehicle_models'])
        if metadata.get('keywords'):
            text_parts.extend(metadata['keywords'])
        
        return " ".join(str(part) for part in text_parts if part)
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        stats = {
            "models_loaded": {
                "sentence_transformer": self.sentence_transformer is not None,
                "openai": settings.openai_api_key is not None
            },
            "cache_available": self.redis_client is not None,
            "model_name": settings.sentence_transformers_model
        }
        
        if self.redis_client:
            try:
                cache_info = await self.redis_client.info("memory")
                stats["cache_memory_mb"] = cache_info.get("used_memory", 0) / (1024 * 1024)
                
                # Count cached embeddings
                keys = await self.redis_client.keys(f"{self.cache_prefix}*")
                stats["cached_embeddings"] = len(keys)
            except Exception as e:
                logger.logger.warning("Failed to get cache stats", error=str(e))
        
        return stats