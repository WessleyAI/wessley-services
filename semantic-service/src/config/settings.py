"""
Configuration settings for the Semantic Search Service.
Manages environment variables, database connections, and service parameters.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service Configuration
    service_name: str = "semantic-search-service"
    version: str = "1.0.0"
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=3003, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    workers: int = Field(default=4, env="WORKERS")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "https://*.wessley.ai"], 
        env="ALLOWED_ORIGINS"
    )
    
    # Vector Database (Qdrant)
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_size: int = Field(default=100000, env="QDRANT_COLLECTION_SIZE")
    
    # Embedding Models
    sentence_transformers_model: str = Field(
        default="all-MiniLM-L6-v2", 
        env="SENTENCE_TRANSFORMERS_MODEL"
    )
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Search Configuration
    default_search_limit: int = Field(default=20, env="DEFAULT_SEARCH_LIMIT")
    max_search_limit: int = Field(default=100, env="MAX_SEARCH_LIMIT")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Cache Configuration (Redis)
    redis_url: str = Field(default="redis://localhost:6379/2", env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    # External Services
    neo4j_url: str = Field(default="bolt://localhost:7687", env="NEO4J_URL")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    
    model_service_url: str = Field(
        default="http://localhost:3001", 
        env="MODEL_SERVICE_URL"
    )
    learning_service_url: str = Field(
        default="http://localhost:3002", 
        env="LEARNING_SERVICE_URL"
    )
    
    # Indexing Configuration
    enable_hybrid_search: bool = Field(default=True, env="ENABLE_HYBRID_SEARCH")
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    indexing_workers: int = Field(default=2, env="INDEXING_WORKERS")
    
    # Monitoring & Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8080, env="METRICS_PORT")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    service_api_keys: List[str] = Field(default=[], env="SERVICE_API_KEYS")
    
    # Performance Tuning
    embedding_cache_size: int = Field(default=10000, env="EMBEDDING_CACHE_SIZE")
    vector_search_timeout: int = Field(default=30, env="VECTOR_SEARCH_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()