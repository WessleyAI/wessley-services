"""Configuration settings for graph service"""

import os
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Neo4j Database Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    # Service Configuration
    service_name: str = Field(default="graph-service", env="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Performance Settings
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    max_concurrent_queries: int = Field(default=10, env="MAX_CONCURRENT_QUERIES")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    
    # Import Settings
    max_import_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_IMPORT_FILE_SIZE")  # 50MB
    import_batch_size: int = Field(default=1000, env="IMPORT_BATCH_SIZE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Security
    allowed_origins: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()