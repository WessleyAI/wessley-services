"""
Structured logging configuration for the Semantic Search Service.
Provides consistent, searchable logs across all components.
"""

import logging
import sys
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class SemanticSearchLogger:
    """Enhanced logger for semantic search operations."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_search_operation(self, query: str, collection: str, 
                           results_count: int, response_time_ms: float, 
                           user_id: str = None, session_id: str = None) -> None:
        """Log search operations with relevant context."""
        self.logger.info(
            "search_operation",
            query=query,
            collection=collection,
            results_count=results_count,
            response_time_ms=response_time_ms,
            user_id=user_id,
            session_id=session_id,
            operation_type="vector_search"
        )
    
    def log_indexing_operation(self, collection: str, documents_count: int,
                             operation_type: str, duration_ms: float,
                             success: bool = True, error: str = None) -> None:
        """Log indexing operations."""
        self.logger.info(
            "indexing_operation",
            collection=collection,
            documents_count=documents_count,
            operation_type=operation_type,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def log_embedding_generation(self, text_length: int, model_name: str,
                               generation_time_ms: float, embedding_dim: int) -> None:
        """Log embedding generation metrics."""
        self.logger.info(
            "embedding_generation",
            text_length=text_length,
            model_name=model_name,
            generation_time_ms=generation_time_ms,
            embedding_dimension=embedding_dim
        )
    
    def log_service_integration(self, target_service: str, operation: str,
                              response_time_ms: float, success: bool,
                              error: str = None) -> None:
        """Log interactions with external services."""
        self.logger.info(
            "service_integration",
            target_service=target_service,
            operation=operation,
            response_time_ms=response_time_ms,
            success=success,
            error=error
        )
    
    def log_user_interaction(self, user_id: str, session_id: str,
                           interaction_type: str, query: str,
                           results_clicked: int = 0) -> None:
        """Log user interaction patterns for ML analysis."""
        self.logger.info(
            "user_interaction",
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            query=query,
            results_clicked=results_clicked
        )


def get_logger(name: str) -> SemanticSearchLogger:
    """Get a configured logger instance."""
    return SemanticSearchLogger(name)