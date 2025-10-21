"""
Pydantic models for search operations and general service responses.
Defines schemas for search queries, responses, and system operations.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SearchType(str, Enum):
    """Types of search operations."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


class CollectionName(str, Enum):
    """Available Qdrant collections."""
    COMPONENTS = "components"
    DOCUMENTATION = "documentation"
    KG_ENTITIES = "kg_entities"


class QueryIntent(str, Enum):
    """Detected intent from user queries."""
    FIND_COMPONENT = "find_component"
    LOCATE_COMPONENT = "locate_component"
    TROUBLESHOOT = "troubleshoot"
    COMPARE_COMPONENTS = "compare_components"
    GET_SPECIFICATIONS = "get_specifications"
    FIND_SIMILAR = "find_similar"
    GENERAL_QUESTION = "general_question"


class SearchFilter(BaseModel):
    """Generic search filter model."""
    field: str = Field(..., description="Field name to filter on")
    operator: str = Field(..., description="Filter operator (eq, gt, lt, in, contains)")
    value: Union[str, int, float, List[Any]] = Field(..., description="Filter value(s)")


class UniversalSearchQuery(BaseModel):
    """Universal search query model for all search types."""
    
    # Core Query
    query: str = Field(..., description="Search query text")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    collections: List[CollectionName] = Field(
        default=[CollectionName.COMPONENTS], 
        description="Collections to search in"
    )
    
    # Context
    vehicle_signature: Optional[str] = Field(None, description="Vehicle context for search")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User session context")
    spatial_context: Optional[Dict[str, Any]] = Field(None, description="3D model spatial context")
    
    # Filters
    filters: List[SearchFilter] = Field(default_factory=list, description="Search filters")
    
    # Search Parameters
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results per collection")
    similarity_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum similarity score")
    boost_factors: Dict[str, float] = Field(
        default_factory=lambda: {"exact_match": 2.0, "semantic": 1.0}, 
        description="Score boost factors"
    )
    
    # Query Enhancement
    expand_query: bool = Field(default=True, description="Enable query expansion")
    include_synonyms: bool = Field(default=True, description="Include technical synonyms")
    rerank_results: bool = Field(default=True, description="Enable result re-ranking")
    
    # Response Options
    include_explanations: bool = Field(default=True, description="Include match explanations")
    include_highlights: bool = Field(default=True, description="Include text highlights")
    include_related: bool = Field(default=False, description="Include related items")


class SearchResult(BaseModel):
    """Individual search result."""
    
    # Core Result Data
    id: str = Field(..., description="Result item ID")
    collection: CollectionName = Field(..., description="Source collection")
    content: Dict[str, Any] = Field(..., description="Result content/payload")
    
    # Scoring
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    rank: int = Field(..., description="Result ranking position")
    
    # Matching Details
    match_type: str = Field(..., description="Type of match (exact, semantic, fuzzy)")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched")
    highlights: Dict[str, List[str]] = Field(default_factory=dict, description="Highlighted text snippets")
    
    # Additional Context
    explanation: Optional[str] = Field(None, description="Why this result was returned")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Result confidence")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")


class SearchResponse(BaseModel):
    """Complete search response."""
    
    # Query Information
    query: str = Field(..., description="Original search query")
    search_type: SearchType = Field(..., description="Search type used")
    processed_query: str = Field(..., description="Processed/expanded query")
    detected_intent: Optional[QueryIntent] = Field(None, description="Detected user intent")
    
    # Results
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total results found")
    results_per_collection: Dict[str, int] = Field(..., description="Results count per collection")
    
    # Performance Metrics
    search_time_ms: float = Field(..., description="Total search time")
    embedding_time_ms: Optional[float] = Field(None, description="Embedding generation time")
    vector_search_time_ms: Optional[float] = Field(None, description="Vector search time")
    rerank_time_ms: Optional[float] = Field(None, description="Re-ranking time")
    
    # Enhancement Data
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions")
    related_queries: List[str] = Field(default_factory=list, description="Related search queries")
    filters_applied: List[SearchFilter] = Field(default_factory=list, description="Applied filters")
    
    # Context
    search_context: Dict[str, Any] = Field(default_factory=dict, description="Search context used")


class ChatEnhancementRequest(BaseModel):
    """Request for enhancing chat context with search results."""
    
    user_query: str = Field(..., description="User's chat query")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous chat messages")
    
    # Context
    model_metadata: Optional[Dict[str, Any]] = Field(None, description="Current 3D model context")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User session context")
    vehicle_signature: Optional[str] = Field(None, description="Current vehicle context")
    
    # Enhancement Options
    max_components: int = Field(default=5, ge=1, le=20, description="Max components to include")
    max_documentation: int = Field(default=3, ge=1, le=10, description="Max docs to include")
    include_spatial: bool = Field(default=True, description="Include spatial context")
    include_relationships: bool = Field(default=True, description="Include component relationships")


class ChatEnhancementResponse(BaseModel):
    """Enhanced context for chat responses."""
    
    # Original Query
    user_query: str = Field(..., description="Original user query")
    enhanced_query: str = Field(..., description="Enhanced/expanded query")
    detected_intent: Optional[QueryIntent] = Field(None, description="Detected intent")
    
    # Context Data
    relevant_components: List[SearchResult] = Field(..., description="Relevant electrical components")
    relevant_documentation: List[SearchResult] = Field(..., description="Relevant documentation")
    spatial_context: Optional[Dict[str, Any]] = Field(None, description="3D spatial context")
    relationship_context: Optional[Dict[str, Any]] = Field(None, description="Component relationships")
    
    # Suggestions
    suggested_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    related_topics: List[str] = Field(default_factory=list, description="Related topics")
    
    # Performance
    enhancement_time_ms: float = Field(..., description="Context enhancement time")


class AnalyticsQuery(BaseModel):
    """Query for search analytics data."""
    
    start_date: Optional[datetime] = Field(None, description="Analytics start date")
    end_date: Optional[datetime] = Field(None, description="Analytics end date")
    user_id: Optional[str] = Field(None, description="Filter by user")
    vehicle_signature: Optional[str] = Field(None, description="Filter by vehicle")
    query_pattern: Optional[str] = Field(None, description="Search for query patterns")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")


class SearchAnalytics(BaseModel):
    """Search analytics response."""
    
    # Query Statistics
    total_searches: int = Field(..., description="Total search count")
    unique_queries: int = Field(..., description="Unique query count")
    average_response_time_ms: float = Field(..., description="Average response time")
    
    # Popular Data
    popular_queries: List[Dict[str, Any]] = Field(..., description="Most popular queries")
    popular_components: List[Dict[str, Any]] = Field(..., description="Most searched components")
    popular_vehicles: List[Dict[str, Any]] = Field(..., description="Most searched vehicles")
    
    # Performance Metrics
    search_success_rate: float = Field(..., description="Percentage of successful searches")
    average_results_per_query: float = Field(..., description="Average results returned")
    user_satisfaction_metrics: Dict[str, float] = Field(..., description="User interaction metrics")
    
    # Time-based Data
    search_volume_over_time: List[Dict[str, Any]] = Field(..., description="Search volume trends")
    response_time_trends: List[Dict[str, Any]] = Field(..., description="Response time trends")


class ServiceHealthResponse(BaseModel):
    """Service health check response."""
    
    # Service Status
    service: str = Field(default="semantic-search-service", description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check time")
    
    # Database Connections
    qdrant_status: str = Field(..., description="Qdrant connection status")
    neo4j_status: str = Field(..., description="Neo4j connection status") 
    redis_status: str = Field(..., description="Redis connection status")
    
    # Performance Metrics
    active_connections: int = Field(..., description="Active database connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    
    # Service Metrics
    total_collections: int = Field(..., description="Total Qdrant collections")
    total_vectors: int = Field(..., description="Total vectors stored")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    
    # Recent Activity
    searches_last_hour: int = Field(..., description="Searches in last hour")
    indexing_operations_active: int = Field(..., description="Active indexing operations")
    average_response_time_ms: float = Field(..., description="Average response time")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid search query parameters",
                "details": {"field": "similarity_threshold", "issue": "must be between 0 and 1"},
                "timestamp": "2023-10-21T12:00:00Z",
                "request_id": "req_123456"
            }
        }