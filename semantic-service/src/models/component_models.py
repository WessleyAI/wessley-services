"""
Pydantic models for electrical component data structures.
Defines the schema for components, specifications, and metadata.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ComponentType(str, Enum):
    """Standard electrical component types."""
    BATTERY = "battery"
    ALTERNATOR = "alternator"
    STARTER = "starter"
    ECU = "ecu"
    RELAY = "relay"
    FUSE = "fuse"
    SWITCH = "switch"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CONNECTOR = "connector"
    WIRE = "wire"
    GROUND_POINT = "ground_point"
    GROUND_PLANE = "ground_plane"
    LIGHT = "light"
    MOTOR = "motor"
    PUMP = "pump"
    INJECTOR = "injector"
    COIL = "coil"


class AnchorZone(str, Enum):
    """Physical zones where components are located."""
    ENGINE_COMPARTMENT = "Engine Compartment"
    DASH_PANEL = "Dash Panel"
    REAR_CARGO = "Rear Cargo/Tailgate"
    CHASSIS = "Chassis"
    CABIN = "Cabin"
    EXTERIOR = "Exterior"


class ComponentSpecification(BaseModel):
    """Technical specifications for electrical components."""
    voltage_rating: Optional[float] = Field(None, description="Operating voltage in volts")
    current_rating: Optional[float] = Field(None, description="Maximum current in amperes")
    power_rating: Optional[float] = Field(None, description="Power consumption in watts")
    resistance: Optional[float] = Field(None, description="Resistance in ohms")
    capacitance: Optional[float] = Field(None, description="Capacitance in farads")
    inductance: Optional[float] = Field(None, description="Inductance in henries")
    frequency_range: Optional[List[float]] = Field(None, description="Operating frequency range in Hz")
    temperature_range: Optional[List[float]] = Field(None, description="Operating temperature range in Celsius")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Physical dimensions in mm")
    weight: Optional[float] = Field(None, description="Weight in grams")
    material: Optional[str] = Field(None, description="Primary material composition")
    protection_rating: Optional[str] = Field(None, description="IP protection rating")
    certification: Optional[List[str]] = Field(None, description="Safety/quality certifications")


class ComponentMetadata(BaseModel):
    """Metadata about component data source and indexing."""
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(..., description="Data source (neo4j, manual, import)")
    version: str = Field(default="1.0", description="Data schema version")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Data confidence score")
    last_verified: Optional[datetime] = Field(None, description="Last verification timestamp")
    verification_source: Optional[str] = Field(None, description="Who/what verified this data")


class ElectricalComponent(BaseModel):
    """Complete electrical component model."""
    
    # Core Identity
    component_id: str = Field(..., description="Unique component identifier")
    vehicle_signature: str = Field(..., description="Vehicle model signature")
    canonical_id: str = Field(..., description="Human-readable component name")
    code_id: str = Field(..., description="Short technical code")
    
    # Classification
    component_type: ComponentType = Field(..., description="Component category")
    node_type: str = Field(..., description="Graph node type")
    anchor_zone: AnchorZone = Field(..., description="Physical location zone")
    
    # Description & Documentation
    description: str = Field(..., description="Detailed component description")
    function: Optional[str] = Field(None, description="Primary function description")
    purpose: Optional[str] = Field(None, description="Why this component exists")
    
    # Technical Details
    manufacturer: Optional[str] = Field(None, description="Component manufacturer")
    part_number: Optional[str] = Field(None, description="Manufacturer part number")
    model_number: Optional[str] = Field(None, description="Model identifier")
    specifications: ComponentSpecification = Field(default_factory=ComponentSpecification)
    
    # Relationships
    connected_to: List[str] = Field(default_factory=list, description="Connected component IDs")
    part_of_circuit: Optional[str] = Field(None, description="Circuit this component belongs to")
    subsystem: Optional[str] = Field(None, description="Electrical subsystem")
    
    # Categories & Tags
    categories: List[str] = Field(default_factory=list, description="Component categories")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    
    # Spatial Information
    position_3d: Optional[Dict[str, float]] = Field(None, description="3D coordinates in model")
    orientation_3d: Optional[Dict[str, float]] = Field(None, description="3D orientation")
    bounding_box: Optional[Dict[str, Any]] = Field(None, description="3D bounding box")
    
    # Metadata
    metadata: ComponentMetadata = Field(default_factory=ComponentMetadata)
    
    class Config:
        use_enum_values = True


class ComponentSearchQuery(BaseModel):
    """Model for component search requests."""
    query: str = Field(..., description="Natural language search query")
    vehicle_signature: Optional[str] = Field(None, description="Filter by vehicle model")
    component_types: Optional[List[ComponentType]] = Field(None, description="Filter by component types")
    anchor_zones: Optional[List[AnchorZone]] = Field(None, description="Filter by location zones")
    voltage_range: Optional[Dict[str, float]] = Field(None, description="Voltage rating filter")
    current_range: Optional[Dict[str, float]] = Field(None, description="Current rating filter")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    include_similar: bool = Field(default=False, description="Include similar components")
    similarity_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum similarity score")


class ComponentSearchResult(BaseModel):
    """Model for component search results."""
    component: ElectricalComponent = Field(..., description="Found component")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity to query")
    match_explanation: str = Field(..., description="Why this component matched")
    highlighted_fields: List[str] = Field(default_factory=list, description="Fields that matched query")


class ComponentSearchResponse(BaseModel):
    """Complete search response model."""
    query: str = Field(..., description="Original search query")
    results: List[ComponentSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of matches")
    search_time_ms: float = Field(..., description="Search execution time")
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class ComponentRecommendation(BaseModel):
    """Model for component recommendations."""
    component: ElectricalComponent = Field(..., description="Recommended component")
    recommendation_score: float = Field(..., ge=0, le=1, description="Recommendation strength")
    reason: str = Field(..., description="Why this was recommended")
    similarity_type: str = Field(..., description="Type of similarity (functional, electrical, etc.)")


class ComponentRecommendationResponse(BaseModel):
    """Response model for component recommendations."""
    source_component_id: str = Field(..., description="Component that recommendations are based on")
    recommendations: List[ComponentRecommendation] = Field(..., description="Recommended components")
    recommendation_time_ms: float = Field(..., description="Time to generate recommendations")


class ComponentIndexRequest(BaseModel):
    """Request model for indexing components."""
    vehicle_signature: str = Field(..., description="Vehicle model to index")
    force_reindex: bool = Field(default=False, description="Force re-indexing existing data")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for processing")
    include_relationships: bool = Field(default=True, description="Include component relationships")


class ComponentIndexStatus(BaseModel):
    """Status of component indexing operation."""
    vehicle_signature: str = Field(..., description="Vehicle being indexed")
    status: str = Field(..., description="current, completed, failed")
    components_processed: int = Field(..., description="Number of components processed")
    total_components: int = Field(..., description="Total components to process")
    start_time: datetime = Field(..., description="Indexing start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")