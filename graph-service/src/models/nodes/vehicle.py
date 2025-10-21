"""
Vehicle node model for vehicle metadata
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class VehicleMetadata(BaseModel):
    """Extended vehicle metadata"""
    engine_type: Optional[str] = None
    transmission: Optional[str] = None
    drive_type: Optional[str] = None      # FWD, RWD, AWD
    fuel_type: Optional[str] = None       # gasoline, diesel, electric, hybrid
    body_style: Optional[str] = None      # sedan, hatchback, SUV, etc.
    trim_level: Optional[str] = None
    production_year_range: Optional[Dict[str, int]] = None  # {"start": 2001, "end": 2006}
    market_regions: Optional[list] = None
    assembly_plant: Optional[str] = None
    vin_pattern: Optional[str] = None


class VehicleNode(BaseModel):
    """
    Vehicle node model matching Neo4j Vehicle schema
    
    Neo4j Schema:
    (:Vehicle {
      signature: "string",
      make: "string",
      model: "string", 
      year: "integer",
      engine: "string",
      market: "string",
      created_at: "datetime",
      metadata: "map"
    })
    """
    
    # Primary identifier
    signature: str = Field(..., description="Unique vehicle identifier")
    
    # Basic vehicle information
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., description="Model year")
    engine: Optional[str] = Field(None, description="Engine specification")
    market: Optional[str] = Field(None, description="Geographic market")
    
    # Extended metadata
    metadata: VehicleMetadata = Field(
        default_factory=VehicleMetadata,
        description="Additional vehicle data"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j property dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert metadata to flat dict
        if self.metadata:
            data['metadata'] = self.metadata.dict(exclude_none=True)
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'VehicleNode':
        """Create instance from Neo4j result dictionary"""
        # Convert metadata dict to VehicleMetadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = VehicleMetadata(**data['metadata'])
        
        # Convert datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def get_display_name(self) -> str:
        """Get formatted display name for the vehicle"""
        return f"{self.year} {self.make} {self.model}"
    
    def get_short_signature(self) -> str:
        """Get shortened signature for display"""
        return self.signature.replace('_', ' ').title()


class VehicleFilter(BaseModel):
    """Filter criteria for vehicle queries"""
    make: Optional[str] = None
    model: Optional[str] = None
    year_range: Optional[Dict[str, int]] = None
    market: Optional[str] = None
    engine_type: Optional[str] = None
    fuel_type: Optional[str] = None


class VehicleUpdate(BaseModel):
    """Model for vehicle updates"""
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    engine: Optional[str] = None
    market: Optional[str] = None
    metadata: Optional[VehicleMetadata] = None


class VehicleCreate(BaseModel):
    """Model for creating new vehicles"""
    signature: str
    make: str
    model: str
    year: int
    engine: Optional[str] = None
    market: Optional[str] = None
    metadata: VehicleMetadata = Field(default_factory=VehicleMetadata)


class VehicleStatistics(BaseModel):
    """Vehicle system statistics"""
    vehicle: VehicleNode
    total_components: int
    total_circuits: int
    total_zones: int
    total_connections: int
    component_types: Dict[str, int]
    circuit_types: Dict[str, int]
    coverage_percentage: float
    data_completeness_score: float
    last_updated: datetime


class VehicleDataQuality(BaseModel):
    """Vehicle data quality assessment"""
    vehicle_signature: str
    completeness_score: float
    components_with_position: int
    components_without_position: int
    orphaned_connectors: int
    unconnected_components: int
    circuits_without_components: int
    data_integrity_issues: List[str]
    recommendations: List[str]