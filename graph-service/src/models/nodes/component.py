"""
Component node model for electrical components in the system
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class Position3D(BaseModel):
    """3D position coordinates"""
    x: float
    y: float
    z: float


class ComponentSpecifications(BaseModel):
    """Component technical specifications"""
    voltage_rating: Optional[float] = None
    current_rating: Optional[float] = None
    power_rating: Optional[float] = None
    resistance: Optional[str] = None
    capacitance: Optional[str] = None
    frequency_range: Optional[str] = None
    # Extensible for component-specific specs
    additional_specs: Dict[str, Any] = Field(default_factory=dict)


class ComponentNode(BaseModel):
    """
    Component node model matching Neo4j Component schema
    
    Neo4j Schema:
    (:Component {
      id: "string",
      vehicle_signature: "string",
      type: "string",
      name: "string",
      part_number: "string",
      manufacturer: "string",
      specifications: "map",
      position: "map",
      zone: "string",
      voltage_rating: "float",
      current_rating: "float",
      created_at: "datetime",
      updated_at: "datetime"
    })
    """
    
    # Primary identifiers
    id: str = Field(..., description="Unique component identifier")
    vehicle_signature: str = Field(..., description="Vehicle isolation key")
    
    # Component information
    type: str = Field(..., description="Component type: relay, fuse, sensor, etc.")
    name: str = Field(..., description="Human-readable component name")
    part_number: Optional[str] = Field(None, description="Manufacturer part number")
    manufacturer: Optional[str] = Field(None, description="Component manufacturer")
    
    # Technical specifications
    specifications: ComponentSpecifications = Field(
        default_factory=ComponentSpecifications,
        description="Technical specifications"
    )
    
    # Spatial information
    position: Optional[Position3D] = Field(None, description="3D position coordinates")
    zone: Optional[str] = Field(None, description="Physical location zone")
    
    # Electrical characteristics (denormalized for quick access)
    voltage_rating: Optional[float] = Field(None, description="Voltage rating (V)")
    current_rating: Optional[float] = Field(None, description="Current rating (A)")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j property dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert position to list format for Neo4j
        if self.position:
            data['position'] = [self.position.x, self.position.y, self.position.z]
        
        # Convert specifications to flat dict
        if self.specifications:
            specs_dict = self.specifications.dict(exclude_none=True)
            data['specifications'] = specs_dict
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'ComponentNode':
        """Create instance from Neo4j result dictionary"""
        # Convert position from list to Position3D
        if 'position' in data and isinstance(data['position'], list):
            if len(data['position']) >= 3:
                data['position'] = Position3D(
                    x=data['position'][0],
                    y=data['position'][1], 
                    z=data['position'][2]
                )
        
        # Convert specifications dict to ComponentSpecifications
        if 'specifications' in data and isinstance(data['specifications'], dict):
            data['specifications'] = ComponentSpecifications(**data['specifications'])
        
        # Convert datetime strings
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class ComponentFilter(BaseModel):
    """Filter criteria for component queries"""
    vehicle_signature: Optional[str] = None
    component_type: Optional[List[str]] = None
    zone: Optional[str] = None
    manufacturer: Optional[str] = None
    has_position: Optional[bool] = None
    voltage_range: Optional[Dict[str, float]] = None  # {"min": 12.0, "max": 24.0}
    current_range: Optional[Dict[str, float]] = None


class ComponentUpdate(BaseModel):
    """Model for component updates"""
    name: Optional[str] = None
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    specifications: Optional[ComponentSpecifications] = None
    position: Optional[Position3D] = None
    zone: Optional[str] = None
    voltage_rating: Optional[float] = None
    current_rating: Optional[float] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ComponentCreate(BaseModel):
    """Model for creating new components"""
    vehicle_signature: str
    type: str
    name: str
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    specifications: ComponentSpecifications = Field(default_factory=ComponentSpecifications)
    position: Optional[Position3D] = None
    zone: Optional[str] = None
    voltage_rating: Optional[float] = None
    current_rating: Optional[float] = None