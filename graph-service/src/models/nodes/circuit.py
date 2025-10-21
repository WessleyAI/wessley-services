"""
Circuit node model for electrical circuits
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class CircuitNode(BaseModel):
    """
    Circuit node model matching Neo4j Circuit schema
    
    Neo4j Schema:
    (:Circuit {
      id: "string",
      vehicle_signature: "string",
      name: "string",
      voltage: "float",
      max_current: "float",
      circuit_type: "string",
      protection: "string",
      created_at: "datetime",
      updated_at: "datetime"
    })
    """
    
    # Primary identifiers
    id: str = Field(..., description="Unique circuit identifier")
    vehicle_signature: str = Field(..., description="Vehicle isolation key")
    
    # Circuit information
    name: str = Field(..., description="Circuit name (e.g., 'Headlight Circuit')")
    voltage: float = Field(..., description="Operating voltage")
    max_current: float = Field(..., description="Maximum current capacity")
    circuit_type: str = Field(default="power", description="Circuit type: power, signal, ground")
    protection: Optional[str] = Field(None, description="Fuse/breaker type and rating")
    
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
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'CircuitNode':
        """Create instance from Neo4j result dictionary"""
        # Convert datetime strings
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class CircuitFilter(BaseModel):
    """Filter criteria for circuit queries"""
    vehicle_signature: Optional[str] = None
    circuit_type: Optional[List[str]] = None
    voltage_range: Optional[Dict[str, float]] = None
    current_range: Optional[Dict[str, float]] = None
    protection_type: Optional[str] = None


class CircuitUpdate(BaseModel):
    """Model for circuit updates"""
    name: Optional[str] = None
    voltage: Optional[float] = None
    max_current: Optional[float] = None
    circuit_type: Optional[str] = None
    protection: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CircuitCreate(BaseModel):
    """Model for creating new circuits"""
    vehicle_signature: str
    name: str
    voltage: float
    max_current: float
    circuit_type: str = "power"
    protection: Optional[str] = None


class CircuitAnalysis(BaseModel):
    """Circuit analysis result model"""
    circuit: CircuitNode
    component_count: int
    total_load: float
    load_percentage: float
    power_sources: List[str]
    critical_components: List[str]
    redundancy_level: str
    fault_points: List[str]