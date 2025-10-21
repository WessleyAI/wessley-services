"""Zone node model for physical zones in the vehicle"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ZoneNode(BaseModel):
    id: str = Field(..., description="Unique zone identifier")
    vehicle_signature: str = Field(..., description="Vehicle isolation key")
    name: str = Field(..., description="Human-readable zone name")
    bounds: Optional[Dict[str, float]] = Field(None, description="3D bounds of the zone")
    access_level: str = Field(default="moderate", description="Access difficulty")
    environmental_conditions: Optional[Dict[str, Any]] = Field(None, description="Environmental conditions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'ZoneNode':
        return cls(**data)