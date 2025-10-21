"""Connector node model for electrical connectors"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ConnectorNode(BaseModel):
    id: str = Field(..., description="Unique connector identifier")
    vehicle_signature: str = Field(..., description="Vehicle isolation key")
    type: str = Field(..., description="Connector type: terminal, plug, socket")
    pin_count: Optional[int] = Field(None, description="Number of pins/terminals")
    connector_family: Optional[str] = Field(None, description="Connector standard/family")
    position: Optional[List[float]] = Field(None, description="3D position")
    pin_assignments: Optional[Dict[str, str]] = Field(None, description="Pin to function mapping")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'ConnectorNode':
        return cls(**data)