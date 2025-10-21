"""
CONNECTS_TO relationship model for physical electrical connections
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ConnectsToRelationship(BaseModel):
    """
    CONNECTS_TO relationship model matching Neo4j schema
    
    Neo4j Schema:
    (:Component)-[:CONNECTS_TO {
      wire_gauge: "string",
      wire_color: "string", 
      wire_length: "float",
      connection_type: "string",
      signal_type: "string",
      created_at: "datetime"
    }]->(:Component)
    """
    
    # Wire specifications
    wire_gauge: Optional[str] = Field(None, description="Wire gauge (e.g., '2.5mm²')")
    wire_color: Optional[str] = Field(None, description="Wire color code")
    wire_length: Optional[float] = Field(None, description="Estimated length (mm)")
    
    # Connection characteristics
    connection_type: str = Field(
        default="direct",
        description="Connection type: direct, via_connector, splice"
    )
    signal_type: str = Field(
        default="power", 
        description="Signal type: power, ground, signal, data"
    )
    
    # Additional connection properties
    impedance: Optional[float] = Field(None, description="Wire impedance (ohms)")
    voltage_drop: Optional[float] = Field(None, description="Voltage drop (V)")
    current_capacity: Optional[float] = Field(None, description="Current capacity (A)")
    shielded: Optional[bool] = Field(None, description="Is wire shielded")
    twisted_pair: Optional[bool] = Field(None, description="Is twisted pair")
    
    # Quality and condition
    condition: Optional[str] = Field(None, description="good, fair, poor")
    insulation_type: Optional[str] = Field(None, description="Insulation material")
    connector_type: Optional[str] = Field(None, description="Connector used")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties dictionary"""
        data = self.dict(exclude_none=True)
        
        # Convert datetime to string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
            
        return data
    
    @classmethod
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'ConnectsToRelationship':
        """Create instance from Neo4j relationship dictionary"""
        # Convert datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def calculate_resistance(self) -> Optional[float]:
        """Calculate wire resistance based on gauge and length"""
        # Wire resistance lookup table (ohms per meter)
        resistance_table = {
            "0.5mm²": 0.036,
            "0.75mm²": 0.024,
            "1.0mm²": 0.018,
            "1.5mm²": 0.012,
            "2.5mm²": 0.007,
            "4.0mm²": 0.0045,
            "6.0mm²": 0.003,
            "10.0mm²": 0.0018,
        }
        
        if self.wire_gauge and self.wire_length:
            resistance_per_meter = resistance_table.get(self.wire_gauge)
            if resistance_per_meter:
                return resistance_per_meter * (self.wire_length / 1000)  # Convert mm to meters
        
        return None
    
    def is_power_connection(self) -> bool:
        """Check if this is a power connection"""
        return self.signal_type in ["power", "ground"]
    
    def is_data_connection(self) -> bool:
        """Check if this is a data connection"""
        return self.signal_type in ["data", "signal"]


class ConnectionPath(BaseModel):
    """Model for representing a complete connection path"""
    start_component_id: str
    end_component_id: str
    path_segments: list[ConnectsToRelationship]
    total_length: float
    total_resistance: float
    path_type: str  # direct, multi_hop, redundant
    reliability_score: float


class WireHarness(BaseModel):
    """Model for wire harness grouping"""
    harness_id: str
    connections: list[ConnectsToRelationship]
    routing_points: list[Dict[str, float]]  # 3D points
    bundle_diameter: Optional[float] = None
    protection_type: Optional[str] = None  # conduit, loom, tape
    bend_radius: Optional[float] = None
    strain_relief: Optional[bool] = None