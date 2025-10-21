"""
POWERED_BY relationship model for power supply relationships
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PoweredByRelationship(BaseModel):
    """
    POWERED_BY relationship model matching Neo4j schema
    
    Neo4j Schema:
    (:Component)-[:POWERED_BY {
      power_type: "string",
      voltage: "float",
      max_current: "float", 
      always_on: "boolean",
      created_at: "datetime"
    }]->(:Component)
    """
    
    # Power characteristics
    power_type: str = Field(
        default="battery",
        description="Power source type: battery, alternator, regulated"
    )
    voltage: float = Field(..., description="Supply voltage")
    max_current: Optional[float] = Field(None, description="Maximum current draw")
    
    # Power behavior
    always_on: bool = Field(
        default=False,
        description="Constant power vs switched"
    )
    switched_by: Optional[str] = Field(None, description="Component that controls switching")
    
    # Power quality
    ripple_voltage: Optional[float] = Field(None, description="Voltage ripple (V)")
    regulation: Optional[float] = Field(None, description="Voltage regulation percentage")
    startup_time: Optional[float] = Field(None, description="Power-on time (ms)")
    
    # Load characteristics
    load_type: Optional[str] = Field(None, description="resistive, inductive, capacitive")
    startup_current: Optional[float] = Field(None, description="Inrush current (A)")
    duty_cycle: Optional[float] = Field(None, description="Duty cycle percentage")
    
    # Protection
    fuse_rating: Optional[str] = Field(None, description="Protection fuse rating")
    overcurrent_protection: Optional[bool] = Field(None, description="Has overcurrent protection")
    reverse_polarity_protection: Optional[bool] = Field(None, description="Has reverse polarity protection")
    
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
    def from_neo4j_dict(cls, data: Dict[str, Any]) -> 'PoweredByRelationship':
        """Create instance from Neo4j relationship dictionary"""
        # Convert datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def calculate_power(self) -> Optional[float]:
        """Calculate power consumption in watts"""
        if self.voltage and self.max_current:
            return self.voltage * self.max_current
        return None
    
    def is_critical_power(self) -> bool:
        """Check if this is critical/always-on power"""
        return self.always_on or self.power_type in ["battery", "alternator"]
    
    def get_power_efficiency(self) -> Optional[float]:
        """Estimate power efficiency based on type"""
        efficiency_map = {
            "battery": 0.95,
            "alternator": 0.85,
            "regulated": 0.80,
            "switched": 0.90
        }
        return efficiency_map.get(self.power_type)


class PowerDistributionTree(BaseModel):
    """Model for power distribution hierarchy"""
    source_component: str
    distribution_levels: list[Dict[str, Any]]
    total_load: float
    total_efficiency: float
    critical_path: bool


class PowerBudget(BaseModel):
    """Model for power budget analysis"""
    vehicle_signature: str
    total_available_power: float
    total_consumed_power: float
    peak_power: float
    average_power: float
    power_margin: float
    power_efficiency: float
    critical_loads: list[str]
    non_critical_loads: list[str]